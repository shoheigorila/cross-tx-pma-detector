"""Event storage using SQLite for simplicity (upgradeable to TimescaleDB)."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Optional

from .events import UnifiedEvent

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "events.db"


class EventStorage:
    """SQLite-backed storage for UnifiedEvent records.

    For production-scale usage, swap to PostgreSQL + TimescaleDB by
    replacing the connection and SQL dialect.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_hash TEXT NOT NULL,
                block_number INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                from_address TEXT NOT NULL,
                to_address TEXT NOT NULL,
                token_address TEXT NOT NULL,
                amount TEXT NOT NULL,
                token_in TEXT,
                token_out TEXT,
                amount_in TEXT,
                amount_out TEXT,
                function_selector TEXT,
                gas_price INTEGER DEFAULT 0,
                contract_address TEXT DEFAULT '',
                log_index INTEGER DEFAULT 0,
                price_impact REAL
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_block
            ON events (block_number)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_from
            ON events (from_address)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_to
            ON events (to_address)
        """)
        self.conn.commit()

    def store_events(self, events: list[UnifiedEvent]) -> int:
        """Insert a batch of events. Returns number of rows inserted."""
        rows = []
        for e in events:
            rows.append((
                e.tx_hash, e.block_number, e.timestamp, e.event_type,
                e.from_address, e.to_address, e.token_address, str(e.amount),
                e.token_in, e.token_out,
                str(e.amount_in) if e.amount_in is not None else None,
                str(e.amount_out) if e.amount_out is not None else None,
                e.function_selector, e.gas_price, e.contract_address,
                e.log_index, e.price_impact,
            ))
        self.conn.executemany("""
            INSERT INTO events (
                tx_hash, block_number, timestamp, event_type,
                from_address, to_address, token_address, amount,
                token_in, token_out, amount_in, amount_out,
                function_selector, gas_price, contract_address,
                log_index, price_impact
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()
        return len(rows)

    def query_events(
        self,
        from_block: int,
        to_block: int,
        address: Optional[str] = None,
    ) -> list[UnifiedEvent]:
        """Query events in a block range, optionally filtered by address."""
        sql = "SELECT * FROM events WHERE block_number >= ? AND block_number <= ?"
        params: list = [from_block, to_block]

        if address:
            sql += " AND (from_address = ? OR to_address = ?)"
            params.extend([address.lower(), address.lower()])

        sql += " ORDER BY block_number, log_index"

        cursor = self.conn.execute(sql, params)
        results = []
        for row in cursor.fetchall():
            results.append(UnifiedEvent(
                tx_hash=row[1],
                block_number=row[2],
                timestamp=row[3],
                event_type=row[4],
                from_address=row[5],
                to_address=row[6],
                token_address=row[7],
                amount=Decimal(row[8]),
                token_in=row[9],
                token_out=row[10],
                amount_in=Decimal(row[11]) if row[11] else None,
                amount_out=Decimal(row[12]) if row[12] else None,
                function_selector=row[13],
                gas_price=row[14],
                contract_address=row[15],
                log_index=row[16],
                price_impact=row[17],
            ))
        return results

    def get_block_range(self) -> tuple[int, int]:
        """Return (min_block, max_block) in storage."""
        cursor = self.conn.execute(
            "SELECT MIN(block_number), MAX(block_number) FROM events"
        )
        row = cursor.fetchone()
        return (row[0] or 0, row[1] or 0)

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM events")
        return cursor.fetchone()[0]

    def close(self) -> None:
        self.conn.close()
