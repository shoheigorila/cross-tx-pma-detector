"""Data collection from Ethereum-compatible chains via Alchemy / Web3 APIs."""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Iterator, Optional

import requests
from web3 import Web3

from .events import UnifiedEvent

logger = logging.getLogger(__name__)

ETH_ADDRESS = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"


class AlchemyCollector:
    """Collect token transfers and internal transactions via Alchemy API."""

    def __init__(self, api_key: str, chain: str = "ethereum"):
        base_urls = {
            "ethereum": f"https://eth-mainnet.g.alchemy.com/v2/{api_key}",
            "bsc": f"https://bnb-mainnet.g.alchemy.com/v2/{api_key}",
        }
        self.url = base_urls.get(chain, base_urls["ethereum"])
        self.rate_limit = 25  # requests per second

    def _rpc_call(self, method: str, params: list) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"RPC error: {data['error']}")
        return data.get("result", {})

    def get_asset_transfers(
        self,
        from_block: int,
        to_block: int,
        from_address: Optional[str] = None,
        to_address: Optional[str] = None,
        categories: Optional[list[str]] = None,
    ) -> Iterator[UnifiedEvent]:
        """Fetch asset transfers via alchemy_getAssetTransfers (paginated)."""
        if categories is None:
            categories = ["external", "internal", "erc20"]

        params: dict = {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "category": categories,
            "withMetadata": True,
            "maxCount": "0x3e8",  # 1000
        }
        if from_address:
            params["fromAddress"] = from_address
        if to_address:
            params["toAddress"] = to_address

        page_key = None
        while True:
            if page_key:
                params["pageKey"] = page_key

            result = self._rpc_call("alchemy_getAssetTransfers", [params])
            transfers = result.get("transfers", [])

            for tx in transfers:
                yield self._parse_asset_transfer(tx)

            page_key = result.get("pageKey")
            if not page_key:
                break

            time.sleep(1 / self.rate_limit)

    def _parse_asset_transfer(self, tx: dict) -> UnifiedEvent:
        metadata = tx.get("metadata", {})
        block_ts = metadata.get("blockTimestamp", "")
        # Parse ISO timestamp to unix
        timestamp = 0
        if block_ts:
            from datetime import datetime, timezone
            try:
                dt = datetime.fromisoformat(block_ts.replace("Z", "+00:00"))
                timestamp = int(dt.timestamp())
            except ValueError:
                pass

        category = tx.get("category", "external")
        if category == "erc20":
            token_addr = tx.get("rawContract", {}).get("address", "")
        else:
            token_addr = ETH_ADDRESS

        value = tx.get("value") or 0

        return UnifiedEvent(
            tx_hash=tx.get("hash", ""),
            block_number=int(tx.get("blockNum", "0x0"), 16),
            timestamp=timestamp,
            event_type="transfer",
            from_address=(tx.get("from") or "").lower(),
            to_address=(tx.get("to") or "").lower(),
            token_address=token_addr.lower(),
            amount=Decimal(str(value)),
            gas_price=0,
            contract_address=token_addr.lower(),
        )


class Web3EventCollector:
    """Collect Swap and other DeFi events via eth_getLogs."""

    # Uniswap V2 Swap event signature
    SWAP_V2_TOPIC = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
    # ERC20 Transfer event signature
    TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

    def __init__(self, rpc_url: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

    def get_swap_events(
        self,
        from_block: int,
        to_block: int,
        pool_addresses: Optional[list[str]] = None,
    ) -> Iterator[UnifiedEvent]:
        """Fetch Uniswap V2-style Swap events."""
        filter_params: dict = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "topics": [self.SWAP_V2_TOPIC],
        }
        if pool_addresses:
            filter_params["address"] = pool_addresses

        logs = self.w3.eth.get_logs(filter_params)

        for log in logs:
            yield self._parse_swap_v2(log)

    def _parse_swap_v2(self, log: dict) -> UnifiedEvent:
        """Parse Uniswap V2 Swap event log."""
        data = log["data"].hex() if isinstance(log["data"], bytes) else log["data"]
        # Remove 0x prefix
        data = data[2:] if data.startswith("0x") else data

        # Swap(address sender, uint amount0In, uint amount1In,
        #      uint amount0Out, uint amount1Out, address to)
        amount0_in = int(data[0:64], 16)
        amount1_in = int(data[64:128], 16)
        amount0_out = int(data[128:192], 16)
        amount1_out = int(data[192:256], 16)

        sender = "0x" + log["topics"][1].hex()[-40:]
        to_addr = "0x" + log["topics"][2].hex()[-40:]

        block = self.w3.eth.get_block(log["blockNumber"])

        return UnifiedEvent(
            tx_hash=log["transactionHash"].hex(),
            block_number=log["blockNumber"],
            timestamp=block["timestamp"],
            event_type="swap",
            from_address=sender.lower(),
            to_address=to_addr.lower(),
            token_address=log["address"].lower(),  # pool address
            amount=Decimal(0),
            amount_in=Decimal(amount0_in) if amount0_in else Decimal(amount1_in),
            amount_out=Decimal(amount0_out) if amount0_out else Decimal(amount1_out),
            contract_address=log["address"].lower(),
            log_index=log["logIndex"],
        )

    def get_transfer_events(
        self,
        from_block: int,
        to_block: int,
        token_addresses: Optional[list[str]] = None,
    ) -> Iterator[UnifiedEvent]:
        """Fetch ERC20 Transfer events."""
        filter_params: dict = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "topics": [self.TRANSFER_TOPIC],
        }
        if token_addresses:
            filter_params["address"] = token_addresses

        logs = self.w3.eth.get_logs(filter_params)

        for log in logs:
            yield self._parse_transfer(log)

    def _parse_transfer(self, log: dict) -> UnifiedEvent:
        """Parse ERC20 Transfer event log."""
        topics = log["topics"]
        from_addr = "0x" + topics[1].hex()[-40:]
        to_addr = "0x" + topics[2].hex()[-40:]

        data = log["data"].hex() if isinstance(log["data"], bytes) else log["data"]
        data = data[2:] if data.startswith("0x") else data
        amount = int(data, 16) if data else 0

        block = self.w3.eth.get_block(log["blockNumber"])

        return UnifiedEvent(
            tx_hash=log["transactionHash"].hex(),
            block_number=log["blockNumber"],
            timestamp=block["timestamp"],
            event_type="transfer",
            from_address=from_addr.lower(),
            to_address=to_addr.lower(),
            token_address=log["address"].lower(),
            amount=Decimal(amount),
            contract_address=log["address"].lower(),
            log_index=log["logIndex"],
        )
