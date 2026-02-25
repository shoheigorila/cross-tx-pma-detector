# Cross-Transaction Price Manipulation Attack Detector

Temporal GNN を用いたクロストランザクション価格操作攻撃の検出システム。

既存の DeFi 価格操作攻撃検出手法（DeFiRanger, DeFort, DeFiGuard, DeFiScope）は全てシングルトランザクション分析に限定されている。本プロジェクトは、INUKO 攻撃（48時間・57,000ブロック）や Beanstalk governance 攻撃（24時間超）のような**複数トランザクションにまたがる攻撃**を検出するマルチスケール Temporal GNN アーキテクチャを提案する。

## アーキテクチャ

```
Data Collection → Graph Construction → Address Clustering → Multi-Scale TGNN → Attack Analysis
```

### マルチスケール検出エンジン

| ブランチ | モデル | ウィンドウ | 対象攻撃 |
|---|---|---|---|
| Short-scale | TGN (GRU memory) | 100 blocks (~20分) | サンドイッチ、即時操作 |
| Medium-scale | DyGFormer (Transformer) | 1,000 blocks (~3.3時間) | 段階的価格操作 |
| Long-scale | GraphMixer (MLP-Mixer) | 10,000 blocks (~33時間) | ガバナンス攻撃、Slow Rug |

### 対象攻撃クラス

| 攻撃タイプ | 時間幅 | 代表例 |
|---|---|---|
| Slow Price Manipulation | 数時間〜数日 | INUKO ($?, 48h) |
| Governance Attack | 1〜7日 | Beanstalk ($182M) |
| Multi-step Oracle Manipulation | 数分〜数時間 | Mango Markets ($117M) |
| Gradual Liquidity Drain | 数日〜数週間 | 多数 |
| Sandwich Accumulation | 数時間 | MEV ボット系 |

## セットアップ

```bash
# 依存関係のインストール
pip install -e ".[dev]"

# 環境変数の設定
export ALCHEMY_API_KEY="your-api-key"
export ETHEREUM_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/your-key"
```

## 使い方

```bash
# 1. データ収集
python scripts/collect_data.py --chain ethereum --from-block 14000000 --to-block 14100000

# 2. グラフ構築
python scripts/build_graphs.py --chain ethereum --window-size 1000 --stride 200

# 3. 学習
python scripts/train.py --chain ethereum --epochs 100

# 4. 評価
python scripts/evaluate.py --chain ethereum --checkpoint data/checkpoints/best_model.pt
```

## プロジェクト構造

```
cross-tx-pma-detector/
├── config/                    # 設定ファイル（チェーン固有設定含む）
├── src/
│   ├── data_collector/        # Alchemy/Web3 データ収集 + SQLite ストレージ
│   ├── graph_builder/         # CTDG 構築 + Time2Vec + 特徴量計算
│   ├── clustering/            # アドレスクラスタリング（Gas Funding, 時間的共起, コントラクト生成）
│   ├── model/                 # マルチスケール TGNN + Focal Loss + 学習/評価
│   ├── analysis/              # 攻撃分類・タイムライン復元・利益推定
│   └── utils/                 # 設定ローダー
├── scripts/                   # CLI スクリプト
├── data/                      # データ格納（gitignore 対象）
├── notebooks/                 # 実験ノートブック
└── tests/                     # テスト
```

## 技術スタック

- **DL**: PyTorch 2.x + PyTorch Geometric
- **データ収集**: Alchemy SDK, Web3.py
- **DB**: SQLite（開発）/ PostgreSQL + TimescaleDB（本番）
- **Ground Truth**: DeFiHackLabs (550+ incidents)

## 参考論文

- DeFiRanger (IEEE TDSC 2023) - ルールベース Cash Flow Tree
- DeFort (ISSTA 2024) - 行動モデル状態遷移オートマトン
- DeFiGuard (arXiv 2024) - GNN グラフ分類
- DeFiScope (arXiv 2025) - LLM 価格モデル推論

## ライセンス

MIT
