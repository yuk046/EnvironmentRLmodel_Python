# 並列β値シミュレーション

## 概要

このシステムは、β値ごとに独立してシミュレーションを実行し、結果を統合してグラフ化します。
各β値を並列実行することで、全体の処理時間を大幅に短縮できます。

## 使い方

### 1. 並列シミュレーションの実行

```bash
./run_parallel_betas.sh
```

このスクリプトは以下を実行します:
- β値 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] それぞれで独立したシミュレーションを起動
- すべてのシミュレーションをバックグラウンドで並列実行
- 結果を `beta_results_parallel/` ディレクトリにJSON形式で保存

### 2. 結果の統合とグラフ化

```bash
python plot_results.py
```

このスクリプトは以下を実行します:
- `beta_results_parallel/` から結果ファイルを読み込み
- データを統合して統計量を計算
- グラフを作成して `addiction_result.png` に保存
- 結果をコンソールに表示

## 詳細オプション

### 単一β値での実行

特定のβ値だけを実行したい場合:

```bash
python addiction_rl_sim.py --beta 0.5 --num-agents 900 --num-runs 10 --output result_beta_0.5.json --mb-forget
```

### カスタム設定での並列実行

`run_parallel_betas.sh` を編集して、以下のパラメータを変更できます:

```bash
NUM_AGENTS=900      # エージェント数
NUM_RUNS=1          # 実行回数（シード数）
SEED=42             # ベースシード値
OUTPUT_DIR="..."    # 出力ディレクトリ
BETAS=(...)         # β値のリスト
```

### プロットのカスタマイズ

異なるディレクトリから結果を読み込む:

```bash
python plot_results.py --input-dir ./my_results --output my_plot.png
```

## ファイル構成

```
.
├── addiction_rl_sim.py          # メインシミュレーションプログラム
├── run_parallel_betas.sh        # 並列実行スクリプト
├── plot_results.py              # 結果統合とグラフ化
└── beta_results_parallel/       # 結果保存ディレクトリ
    ├── beta_0.0.json
    ├── beta_0.2.json
    ├── beta_0.4.json
    ├── beta_0.6.json
    ├── beta_0.8.json
    └── beta_1.0.json
```

## コマンドライン引数

### addiction_rl_sim.py

- `--beta`: 実行するβ値（省略時は全β値を逐次実行）
- `--num-agents`: エージェント数（デフォルト: 900）
- `--num-runs`: 実行回数（デフォルト: 1）
- `--seed`: ランダムシード（デフォルト: 42）
- `--output`: 結果出力ファイル（JSON形式）
- `--mb-forget`: モデルベース学習のリセット設定
- `--debug-episode`: デバッグ出力を有効化
- `--debug-csv`: デバッグCSV出力パス

### plot_results.py

- `--input-dir`: 結果ファイルのディレクトリ（デフォルト: ./beta_results_parallel）
- `--output`: 出力画像ファイル（デフォルト: addiction_result.png）

## パフォーマンス

- **逐次実行**: 全β値を順番に実行（約6倍の時間）
- **並列実行**: 全β値を同時実行（最も遅いβ値の実行時間とほぼ同じ）

例: β値1つあたり10分かかる場合
- 逐次実行: 約60分
- 並列実行: 約10分
