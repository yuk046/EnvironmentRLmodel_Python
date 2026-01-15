# 時系列状態占有率グラフの作成方法

このドキュメントでは、論文スタイルの状態占有率グラフを作成する方法を説明します。

## 概要

以下の3つのスクリプトを使用して、状態占有率を可視化できます：

1. **plot_state_occupancy.py** - 既存データから状態占有率グラフを作成（推奨）
2. **plot_occupancy_timeline.py** - 時系列の3パネルグラフを作成
3. **collect_occupancy_data.py** - 時系列データを収集

## クイックスタート（既存データ使用）

### 既存の実験結果から状態占有率を可視化

最も簡単な方法は、既に実行済みの結果ディレクトリを使用することです：

```powershell
# beta_comparison_resultsから作成
python plot_state_occupancy.py --results-dir beta_comparison_results

# beta_reversal_resultsから作成
python plot_state_occupancy.py --results-dir beta_reversal_results `
    --output-comparison state_occupancy_reversal.png `
    --output-heatmap state_occupancy_reversal_heatmap.png

# カスタム状態を指定（例: Goal, Drug, Withdrawal）
python plot_state_occupancy.py --states 0 7 14
```

これにより以下のグラフが生成されます：
- **state_occupancy_comparison.png** - 2パネル（Addiction/Reversal Phase）の棒グラフ
- **state_occupancy_heatmap.png** - 全状態×β値のヒートマップ

### 生成されるグラフ

#### 1. 状態占有率比較グラフ（2パネル）
- **Panel A**: Addiction Phase - 各状態の占有率を比較
- **Panel B**: Reversal Phase - Goal Switch後の状態占有率

各パネルには：
- X軸: β値（0.0=Model-Free 〜 1.0=Model-Based）
- Y軸: 状態占有率（%）
- 棒グラフ: 選択した状態ごとに異なる色

#### 2. 状態占有率ヒートマップ
- X軸: 状態ID（0〜21）
- Y軸: β値
- 色: 占有率の高さ（黄色→赤）
- 青い点線: 重要な状態（Goal, Drug, Withdrawal）

## 高度な使用方法

### 時系列グラフ（3パネル）の作成

論文のような時間経過を示す3パネルグラフを作成：

### 方法1: ダミーデータで素早くテスト

実際のシミュレーションを実行せずに、サンプルグラフを生成できます：

```powershell
python plot_occupancy_timeline.py --output test_occupancy.png
```

これにより、ダミーデータを使用した3パネルグラフが `test_occupancy.png` として生成されます。

### 方法2: 実際のシミュレーションデータを使用

#### ステップ1: データ収集

```powershell
# デフォルト設定（β=0.0, 0.25, 0.5, 0.75, 1.0）
python collect_occupancy_data.py

# カスタム設定
python collect_occupancy_data.py `
    --beta-values 0.0 0.5 1.0 `
    --num-agents 500 `
    --num-runs 30 `
    --output-dir my_occupancy_data
```

これにより、各β値についてシミュレーションが実行され、時系列データが `occupancy_data/` ディレクトリに保存されます。

**注意**: 現在のバージョンは合成データを生成します。実際のシミュレーションデータを使用するには、`addiction_rl_sim.py` に時系列データ収集機能を追加する必要があります。

#### ステップ2: グラフ作成

```powershell
# デフォルト設定
python plot_occupancy_timeline.py --results-dir occupancy_data

# カスタム設定
python plot_occupancy_timeline.py `
    --results-dir my_occupancy_data `
    --output my_occupancy_graph.png `
    --beta-values 0.0 0.5 1.0 `
    --switch-step 2000 `
    --max-steps 15000
```

## グラフの説明

生成されるグラフは3つのパネル（A, B, C）で構成されます：

- **Panel A**: Addictive Reward - 薬物状態（State 7）の占有率
- **Panel B**: Healthy Reward, Position 1 - ゴール状態（State 0）の占有率
- **Panel C**: Healthy Reward, Position 2 - その他の報酬状態の占有率

各パネルには：
- X軸: 時間ステップ（0〜15000）
- Y軸: 状態占有率（% occupancy）
- 青い点線: Goal Switch時点（デフォルト: 2000ステップ）
- 複数の線: 異なるβ値の条件
- エラーバー: 標準誤差（SEM）

## コマンドラインオプション

### collect_occupancy_data.py

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| --beta-values | 0.0 0.25 0.5 0.75 1.0 | 収集するβ値のリスト |
| --num-agents | 900 | エージェント数 |
| --num-runs | 50 | 実行回数 |
| --seed | 42 | 乱数シード |
| --output-dir | occupancy_data | 出力ディレクトリ |
| --time-bin-size | 2000 | 時間区間のサイズ |
| --max-steps | 15000 | 最大ステップ数 |

### plot_occupancy_timeline.py

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| --results-dir | beta_comparison_results | 結果ディレクトリ |
| --output | occupancy_timeline.png | 出力ファイル名 |
| --beta-values | 0.0 0.25 0.5 0.75 1.0 | 比較するβ値 |
| --switch-step | 2000 | Goal Switch発生ステップ |
| --max-steps | 15000 | 最大ステップ数 |

## カスタマイズ

### 色とスタイルの変更

[plot_occupancy_timeline.py](plot_occupancy_timeline.py) の `beta_colors` 辞書を編集：

```python
beta_colors = {
    0.0: '#666666',    # β=0 (Model-Free)
    0.25: '#999999',
    0.5: '#AAAAAA',
    0.75: '#CCCCCC',
    1.0: '#FF0000',    # β=1 (Model-Based)
}
```

### 表示する状態の変更

[plot_occupancy_timeline.py](plot_occupancy_timeline.py) の `panel_configs` を編集：

```python
panel_configs = [
    {'title': 'Addictive Reward', 'state': STATE_DRUG, 'ylabel': '% occupancy'},
    {'title': 'Healthy Reward, Position 1', 'state': STATE_GOAL, 'ylabel': '% occupancy'},
    {'title': 'Healthy Reward, Position 2', 'state': 1, 'ylabel': '% occupancy'}
]
```

## トラブルシューティング

### データファイルが見つからない

エラー: `警告: beta_X_episodes.json が見つかりません`

解決策:
1. `collect_occupancy_data.py` を先に実行してデータを収集
2. または `--results-dir` オプションで正しいディレクトリを指定

### グラフが表示されない

生成されたPNGファイルを直接開いて確認してください：

```powershell
# Windowsでファイルを開く
Start-Process occupancy_timeline.png
```

### メモリ不足エラー

大量のデータを処理する場合は、`--num-agents` や `--num-runs` を減らしてください。

## 次のステップ

実際のシミュレーションデータを使用するには、`addiction_rl_sim.py` に以下の機能を追加する必要があります：

1. タイムステップごとの状態訪問を記録
2. 時間区間ごとに状態占有率を集計
3. JSON形式でエピソードデータを保存

詳細は開発者に相談してください。
