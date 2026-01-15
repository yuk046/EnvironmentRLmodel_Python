# 状態占有率グラフ作成ツール - クイックリファレンス

## 作成されたファイル

### メインスクリプト

1. **plot_state_occupancy.py** ⭐ おすすめ
   - 既存の実験結果から状態占有率を可視化
   - 2種類のグラフを生成（比較棒グラフ + ヒートマップ）
   - すぐに使える

2. **plot_occupancy_timeline.py**
   - 時系列の3パネルグラフ（論文スタイル）
   - エピソードデータが必要

3. **collect_occupancy_data.py**
   - 時系列データ収集用

### ドキュメント

- **README_OCCUPANCY.md** - 詳細な使用方法

## 基本的な使い方

### 1. 既存データからグラフ作成（最も簡単）

```powershell
# デフォルト設定
python plot_state_occupancy.py

# 別のディレクトリから
python plot_state_occupancy.py --results-dir beta_reversal_results

# 特定の状態のみプロット（Goal=0, Drug=7, Withdrawal=14）
python plot_state_occupancy.py --states 0 7 14
```

**出力:**
- `state_occupancy_comparison.png` - 2パネル比較グラフ
- `state_occupancy_heatmap.png` - ヒートマップ

### 2. 時系列グラフ作成（テスト用ダミーデータ）

```powershell
# サンプルグラフを生成
python plot_occupancy_timeline.py --output sample_timeline.png

# カスタム設定
python plot_occupancy_timeline.py `
    --beta-values 0.0 0.5 1.0 `
    --switch-step 2000 `
    --max-steps 15000
```

**出力:**
- `sample_timeline.png` - 3パネル時系列グラフ

## コマンドラインオプション一覧

### plot_state_occupancy.py

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--results-dir` | beta_comparison_results | 結果ディレクトリ |
| `--output-comparison` | state_occupancy_comparison.png | 比較グラフ出力名 |
| `--output-heatmap` | state_occupancy_heatmap.png | ヒートマップ出力名 |
| `--beta-values` | 0.0 0.2 0.4 0.6 0.8 1.0 | 比較するβ値 |
| `--states` | 0 7 | プロットする状態ID |

### plot_occupancy_timeline.py

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `--results-dir` | beta_comparison_results | 結果ディレクトリ |
| `--output` | occupancy_timeline.png | 出力ファイル名 |
| `--beta-values` | 0.0 0.25 0.5 0.75 1.0 | 比較するβ値 |
| `--switch-step` | 2000 | Goal Switch時点 |
| `--max-steps` | 15000 | 最大ステップ数 |

## よく使う状態ID

| ID | 名前 | 説明 |
|----|------|------|
| 0 | Goal State | 健康的な報酬（ゴール） |
| 7 | Drug State | 依存性物質（薬物） |
| 14 | Withdrawal State | 離脱症状状態 |
| 1-6 | Neutral States | 中立状態 |
| 8-21 | After-effect States | 薬物使用後の状態 |

## 実用例

### 例1: Goal状態とDrug状態を比較

```powershell
python plot_state_occupancy.py --states 0 7
```

### 例2: 複数のβ値で比較

```powershell
python plot_state_occupancy.py --beta-values 0.0 0.5 1.0
```

### 例3: カスタム出力名

```powershell
python plot_state_occupancy.py `
    --output-comparison my_comparison.png `
    --output-heatmap my_heatmap.png
```

### 例4: 全ての重要な状態を表示

```powershell
python plot_state_occupancy.py --states 0 7 14
```

## トラブルシューティング

### データファイルが見つからない

```
警告: beta_X.json が見つかりません
```

**解決策:**
- 正しいディレクトリを `--results-dir` で指定
- または先に `addiction_rl_sim.py` でシミュレーションを実行

### グラフが空白

**解決策:**
- `--beta-values` が結果ディレクトリのファイルと一致しているか確認
- `--states` の状態IDが有効な範囲（0-21）か確認

### メモリエラー

**解決策:**
- ヒートマップのみをスキップして比較グラフだけ作成
- または少ないβ値で試す

## 生成されるグラフの見方

### 比較グラフ（2パネル）

- **左側の棒**: 各状態の占有率
- **色の濃さ**: β値を表す（濃いグレー=0, 赤=1.0）
- **高さ**: その状態を訪問した時間の割合

### ヒートマップ

- **横軸**: 全ての状態（0-21）
- **縦軸**: β値
- **色**: 占有率（黄色=低い、赤=高い）
- **青い点線**: 重要な状態マーカー

## さらに詳しく

詳細な説明は [README_OCCUPANCY.md](README_OCCUPANCY.md) を参照してください。
