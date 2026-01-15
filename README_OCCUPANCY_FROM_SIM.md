# addiction_rl_simから時系列占有率データを取得する方法

## 概要

`addiction_rl_sim.py`が直接時系列の状態占有率データを収集・保存できるようになりました。`--save-occupancy-data`オプションを使用することで、実際のシミュレーションデータから占有率の時間変化を可視化できます。

## 使用方法

### ステップ1: シミュレーション実行とデータ収集

```powershell
# β=0.0 (Model-Free)
python addiction_rl_sim.py `
    --num-agents 500 `
    --num-runs 10 `
    --fixed-beta 0.0 `
    --save-occupancy-data `
    --output beta_0.0_result.json `
    --occupancy-output beta_0.0_episodes.json

# β=1.0 (Model-Based)
python addiction_rl_sim.py `
    --num-agents 500 `
    --num-runs 10 `
    --fixed-beta 1.0 `
    --save-occupancy-data `
    --output beta_1.0_result.json `
    --occupancy-output beta_1.0_episodes.json

# β学習モード
python addiction_rl_sim.py `
    --num-agents 500 `
    --num-runs 10 `
    --save-occupancy-data `
    --output beta_learning_result.json `
    --occupancy-output beta_learning_episodes.json
```

**重要なオプション:**
- `--save-occupancy-data`: 時系列占有率データの収集を有効化
- `--occupancy-output`: 占有率データの保存先ファイル（省略可、デフォルトは`beta_X_episodes.json`）
- `--time-bin-size`: 時間区間のサイズ（デフォルト: 2000ステップ）

### ステップ2: グラフ作成

```powershell
# 実際のデータから時系列グラフを作成
python plot_occupancy_timeline.py `
    --results-dir . `
    --beta-values 0.0 1.0 `
    --output occupancy_timeline.png

# カスタム設定
python plot_occupancy_timeline.py `
    --results-dir my_data `
    --beta-values 0.0 0.5 1.0 `
    --output my_timeline.png `
    --switch-step 2000 `
    --max-steps 15000
```

## 生成されるデータ形式

### occupancy data JSON構造

```json
{
  "beta": 0.0,
  "beta_mode": "fixed",
  "n_runs": 10,
  "time_bins": [0, 2000, 4000, ...],
  "time_bin_size": 2000,
  "total_steps": 2050,
  "states": {
    "0": [  // State 0 (Goal)の占有率
      [0.05, 0.10, ...],  // Run 0の各時間区間での占有率
      [0.06, 0.11, ...],  // Run 1の各時間区間での占有率
      ...
    ],
    "7": [  // State 7 (Drug)の占有率
      [0.02, 0.15, ...],
      ...
    ],
    ...
  }
}
```

**フィールド説明:**
- `beta`: β値（固定モードの場合）またはnull（学習モード）
- `beta_mode`: "fixed"または"learning"
- `n_runs`: シミュレーション実行回数
- `time_bins`: 時間区間の境界値
- `states`: 各状態の占有率データ（run × time_bin行列）

## 完全なワークフロー例

### 複数のβ値を比較

```powershell
# 各β値でデータ収集
$beta_values = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

foreach ($beta in $beta_values) {
    python addiction_rl_sim.py `
        --num-agents 500 `
        --num-runs 10 `
        --fixed-beta $beta `
        --save-occupancy-data `
        --output "beta_$beta`_result.json" `
        --occupancy-output "beta_$beta`_episodes.json"
}

# グラフ作成
python plot_occupancy_timeline.py `
    --beta-values 0.0 0.2 0.4 0.6 0.8 1.0 `
    --output occupancy_comparison_all.png
```

### 既存データとの比較

```powershell
# 既存の状態占有率グラフと併用
python plot_state_occupancy.py --results-dir .
python plot_occupancy_timeline.py --results-dir .
```

## パフォーマンス考慮事項

### メモリ使用量

時系列データ収集は追加のメモリを使用します：
- 各エージェント: 約2050ステップ × 状態ID = 約16KB
- 500エージェント × 10 runs = 約80MB

大規模実験では`--num-agents`を調整してください。

### 処理時間

占有率データ収集により、わずかな処理時間の増加（約5-10%）があります。

## トラブルシューティング

### メモリ不足

```powershell
# エージェント数を減らす
python addiction_rl_sim.py --num-agents 200 --save-occupancy-data ...

# または実行回数を減らす
python addiction_rl_sim.py --num-runs 5 --save-occupancy-data ...
```

### ファイルが見つからない

エラー: `警告: beta_X_episodes.json が見つかりません`

解決策:
```powershell
# --occupancy-outputで明示的に指定
python addiction_rl_sim.py --occupancy-output my_data.json --save-occupancy-data ...

# または --results-dir を正しく設定
python plot_occupancy_timeline.py --results-dir path/to/data
```

### グラフが空または不完全

- データ収集時に`--save-occupancy-data`オプションが指定されているか確認
- `occupancy_output`ファイルが実際に生成されているか確認
- ファイル内の`states`キーにデータが含まれているか確認

## 高度な使用例

### カスタム時間区間

```powershell
# 500ステップごとに集計
python addiction_rl_sim.py `
    --save-occupancy-data `
    --time-bin-size 500 `
    --occupancy-output fine_grained_data.json
```

### 特定の状態のみ可視化

プロットスクリプトで`panel_configs`を編集し、対象状態を変更できます。

## 従来の方法との比較

### 新方式（推奨）
```powershell
# ワンステップでデータ収集
python addiction_rl_sim.py --save-occupancy-data ...
python plot_occupancy_timeline.py ...
```

**利点:**
- ✅ 実際のシミュレーションデータを使用
- ✅ 正確な時系列情報
- ✅ 追加のスクリプト不要

### 旧方式
```powershell
# collect_occupancy_data.pyを使用（ダミーデータ生成）
python collect_occupancy_data.py ...
python plot_occupancy_timeline.py ...
```

**欠点:**
- ❌ 合成データのみ
- ❌ 近似値
- ❌ 追加の処理ステップ

## まとめ

`--save-occupancy-data`オプションにより、addiction_rl_sim.pyから直接実際の時系列占有率データを取得できるようになりました。これにより、より正確で信頼性の高い分析が可能になります。
