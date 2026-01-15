# 全β値の時系列占有率グラフ作成ガイド

## 完成！✨

全てのβ値（0.0, 0.2, 0.4, 0.6, 0.8, 1.0）とlearningモードを含む時系列占有率グラフが作成できるようになりました。

## クイックスタート

### 1. データ収集（全β値）

```powershell
# 小規模で素早く
.\quick_collect_occupancy.ps1

# または大規模で詳細に
.\run_all_occupancy.ps1
```

これにより以下のファイルが生成されます：
- `beta_0.0_episodes.json`
- `beta_0.2_episodes.json`
- `beta_0.4_episodes.json`
- `beta_0.6_episodes.json`
- `beta_0.8_episodes.json`
- `beta_1.0_episodes.json`
- `beta_learning_episodes.json`

### 2. グラフ作成

```powershell
# デフォルト（全β値 + learning）
python plot_occupancy_timeline.py --results-dir .

# 出力ファイル名を指定
python plot_occupancy_timeline.py --results-dir . --output my_timeline.png
```

**デフォルトで以下が含まれます：**
- β=0.0 (Model-Free) - 青
- β=0.2 - オレンジ
- β=0.4 - 緑
- β=0.6 - 赤
- β=0.8 - 紫
- β=1.0 (Model-Based) - 茶色
- β=Learning (適応的) - ピンク

## カスタマイズ

### 特定のβ値のみをプロット

```powershell
# β=0, 0.5, 1のみ
python plot_occupancy_timeline.py --beta-values 0.0 0.5 1.0

# learningモードを除外
python plot_occupancy_timeline.py --beta-values 0.0 0.2 0.4 0.6 0.8 1.0

# learningモードのみ
python plot_occupancy_timeline.py --beta-values learning
```

### 各β値で個別にデータ収集

```powershell
# 特定のβ値のみ
python addiction_rl_sim.py `
    --num-agents 200 `
    --num-runs 10 `
    --fixed-beta 0.5 `
    --save-occupancy-data `
    --occupancy-output beta_0.5_episodes.json
```

## 生成されるグラフ

### 3パネル構成

**Panel A: Addictive Reward**
- Drug state (State 7) の占有率
- Switch後の依存行動の持続性を表示

**Panel B: Healthy Reward, Position 1**
- Goal state (State 0) の占有率
- Switch前後での健康的行動の変化を表示

**Panel C: Healthy Reward, Position 2**
- その他の報酬状態の占有率
- 代替行動パターンを表示

### 各β値の色分け

- **青** (β=0.0): 完全Model-Free - 習慣的
- **オレンジ〜赤** (β=0.2-0.6): 混合型
- **紫〜茶** (β=0.8-1.0): 完全Model-Based - 計画的
- **ピンク** (Learning): 適応的β選択

## 実際の使用例

### 論文用の高品質グラフ

```powershell
# 大規模データ収集
foreach ($beta in 0.0, 0.2, 0.4, 0.6, 0.8, 1.0) {
    python addiction_rl_sim.py `
        --num-agents 500 `
        --num-runs 20 `
        --fixed-beta $beta `
        --save-occupancy-data `
        --occupancy-output "beta_$beta`_episodes.json"
}

# learningモード
python addiction_rl_sim.py `
    --num-agents 500 `
    --num-runs 20 `
    --save-occupancy-data `
    --occupancy-output "beta_learning_episodes.json"

# 高解像度グラフ作成
python plot_occupancy_timeline.py --results-dir . --output publication_timeline.png
```

### 比較分析

```powershell
# Model-Free vs Model-Based
python plot_occupancy_timeline.py --beta-values 0.0 1.0 --output mf_vs_mb.png

# 全段階
python plot_occupancy_timeline.py --beta-values 0.0 0.2 0.4 0.6 0.8 1.0 learning --output all_comparison.png
```

## トラブルシューティング

### データファイルが見つからない

```powershell
# 現在のディレクトリを確認
ls beta_*_episodes.json

# 不足しているβ値のデータを収集
python addiction_rl_sim.py --fixed-beta 0.4 --save-occupancy-data --occupancy-output beta_0.4_episodes.json
```

### グラフが混雑している

```powershell
# β値を減らす
python plot_occupancy_timeline.py --beta-values 0.0 0.5 1.0 learning

# または2つのグラフに分割
python plot_occupancy_timeline.py --beta-values 0.0 0.2 0.4 --output timeline_low.png
python plot_occupancy_timeline.py --beta-values 0.6 0.8 1.0 learning --output timeline_high.png
```

## パフォーマンス

### データ収集時間の目安

- 100 agents × 5 runs: 約1-2分/β値
- 200 agents × 10 runs: 約3-5分/β値
- 500 agents × 20 runs: 約10-15分/β値

**全7つのβ値（learningを含む）:**
- 小規模: 約7-14分
- 中規模: 約21-35分
- 大規模: 約70-105分

## まとめ

これで、添付画像のような論文品質の時系列占有率グラフが、実際のシミュレーションデータから作成できます。全てのβ値とlearningモードを含めることで、Model-FreeからModel-Basedまでの連続的な変化と、適応的学習の効果を視覚的に比較できます。

**生成されたファイル:**
- `occupancy_timeline_all_betas.png` - 全β値を含む完全版
- 各`beta_X_episodes.json` - 時系列データ
