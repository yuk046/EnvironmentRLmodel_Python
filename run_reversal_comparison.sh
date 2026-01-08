#!/bin/bash
# β値固定実験とβ学習モードの比較実験スクリプト（Reversalフェーズ分析対応版）
# 各β値で実験を並列実行し、結果をJSONファイルに保存

# Pythonパス設定（仮想環境使用）
PYTHON="/Users/yukinari/EnvironmentRLmodel_Python/.venv/bin/python"

NUM_AGENTS=900
NUM_RUNS=1
SEED=42

# 結果保存用ディレクトリ
RESULTS_DIR="beta_reversal_results"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "β値比較実験開始（Reversalフェーズ分析対応）"
echo "Agents: $NUM_AGENTS, Runs: $NUM_RUNS"
echo "========================================"
echo ""

# β値のリスト
BETA_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)

# PIDのリスト
PIDS=()

# 各β値で実験を並列起動
for beta in "${BETA_VALUES[@]}"; do
    output_file="$RESULTS_DIR/beta_$beta.json"
    
    echo "[β=$beta] ジョブ起動中..."
    
    $PYTHON addiction_rl_sim.py \
        --num-agents $NUM_AGENTS \
        --num-runs $NUM_RUNS \
        --seed $SEED \
        --fixed-beta $beta \
        --output "$output_file" &
    
    PIDS+=($!)
done

# β学習モードも並列起動
echo "[Learning] ジョブ起動中..."
learning_output="$RESULTS_DIR/beta_learning.json"

$PYTHON addiction_rl_sim.py \
    --num-agents $NUM_AGENTS \
    --num-runs $NUM_RUNS \
    --seed $SEED \
    --output "$learning_output" &

PIDS+=($!)

echo ""
echo "全 ${#PIDS[@]} ジョブを起動しました"
echo "実行中のジョブを待機しています..."
echo ""

# 全ジョブの完了を待機
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "ジョブ (PID: $pid) 完了"
done

echo ""
echo "全ジョブ完了！"
echo ""

# 結果の可視化
echo "========================================"
echo "結果のグラフ化"
echo "========================================"

$PYTHON plot_beta_comparison.py --results-dir "$RESULTS_DIR" --output-prefix "$RESULTS_DIR/reversal_analysis"

if [ $? -eq 0 ]; then
    echo ""
    echo "すべての実験が完了しました！"
    echo "結果は $RESULTS_DIR フォルダに保存されています"
    echo ""
    echo "生成されたグラフ:"
    ls -la "$RESULTS_DIR"/*.png 2>/dev/null || echo "（グラフファイルなし）"
else
    echo ""
    echo "グラフ化でエラーが発生しました"
fi
