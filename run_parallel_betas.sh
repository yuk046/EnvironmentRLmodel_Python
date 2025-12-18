#!/bin/bash

# Python実行ファイルのパス（仮想環境を使用）
PYTHON=".venv/bin/python"

# 並列実行設定
NUM_AGENTS=900
NUM_RUNS=10
SEED=42
OUTPUT_DIR="./beta_results_parallel"

# 出力ディレクトリを作成
mkdir -p "$OUTPUT_DIR"

# β値のリスト
BETAS=(0.0 0.2 0.4 0.6 0.8 1.0)

echo "Starting parallel simulations..."
echo "Output directory: $OUTPUT_DIR"
echo "================================"

# 各β値を並列実行
for beta in "${BETAS[@]}"; do
    output_file="$OUTPUT_DIR/beta_${beta}.json"
    echo "Launching simulation for beta=$beta..."
    
    # バックグラウンドで実行
    $PYTHON addiction_rl_sim.py \
        --beta "$beta" \
        --num-agents "$NUM_AGENTS" \
        --num-runs "$NUM_RUNS" \
        --seed "$SEED" \
        --output "$output_file" \
        --mb-forget &
done

# すべてのバックグラウンドジョブが完了するまで待機
echo "Waiting for all simulations to complete..."
wait

echo "================================"
echo "All simulations completed!"
echo "Results saved in $OUTPUT_DIR"
echo ""
echo "To plot results, run:"
echo "  $PYTHON plot_results.py"
