#!/bin/bash

# 全てのβ値でoccupancyデータを収集するスクリプト

python_path="/Users/yukinari/EnvironmentRLmodel_Python/.venv/bin/python"
beta_values=(0.0 0.2 0.4 0.6 0.8 1.0)
num_agents=100
num_runs=1

echo "======================================"
echo "Occupancy Data Collection"
echo "======================================"
echo "β values: ${beta_values[@]}"
echo "Agents: $num_agents"
echo "Runs: $num_runs"
echo ""

# 固定β値でのシミュレーション
for beta in "${beta_values[@]}"; do
    beta_str="$beta"
    echo "[$beta_str] Starting simulation..."
    
    $python_path addiction_rl_sim.py \
        --num-agents $num_agents \
        --num-runs $num_runs \
        --fixed-beta $beta \
        --save-occupancy-data \
        --output "beta_${beta_str}_result.json" \
        --occupancy-output "beta_${beta_str}_episodes.json"
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32m[$beta_str] ✓ Completed successfully\033[0m"
    else
        echo -e "\033[0;31m[$beta_str] ✗ Failed with exit code $?\033[0m"
    fi
    echo ""
done

# Learning βモードでのシミュレーション
echo "[Learning] Starting simulation..."
$python_path addiction_rl_sim.py \
    --num-agents $num_agents \
    --num-runs $num_runs \
    --save-occupancy-data \
    --output "beta_learning_result.json" \
    --occupancy-output "beta_learning_episodes.json"

if [ $? -eq 0 ]; then
    echo -e "\033[0;32m[Learning] ✓ Completed successfully\033[0m"
else
    echo -e "\033[0;31m[Learning] ✗ Failed with exit code $?\033[0m"
fi

echo ""
echo "======================================"
echo "All simulations completed!"
echo "======================================"
echo ""
echo "Generated files:"
ls -lh *_episodes.json 2>/dev/null | awk '{print $9, $5, $6, $7, $8}'

echo ""
echo "To create occupancy timeline plot, run:"
echo "  $python_path plot_occupancy_timeline.py --beta-values 0.0 0.2 0.4 0.6 0.8 1.0 learning"
