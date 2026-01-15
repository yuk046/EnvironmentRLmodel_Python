# 全てのβ値でoccupancyデータを収集するスクリプト

$beta_values = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
$num_agents = 200
$num_runs = 10

Write-Host "======================================"
Write-Host "Occupancy Data Collection"
Write-Host "======================================"
Write-Host "β values: $beta_values"
Write-Host "Agents: $num_agents"
Write-Host "Runs: $num_runs"
Write-Host ""

# 固定β値でのシミュレーション
foreach ($beta in $beta_values) {
    $beta_str = "$beta"
    Write-Host "[$beta_str] Starting simulation..."
    
    python addiction_rl_sim.py `
        --num-agents $num_agents `
        --num-runs $num_runs `
        --fixed-beta $beta `
        --save-occupancy-data `
        --output "beta_$beta_str`_result.json" `
        --occupancy-output "beta_$beta_str`_episodes.json"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$beta_str] ✓ Completed successfully" -ForegroundColor Green
    } else {
        Write-Host "[$beta_str] ✗ Failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
    Write-Host ""
}

# Learning βモードでのシミュレーション
Write-Host "[Learning] Starting simulation..."
python addiction_rl_sim.py `
    --num-agents $num_agents `
    --num-runs $num_runs `
    --save-occupancy-data `
    --output "beta_learning_result.json" `
    --occupancy-output "beta_learning_episodes.json"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[Learning] ✓ Completed successfully" -ForegroundColor Green
} else {
    Write-Host "[Learning] ✗ Failed with exit code $LASTEXITCODE" -ForegroundColor Red
}

Write-Host ""
Write-Host "======================================"
Write-Host "All simulations completed!"
Write-Host "======================================"
Write-Host ""
Write-Host "Generated files:"
Get-ChildItem -Filter "*_episodes.json" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize

Write-Host ""
Write-Host "To create occupancy timeline plot, run:"
Write-Host "  python plot_occupancy_timeline.py --beta-values 0.0 0.2 0.4 0.6 0.8 1.0"
