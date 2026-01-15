# 全β値のoccupancyデータを迅速に収集（小規模版）

$beta_values = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
$num_agents = 100
$num_runs = 5

Write-Host "Collecting occupancy data for all β values..." -ForegroundColor Cyan
Write-Host "Agents: $num_agents, Runs: $num_runs`n"

foreach ($beta in $beta_values) {
    Write-Host "β=$beta " -NoNewline -ForegroundColor Yellow
    
    python addiction_rl_sim.py `
        --num-agents $num_agents `
        --num-runs $num_runs `
        --fixed-beta $beta `
        --save-occupancy-data `
        --output "beta_$beta`_result.json" `
        --occupancy-output "beta_$beta`_episodes.json" 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓" -ForegroundColor Green
    } else {
        Write-Host "✗" -ForegroundColor Red
    }
}

Write-Host "`nβ=Learning " -NoNewline -ForegroundColor Yellow
python addiction_rl_sim.py `
    --num-agents $num_agents `
    --num-runs $num_runs `
    --save-occupancy-data `
    --output "beta_learning_result.json" `
    --occupancy-output "beta_learning_episodes.json" 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓" -ForegroundColor Green
} else {
    Write-Host "✗" -ForegroundColor Red
}

Write-Host "`n✓ All data collected!`n" -ForegroundColor Green
