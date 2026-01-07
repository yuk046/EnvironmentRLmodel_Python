# β値固定実験とβ学習モードの比較実験スクリプト（並列実行版）
# 各β値で実験を並列実行し、結果をJSONファイルに保存

$NUM_AGENTS = 900
$NUM_RUNS = 1
$SEED = 42

# 結果保存用ディレクトリ
$RESULTS_DIR = "beta_comparison_results"
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "β値比較実験開始（並列実行モード）" -ForegroundColor Cyan
Write-Host "Agents: $NUM_AGENTS, Runs: $NUM_RUNS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# β値のリスト
$BETA_VALUES = @(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ジョブのリスト
$jobs = @()

# 各β値で実験を並列起動
foreach ($beta in $BETA_VALUES) {
    $output_file = "$RESULTS_DIR/beta_$beta.json"
    
    Write-Host "[β=$beta] ジョブ起動中..." -ForegroundColor Yellow
    
    $job = Start-Job -ScriptBlock {
        param($beta, $num_agents, $num_runs, $seed, $output_file)
        
        $startTime = Get-Date
        
        python addiction_rl_sim.py `
            --num-agents $num_agents `
            --num-runs $num_runs `
            --seed $seed `
            --fixed-beta $beta `
            --output $output_file
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        return @{
            Beta = $beta
            ExitCode = $LASTEXITCODE
            Duration = $duration
            OutputFile = $output_file
        }
    } -ArgumentList $beta, $NUM_AGENTS, $NUM_RUNS, $SEED, $output_file
    
    $jobs += @{
        Job = $job
        Beta = $beta
        Label = "β=$beta"
    }
}

# β学習モードも並列起動
Write-Host "[Learning] ジョブ起動中..." -ForegroundColor Yellow
$learning_output = "$RESULTS_DIR/beta_learning.json"

$learning_job = Start-Job -ScriptBlock {
    param($num_agents, $num_runs, $seed, $output_file)
    
    $startTime = Get-Date
    
    python addiction_rl_sim.py `
        --num-agents $num_agents `
        --num-runs $num_runs `
        --seed $seed `
        --output $output_file
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    return @{
        Beta = "Learning"
        ExitCode = $LASTEXITCODE
        Duration = $duration
        OutputFile = $output_file
    }
} -ArgumentList $NUM_AGENTS, $NUM_RUNS, $SEED, $learning_output

$jobs += @{
    Job = $learning_job
    Beta = "Learning"
    Label = "β=Learning"
}

Write-Host ""
Write-Host "全 $($jobs.Count) ジョブを起動しました" -ForegroundColor Green
Write-Host "実行中のジョブを監視しています..." -ForegroundColor Cyan
Write-Host ""

# ジョブの完了を待機
$completed = 0
$total = $jobs.Count

while ($completed -lt $total) {
    Start-Sleep -Seconds 5
    
    foreach ($jobInfo in $jobs) {
        $job = $jobInfo.Job
        $label = $jobInfo.Label
        
        if ($job.State -eq "Completed" -and -not $jobInfo.Reported) {
            $result = Receive-Job -Job $job
            $completed++
            
            Write-Host "[$completed/$total] $label 完了!" -ForegroundColor Green
            if ($result) {
                Write-Host "  所要時間: $($result.Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Gray
                Write-Host "  出力: $($result.OutputFile)" -ForegroundColor Gray
            }
            
            $jobInfo.Reported = $true
        }
        elseif ($job.State -eq "Failed") {
            $completed++
            Write-Host "[$completed/$total] $label 失敗" -ForegroundColor Red
            $jobInfo.Reported = $true
        }
    }
    
    # 進捗表示
    $running = ($jobs | Where-Object { $_.Job.State -eq "Running" }).Count
    if ($running -gt 0) {
        Write-Host "`r実行中: $running ジョブ | 完了: $completed/$total" -NoNewline -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host ""

# ジョブのクリーンアップ
foreach ($jobInfo in $jobs) {
    Remove-Job -Job $jobInfo.Job -Force
}

Write-Host "全ジョブ完了！" -ForegroundColor Green
Write-Host ""

# 結果の可視化
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "結果のグラフ化" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python plot_beta_comparison.py --results-dir $RESULTS_DIR

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "すべての実験が完了しました！" -ForegroundColor Green
    Write-Host "結果は $RESULTS_DIR フォルダに保存されています" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "グラフ化でエラーが発生しました" -ForegroundColor Red
}
