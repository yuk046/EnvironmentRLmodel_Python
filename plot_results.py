#!/usr/bin/env python
"""
並列実行されたβ値ごとのシミュレーション結果を統合してグラフ化するスクリプト
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(result_dir="./beta_results_parallel"):
    """結果ディレクトリからすべてのJSONファイルを読み込む"""
    result_files = sorted(glob.glob(f"{result_dir}/beta_*.json"))
    
    if not result_files:
        print(f"Error: No result files found in {result_dir}")
        return None
    
    all_data = []
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    
    return all_data


def plot_results(all_data, save_path="addiction_result.png"):
    """結果をプロットする"""
    # データを整理
    beta_values = []
    mean_rates = []
    all_run_rates = []
    
    for data in all_data:
        beta_values.extend(data["beta_values"])
        mean_rates.extend(data["mean_rates"])
        all_run_rates.extend(data["run_rates"])
    
    # β値でソート
    sorted_indices = np.argsort(beta_values)
    beta_values = [beta_values[i] for i in sorted_indices]
    mean_rates = [mean_rates[i] for i in sorted_indices]
    all_run_rates = [all_run_rates[i] for i in sorted_indices]
    
    # 統計量を計算
    means = np.array(mean_rates)
    run_matrix = np.array(all_run_rates)  # shape: (num_betas, num_runs)
    
    # SEMと95%CIを計算
    if run_matrix.shape[1] > 1:
        sems = np.std(run_matrix, axis=1, ddof=1) / np.sqrt(run_matrix.shape[1])
        # 95%信頼区間: t値を使用（サンプル数が少ない場合に適切）
        from scipy import stats
        n = run_matrix.shape[1]
        t_value = stats.t.ppf(0.975, n - 1)  # 95% CI (両側)
        ci_95 = t_value * sems
    else:
        sems = np.zeros_like(means)
        ci_95 = np.zeros_like(means)
    
    # プロット設定
    plt.figure(figsize=(8, 5))
    
    # 薄い点で各runの値をプロット（軽く横にジッタ）
    for i, vals in enumerate(run_matrix):
        if len(vals) > 0:
            x = np.full(len(vals), beta_values[i]) + (np.random.random(len(vals)) - 0.5) * 0.01
            plt.scatter(x, vals, color='gray', alpha=0.25, s=10)
    
    # 平均 ± 95%CI を太い色強い線で描画
    plt.errorbar(
        beta_values, means, yerr=ci_95, 
        fmt='-o', color='C0', ecolor='C0', 
        elinewidth=1.5, capsize=4, linewidth=3, 
        markersize=6, label='Mean ± 95% CI'
    )
    
    # グラフの装飾
    num_agents = all_data[0]["num_agents"]
    num_runs = all_data[0]["num_runs"]
    total_agents = num_agents * num_runs * len(beta_values)
    
    plt.title(f"Transition to Addiction (N={total_agents})")
    plt.xlabel("Degree of MB Control (Beta)")
    plt.ylabel("Addiction Rate (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # 画像を保存
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    
    # 結果をテキストで表示
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for i, beta in enumerate(beta_values):
        print(f"Beta={beta:.1f} => Addiction Rate={means[i]:.2f}% ± {ci_95[i]:.2f}% (95% CI)")
        if run_matrix.shape[1] > 1:
            print(f"           SEM={sems[i]:.2f}%")
    print("="*60)
    
    # グラフを表示
    plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot results from parallel beta simulations"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./beta_results_parallel",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="addiction_result.png",
        help="Output image file path"
    )
    args = parser.parse_args()
    
    # 結果を読み込み
    print(f"Loading results from {args.input_dir}...")
    all_data = load_results(args.input_dir)
    
    if all_data is None:
        return
    
    print(f"Loaded {len(all_data)} result files")
    
    # プロット
    plot_results(all_data, args.output)


if __name__ == "__main__":
    main()
