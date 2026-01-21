"""
β固定 vs 適応的β学習の比較実験スクリプト
最終発表用の包括的な分析を実行
"""

import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
from scipy import stats

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 実験パラメータ
BETA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NUM_AGENTS = 900  # 1回あたりのエージェント数
NUM_RUNS = 30     # 統計的信頼性のための繰り返し回数
BASE_SEED = 42
OUTPUT_DIR = Path("beta_comparison_final")

def run_experiment(beta_value=None, num_agents=900, num_runs=30, seed=42):
    """
    単一の実験を実行
    
    Args:
        beta_value: 固定β値（Noneの場合は適応的学習）
        num_agents: エージェント数
        num_runs: 繰り返し回数
        seed: 乱数シード
    
    Returns:
        dict: 実験結果
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if beta_value is not None:
        label = f"beta_{beta_value:.1f}"
        output_file = OUTPUT_DIR / f"{label}.json"
        cmd = [
            "python", "addiction_rl_sim.py",
            "--num-agents", str(num_agents),
            "--num-runs", str(num_runs),
            "--seed", str(seed),
            "--fixed-beta", str(beta_value),
            "--output", str(output_file)
        ]
        print(f"\n{'='*60}")
        print(f"実験実行中: β = {beta_value} (固定)")
        print(f"{'='*60}")
    else:
        label = "beta_adaptive"
        output_file = OUTPUT_DIR / f"{label}.json"
        cmd = [
            "python", "addiction_rl_sim.py",
            "--num-agents", str(num_agents),
            "--num-runs", str(num_runs),
            "--seed", str(seed),
            "--output", str(output_file),
            "--plot-beta"
        ]
        print(f"\n{'='*60}")
        print(f"実験実行中: 適応的β学習")
        print(f"{'='*60}")
    
    subprocess.run(cmd, check=True)
    
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    return results

def analyze_and_visualize(results_dict):
    """
    全実験結果を統合分析・可視化
    
    Args:
        results_dict: {label: results} の辞書
    """
    print(f"\n{'='*60}")
    print("統合分析と可視化を開始")
    print(f"{'='*60}\n")
    
    # ===== Figure 1: β値と依存率の関係（添付画像1の再現） =====
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    beta_vals = []
    addiction_means = []
    addiction_sems = []
    
    for beta in BETA_VALUES:
        label = f"beta_{beta:.1f}"
        if label in results_dict:
            results = results_dict[label]
            run_rates = results['run_rates']
            
            beta_vals.append(beta)
            mean_rate = np.mean(run_rates)
            sem_rate = stats.sem(run_rates)
            
            addiction_means.append(mean_rate)
            addiction_sems.append(sem_rate * 1.96)  # 95% CI
    
    # プロット
    ax1.errorbar(beta_vals, addiction_means, yerr=addiction_sems,
                marker='o', markersize=12, linewidth=3, capsize=8, capthick=2,
                color='#2e86de', label='Mean ± 95% CI')
    
    ax1.set_xlabel('Degree of MB Control (Beta)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Addiction Rate (%)', fontsize=16, fontweight='bold')
    ax1.set_title(f'Transition to Addiction (N={NUM_AGENTS * NUM_RUNS})', 
                 fontsize=18, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_xticks(BETA_VALUES)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=14, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'addiction_rate_vs_beta.png', dpi=300, bbox_inches='tight')
    print(f"保存完了: {OUTPUT_DIR / 'addiction_rate_vs_beta.png'}")
    plt.close()
    
    # ===== Figure 2: 統計的比較表 =====
    print("\n" + "="*80)
    print("統計分析結果")
    print("="*80)
    print(f"{'Beta':>8} {'依存率(%)':>12} {'95% CI':>20} {'サンプル数':>12}")
    print("-"*80)
    
    for beta in BETA_VALUES:
        label = f"beta_{beta:.1f}"
        if label in results_dict:
            results = results_dict[label]
            run_rates = results['run_rates']
            mean_rate = np.mean(run_rates)
            sem_rate = stats.sem(run_rates)
            ci_lower = mean_rate - 1.96 * sem_rate
            ci_upper = mean_rate + 1.96 * sem_rate
            n_total = results['num_agents'] * results['num_runs']
            
            print(f"{beta:>8.1f} {mean_rate:>12.2f} [{ci_lower:>6.2f}, {ci_upper:>6.2f}] {n_total:>12}")
    
    # 適応的学習の結果
    if "beta_adaptive" in results_dict:
        results = results_dict["beta_adaptive"]
        run_rates = results['run_rates']
        mean_rate = np.mean(run_rates)
        sem_rate = stats.sem(run_rates)
        ci_lower = mean_rate - 1.96 * sem_rate
        ci_upper = mean_rate + 1.96 * sem_rate
        n_total = results['num_agents'] * results['num_runs']
        
        print("-"*80)
        print(f"{'適応的':>8} {mean_rate:>12.2f} [{ci_lower:>6.2f}, {ci_upper:>6.2f}] {n_total:>12}")
    
    print("="*80)
    
    # ===== Figure 3: β固定vs適応的学習の詳細比較 =====
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    
    # (1) 各β値での依存率分布（バイオリンプロット）
    ax3_1 = axes3[0, 0]
    
    data_for_violin = []
    labels_for_violin = []
    
    for beta in BETA_VALUES:
        label = f"beta_{beta:.1f}"
        if label in results_dict:
            data_for_violin.append(results_dict[label]['run_rates'])
            labels_for_violin.append(f'β={beta:.1f}')
    
    if "beta_adaptive" in results_dict:
        data_for_violin.append(results_dict["beta_adaptive"]['run_rates'])
        labels_for_violin.append('適応的')
    
    parts = ax3_1.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                             showmeans=True, showmedians=True)
    
    ax3_1.set_xticks(range(len(labels_for_violin)))
    ax3_1.set_xticklabels(labels_for_violin, rotation=45, ha='right')
    ax3_1.set_ylabel('依存率 (%)', fontsize=12, fontweight='bold')
    ax3_1.set_title('依存率の分布比較', fontsize=13, fontweight='bold')
    ax3_1.grid(True, alpha=0.3, axis='y')
    
    # (2) 分散の比較
    ax3_2 = axes3[0, 1]
    
    variances = []
    beta_labels = []
    
    for beta in BETA_VALUES:
        label = f"beta_{beta:.1f}"
        if label in results_dict:
            variances.append(np.var(results_dict[label]['run_rates']))
            beta_labels.append(f'β={beta:.1f}')
    
    if "beta_adaptive" in results_dict:
        variances.append(np.var(results_dict["beta_adaptive"]['run_rates']))
        beta_labels.append('適応的')
    
    colors = ['#3498db'] * len(BETA_VALUES) + ['#e74c3c']
    bars = ax3_2.bar(range(len(variances)), variances, color=colors, alpha=0.7)
    ax3_2.set_xticks(range(len(beta_labels)))
    ax3_2.set_xticklabels(beta_labels, rotation=45, ha='right')
    ax3_2.set_ylabel('分散', fontsize=12, fontweight='bold')
    ax3_2.set_title('依存率の分散比較', fontsize=13, fontweight='bold')
    ax3_2.grid(True, alpha=0.3, axis='y')
    
    # (3) 各β値と適応的学習の差分
    ax3_3 = axes3[1, 0]
    
    if "beta_adaptive" in results_dict:
        adaptive_mean = np.mean(results_dict["beta_adaptive"]['run_rates'])
        
        diffs = []
        for beta in BETA_VALUES:
            label = f"beta_{beta:.1f}"
            if label in results_dict:
                beta_mean = np.mean(results_dict[label]['run_rates'])
                diffs.append(beta_mean - adaptive_mean)
        
        colors_diff = ['#e74c3c' if d > 0 else '#2ecc71' for d in diffs]
        bars_diff = ax3_3.bar(BETA_VALUES, diffs, color=colors_diff, alpha=0.7, width=0.15)
        ax3_3.axhline(0, color='black', linewidth=2)
        ax3_3.set_xlabel('β値（固定）', fontsize=12, fontweight='bold')
        ax3_3.set_ylabel('依存率の差（固定 - 適応的） (%)', fontsize=12, fontweight='bold')
        ax3_3.set_title(f'適応的学習との比較（適応的={adaptive_mean:.2f}%）', 
                       fontsize=13, fontweight='bold')
        ax3_3.set_xticks(BETA_VALUES)
        ax3_3.grid(True, alpha=0.3, axis='y')
        
        # 値をバーに表示
        for bar, diff in zip(bars_diff, diffs):
            height = bar.get_height()
            ax3_3.text(bar.get_x() + bar.get_width()/2, height,
                      f'{diff:+.1f}', ha='center', 
                      va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    # (4) 統計的有意性検定（適応的 vs 各固定β）
    ax3_4 = axes3[1, 1]
    
    if "beta_adaptive" in results_dict:
        adaptive_data = results_dict["beta_adaptive"]['run_rates']
        
        p_values = []
        t_statistics = []
        
        for beta in BETA_VALUES:
            label = f"beta_{beta:.1f}"
            if label in results_dict:
                beta_data = results_dict[label]['run_rates']
                t_stat, p_val = stats.ttest_ind(beta_data, adaptive_data)
                p_values.append(p_val)
                t_statistics.append(t_stat)
        
        # p値のプロット（対数スケール）
        ax3_4.bar(BETA_VALUES, p_values, color='#9b59b6', alpha=0.7, width=0.15)
        ax3_4.axhline(0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')
        ax3_4.axhline(0.01, color='darkred', linestyle='--', linewidth=2, label='p=0.01')
        ax3_4.set_xlabel('β値（固定）', fontsize=12, fontweight='bold')
        ax3_4.set_ylabel('p値（t検定）', fontsize=12, fontweight='bold')
        ax3_4.set_title('適応的学習との統計的有意差', fontsize=13, fontweight='bold')
        ax3_4.set_xticks(BETA_VALUES)
        ax3_4.set_yscale('log')
        ax3_4.legend()
        ax3_4.grid(True, alpha=0.3, axis='y')
        
        # 有意性を注釈
        for beta, p_val in zip(BETA_VALUES, p_values):
            if p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'
            ax3_4.text(beta, p_val, sig, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detailed_comparison.png', dpi=300, bbox_inches='tight')
    print(f"保存完了: {OUTPUT_DIR / 'detailed_comparison.png'}")
    plt.close()
    
    # ===== Figure 4: 効果量（Cohen's d）の計算と可視化 =====
    if "beta_adaptive" in results_dict:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        
        adaptive_data = results_dict["beta_adaptive"]['run_rates']
        adaptive_mean = np.mean(adaptive_data)
        adaptive_std = np.std(adaptive_data, ddof=1)
        
        cohens_d_values = []
        
        for beta in BETA_VALUES:
            label = f"beta_{beta:.1f}"
            if label in results_dict:
                beta_data = results_dict[label]['run_rates']
                beta_mean = np.mean(beta_data)
                beta_std = np.std(beta_data, ddof=1)
                
                # Cohen's d の計算
                pooled_std = np.sqrt((beta_std**2 + adaptive_std**2) / 2)
                cohens_d = (beta_mean - adaptive_mean) / pooled_std
                cohens_d_values.append(cohens_d)
        
        # プロット
        colors_d = ['#e74c3c' if d > 0 else '#2ecc71' for d in cohens_d_values]
        bars_d = ax4.bar(BETA_VALUES, cohens_d_values, color=colors_d, alpha=0.7, width=0.15)
        ax4.axhline(0, color='black', linewidth=2)
        ax4.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='小効果')
        ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='中効果')
        ax4.axhline(0.8, color='gray', linestyle='--', alpha=0.9, label='大効果')
        ax4.axhline(-0.2, color='gray', linestyle='--', alpha=0.5)
        ax4.axhline(-0.5, color='gray', linestyle='--', alpha=0.7)
        ax4.axhline(-0.8, color='gray', linestyle='--', alpha=0.9)
        
        ax4.set_xlabel('β値（固定）', fontsize=14, fontweight='bold')
        ax4.set_ylabel("Cohen's d（効果量）", fontsize=14, fontweight='bold')
        ax4.set_title('適応的学習に対する効果量（固定β - 適応的）', fontsize=16, fontweight='bold')
        ax4.set_xticks(BETA_VALUES)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 値をバーに表示
        for bar, d_val in zip(bars_d, cohens_d_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height,
                    f'{d_val:+.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'effect_sizes.png', dpi=300, bbox_inches='tight')
        print(f"保存完了: {OUTPUT_DIR / 'effect_sizes.png'}")
        plt.close()
        
        # 効果量の統計を出力
        print("\n" + "="*80)
        print("効果量分析（Cohen's d）")
        print("="*80)
        print(f"{'Beta':>8} {'効果量(d)':>12} {'解釈':>20}")
        print("-"*80)
        
        for beta, d_val in zip(BETA_VALUES, cohens_d_values):
            if abs(d_val) < 0.2:
                interpretation = "無視可能"
            elif abs(d_val) < 0.5:
                interpretation = "小効果"
            elif abs(d_val) < 0.8:
                interpretation = "中効果"
            else:
                interpretation = "大効果"
            
            print(f"{beta:>8.1f} {d_val:>12.2f} {interpretation:>20}")
        
        print("="*80)

def main():
    """メイン実行関数"""
    print("\n" + "="*80)
    print("β固定 vs 適応的β学習：包括的比較実験")
    print("="*80)
    print(f"エージェント数: {NUM_AGENTS}")
    print(f"繰り返し回数: {NUM_RUNS}")
    print(f"合計サンプル: {NUM_AGENTS * NUM_RUNS}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    results_dict = {}
    
    # 固定β値での実験
    for beta in BETA_VALUES:
        results = run_experiment(
            beta_value=beta,
            num_agents=NUM_AGENTS,
            num_runs=NUM_RUNS,
            seed=BASE_SEED
        )
        label = f"beta_{beta:.1f}"
        results_dict[label] = results
    
    # 適応的β学習での実験
    results = run_experiment(
        beta_value=None,  # 適応的学習
        num_agents=NUM_AGENTS,
        num_runs=NUM_RUNS,
        seed=BASE_SEED
    )
    results_dict["beta_adaptive"] = results
    
    # 統合分析と可視化
    analyze_and_visualize(results_dict)
    
    print("\n" + "="*80)
    print("全実験完了！")
    print(f"結果は {OUTPUT_DIR} に保存されました")
    print("="*80)

if __name__ == "__main__":
    main()
