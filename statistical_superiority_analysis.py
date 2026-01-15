"""
適応的βの統計的優位性の詳細レポート
====================================
論文用の包括的な統計分析と効果量の計算
"""

import pickle
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys
import argparse

sys.path.insert(0, '.')
from large_scale_volatility_experiment import ExperimentCondition, StatisticalComparison, AgentPerformance

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def calculate_effect_sizes_detailed(data, output_dir):
    """詳細な効果量分析"""
    conditions = data['conditions']
    comparisons = data['comparisons']
    
    print("\n" + "="*70)
    print("EFFECT SIZE ANALYSIS (Cohen's d)")
    print("="*70)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    results = []
    
    for vol_interval in volatility_intervals:
        print(f"\n--- Volatility Interval: {vol_interval} ---")
        
        adaptive_cond = next(c for c in conditions 
                           if c.beta_type == 'adaptive' and c.volatility_interval == vol_interval)
        
        adaptive_rewards = [p.total_reward for p in adaptive_cond.agent_performances]
        
        for beta_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            fixed_cond = next((c for c in conditions 
                             if c.beta_type == 'fixed' and c.fixed_beta == beta_val 
                             and c.volatility_interval == vol_interval), None)
            
            if fixed_cond is None:
                continue
            
            fixed_rewards = [p.total_reward for p in fixed_cond.agent_performances]
            
            # 効果量計算
            pooled_std = np.sqrt((np.std(adaptive_rewards, ddof=1)**2 + 
                                 np.std(fixed_rewards, ddof=1)**2) / 2)
            cohens_d = (np.mean(adaptive_rewards) - np.mean(fixed_rewards)) / pooled_std
            
            # t検定
            t_stat, p_val = stats.ttest_ind(adaptive_rewards, fixed_rewards)
            
            # 効果量の解釈
            if abs(cohens_d) < 0.2:
                interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                interpretation = "small"
            elif abs(cohens_d) < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            
            advantage = "ADAPTIVE" if cohens_d > 0 else "FIXED"
            
            print(f"  vs β={beta_val:.1f}: d={cohens_d:+.3f} ({interpretation:>10s}) p={p_val:.4f}{sig_marker:>4s} → {advantage}")
            
            results.append({
                'Volatility': vol_interval,
                'Fixed_Beta': beta_val,
                'Cohens_d': cohens_d,
                'Interpretation': interpretation,
                'p_value': p_val,
                'Significant': p_val < 0.05,
                'Advantage': advantage,
                'Mean_Diff': np.mean(adaptive_rewards) - np.mean(fixed_rewards)
            })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/effect_sizes_detailed.csv', index=False)
    print(f"\nSaved: {output_dir}/effect_sizes_detailed.csv")
    
    return df


def analyze_robustness(data, output_dir):
    """頑健性分析: 分散と外れ値への耐性"""
    conditions = data['conditions']
    
    print("\n" + "="*70)
    print("ROBUSTNESS ANALYSIS")
    print("="*70)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    results = []
    
    for vol_interval in volatility_intervals:
        print(f"\n--- Volatility Interval: {vol_interval} ---")
        
        vol_conditions = [c for c in conditions if c.volatility_interval == vol_interval]
        
        for cond in vol_conditions:
            rewards = [p.total_reward for p in cond.agent_performances]
            
            # 統計量計算
            mean_reward = np.mean(rewards)
            median_reward = np.median(rewards)
            std_reward = np.std(rewards, ddof=1)
            cv = std_reward / abs(mean_reward) if mean_reward != 0 else np.inf  # 変動係数
            q25, q75 = np.percentile(rewards, [25, 75])
            iqr = q75 - q25
            
            # 外れ値の検出
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = [r for r in rewards if r < lower_bound or r > upper_bound]
            outlier_pct = len(outliers) / len(rewards) * 100
            
            label = f"Adaptive" if cond.beta_type == 'adaptive' else f"β={cond.fixed_beta:.1f}"
            print(f"  {label:>12s}: CV={cv:.3f}, IQR={iqr:.1f}, Outliers={outlier_pct:.1f}%")
            
            results.append({
                'Volatility': vol_interval,
                'Condition': label,
                'Beta_Type': cond.beta_type,
                'Mean': mean_reward,
                'Median': median_reward,
                'Std': std_reward,
                'CV': cv,
                'IQR': iqr,
                'Outlier_Pct': outlier_pct,
                'Q25': q25,
                'Q75': q75
            })
    
    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/robustness_analysis.csv', index=False)
    print(f"\nSaved: {output_dir}/robustness_analysis.csv")
    
    return df


def comparative_ranking_analysis(data, output_dir):
    """各条件でのランキング分析"""
    conditions = data['conditions']
    
    print("\n" + "="*70)
    print("COMPARATIVE RANKING ANALYSIS")
    print("="*70)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    results = []
    
    for vol_interval in volatility_intervals:
        print(f"\n--- Volatility Interval: {vol_interval} ---")
        print(f"{'Rank':<6} {'Condition':<20} {'Mean Reward':<15} {'Recovery Speed':<15}")
        print("-" * 60)
        
        vol_conditions = [c for c in conditions if c.volatility_interval == vol_interval]
        sorted_conds = sorted(vol_conditions, key=lambda x: x.mean_total_reward, reverse=True)
        
        for rank, cond in enumerate(sorted_conds, 1):
            label = "Adaptive β" if cond.beta_type == 'adaptive' else f"Fixed β={cond.fixed_beta:.1f}"
            marker = " ★" if cond.beta_type == 'adaptive' else ""
            print(f"{rank:<6} {label:<20} {cond.mean_total_reward:>14.2f} {cond.mean_recovery_speed:>14.4f}{marker}")
            
            results.append({
                'Volatility': vol_interval,
                'Rank': rank,
                'Condition': label,
                'Beta_Type': cond.beta_type,
                'Mean_Reward': cond.mean_total_reward,
                'Std_Reward': cond.std_total_reward,
                'Recovery_Speed': cond.mean_recovery_speed
            })
    
    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/ranking_analysis.csv', index=False)
    print(f"\nSaved: {output_dir}/ranking_analysis.csv")
    
    return df


def win_rate_analysis(data, output_dir):
    """勝率分析: ペアワイズ比較での勝利数"""
    conditions = data['conditions']
    comparisons = data['comparisons']
    
    print("\n" + "="*70)
    print("WIN RATE ANALYSIS")
    print("="*70)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    overall_results = []
    
    for vol_interval in volatility_intervals:
        print(f"\n--- Volatility Interval: {vol_interval} ---")
        
        adaptive_cond = next(c for c in conditions 
                           if c.beta_type == 'adaptive' and c.volatility_interval == vol_interval)
        
        wins = 0
        sig_wins = 0
        total = 0
        
        win_details = []
        
        for comp in comparisons:
            if comp.condition_a == adaptive_cond.condition_name:
                total += 1
                if comp.mean_difference > 0:
                    wins += 1
                    if comp.significant:
                        sig_wins += 1
                        sig_marker = "***" if comp.p_value < 0.001 else ("**" if comp.p_value < 0.01 else "*")
                        win_details.append(f"vs {comp.condition_b.split('_')[1]}: +{comp.mean_difference:.1f} {sig_marker}")
        
        win_rate = wins / total * 100 if total > 0 else 0
        sig_win_rate = sig_wins / total * 100 if total > 0 else 0
        
        print(f"  Total Wins: {wins}/{total} ({win_rate:.1f}%)")
        print(f"  Significant Wins: {sig_wins}/{total} ({sig_win_rate:.1f}%)")
        
        if win_details:
            print(f"  Significant victories:")
            for detail in win_details:
                print(f"    - {detail}")
        
        overall_results.append({
            'Volatility': vol_interval,
            'Total_Comparisons': total,
            'Wins': wins,
            'Win_Rate_Pct': win_rate,
            'Significant_Wins': sig_wins,
            'Sig_Win_Rate_Pct': sig_win_rate
        })
    
    df = pd.DataFrame(overall_results)
    df.to_csv(f'{output_dir}/win_rate_analysis.csv', index=False)
    print(f"\nSaved: {output_dir}/win_rate_analysis.csv")
    
    return df


def create_comprehensive_summary(data, output_dir):
    """包括的なサマリーレポート"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Load analyses
    effect_sizes = pd.read_csv(f'{output_dir}/effect_sizes_detailed.csv')
    robustness = pd.read_csv(f'{output_dir}/robustness_analysis.csv')
    ranking = pd.read_csv(f'{output_dir}/ranking_analysis.csv')
    win_rate = pd.read_csv(f'{output_dir}/win_rate_analysis.csv')
    
    # Plot 1: Effect sizes heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    pivot_effect = effect_sizes.pivot(index='Fixed_Beta', columns='Volatility', values='Cohens_d')
    sns.heatmap(pivot_effect, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': "Cohen's d"}, ax=ax1)
    ax1.set_title("Effect Sizes (Adaptive vs Fixed β)")
    ax1.set_ylabel("Fixed β")
    ax1.set_xlabel("Volatility Interval")
    
    # Plot 2: Win rates
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(win_rate))
    ax2.bar(x_pos - 0.2, win_rate['Win_Rate_Pct'], width=0.4, label='All Wins', alpha=0.7)
    ax2.bar(x_pos + 0.2, win_rate['Sig_Win_Rate_Pct'], width=0.4, label='Significant Wins', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Vol={v}" for v in win_rate['Volatility']])
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Adaptive β Win Rates')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)
    
    # Plot 3: Robustness comparison (CV)
    ax3 = fig.add_subplot(gs[1, :])
    adaptive_rob = robustness[robustness['Beta_Type'] == 'adaptive']
    fixed_rob = robustness[robustness['Beta_Type'] == 'fixed']
    
    for vol in sorted(adaptive_rob['Volatility'].unique()):
        ada_cv = adaptive_rob[adaptive_rob['Volatility'] == vol]['CV'].values[0]
        fixed_cvs = fixed_rob[fixed_rob['Volatility'] == vol]['CV'].values
        
        positions = [vol - 0.3] + [vol + i*0.1 for i in range(len(fixed_cvs))]
        values = [ada_cv] + list(fixed_cvs)
        colors = ['red'] + ['blue'] * len(fixed_cvs)
        
        ax3.bar(positions, values, width=0.08, color=colors, alpha=0.7)
    
    ax3.set_xlabel('Volatility Interval')
    ax3.set_ylabel('Coefficient of Variation')
    ax3.set_title('Robustness: Lower CV = More Stable Performance')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(['Adaptive', 'Fixed'], loc='upper right')
    
    # Plot 4: Rankings distribution
    ax4 = fig.add_subplot(gs[2, 0])
    adaptive_ranks = ranking[ranking['Beta_Type'] == 'adaptive']['Rank'].values
    ax4.bar(range(len(adaptive_ranks)), adaptive_ranks, color='red', alpha=0.7)
    ax4.set_xticks(range(len(adaptive_ranks)))
    ax4.set_xticklabels([f"Vol={v}" for v in sorted(ranking['Volatility'].unique())])
    ax4.set_ylabel('Rank (1=Best)')
    ax4.set_title('Adaptive β Rankings')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='1st Place')
    ax4.legend()
    
    # Plot 5: Performance improvement
    ax5 = fig.add_subplot(gs[2, 1])
    volatilities = sorted(effect_sizes['Volatility'].unique())
    improvements = []
    
    for vol in volatilities:
        vol_effects = effect_sizes[effect_sizes['Volatility'] == vol]
        avg_effect = vol_effects['Cohens_d'].mean()
        improvements.append(avg_effect)
    
    colors_imp = ['green' if x > 0 else 'red' for x in improvements]
    ax5.bar(range(len(volatilities)), improvements, color=colors_imp, alpha=0.7)
    ax5.set_xticks(range(len(volatilities)))
    ax5.set_xticklabels([f"Vol={v}" for v in volatilities])
    ax5.set_ylabel("Average Cohen's d")
    ax5.set_title('Average Effect Size (Adaptive vs All Fixed)')
    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f'{output_dir}/comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/comprehensive_summary.pdf', bbox_inches='tight')
    print(f"\nSaved: {output_dir}/comprehensive_summary.png/pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Statistical superiority analysis')
    parser.add_argument('--input-dir', type=str, default='uncertainty_beta_results')
    parser.add_argument('--output-dir', type=str, default='uncertainty_beta_results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("STATISTICAL SUPERIORITY ANALYSIS")
    print("="*70)
    
    # Load data
    with open(f'{args.input_dir}/full_experiment_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Run all analyses
    calculate_effect_sizes_detailed(data, args.output_dir)
    analyze_robustness(data, args.output_dir)
    comparative_ranking_analysis(data, args.output_dir)
    win_rate_analysis(data, args.output_dir)
    
    print("\nCreating comprehensive summary visualization...")
    create_comprehensive_summary(data, args.output_dir)
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
