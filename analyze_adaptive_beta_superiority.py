"""
不確実性ベース適応βの優位性分析
======================================
詳細な統計分析と可視化により、適応的β戦略の優れた性能を実証
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import argparse
from typing import List, Dict
import sys

# Import data structures from experiment script
sys.path.insert(0, '.')
from large_scale_volatility_experiment import ExperimentCondition, StatisticalComparison, AgentPerformance

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5


def analyze_beta_dynamics(data, output_dir):
    """βの時間変化と環境変化への応答を分析"""
    conditions = data['conditions']
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    for idx, vol_interval in enumerate(volatility_intervals):
        # Adaptive beta condition
        adaptive_cond = next(c for c in conditions 
                           if c.beta_type == 'adaptive' and c.volatility_interval == vol_interval)
        
        # サンプルエージェントのβ履歴を取得
        sample_agents = adaptive_cond.agent_performances[:5]  # 最初の5エージェント
        
        # β時系列
        ax1 = fig.add_subplot(gs[idx, 0])
        for i, agent in enumerate(sample_agents):
            if hasattr(agent, 'beta_history') and len(agent.beta_history) > 0:
                steps = range(len(agent.beta_history))
                ax1.plot(steps, agent.beta_history, alpha=0.6, linewidth=1, 
                        label=f'Agent {i+1}')
                
                # 環境変化点をマーク
                for change_point in agent.env_change_points:
                    ax1.axvline(change_point, color='red', alpha=0.3, linewidth=0.5, linestyle='--')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('β value')
        ax1.set_title(f'β Dynamics (Volatility={vol_interval})')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax1.legend(fontsize=8, loc='upper right')
        
        # 平均βの分布
        ax2 = fig.add_subplot(gs[idx, 1])
        all_betas = []
        for agent in adaptive_cond.agent_performances:
            if hasattr(agent, 'beta_history') and len(agent.beta_history) > 0:
                all_betas.extend(agent.beta_history)
        
        if all_betas:
            ax2.hist(all_betas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(all_betas), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean={np.mean(all_betas):.3f}')
            ax2.set_xlabel('β value')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'β Distribution (Vol={vol_interval})')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # 環境変化前後のβ変化
        ax3 = fig.add_subplot(gs[idx, 2])
        pre_change_betas = []
        post_change_betas = []
        
        for agent in adaptive_cond.agent_performances:
            if not hasattr(agent, 'beta_history') or not hasattr(agent, 'env_change_points'):
                continue
            if len(agent.beta_history) == 0 or len(agent.env_change_points) == 0:
                continue
                
            for change_point in agent.env_change_points:
                if change_point >= 10 and change_point + 20 < len(agent.beta_history):
                    pre_betas = agent.beta_history[max(0, change_point-10):change_point]
                    post_betas = agent.beta_history[change_point:min(len(agent.beta_history), change_point+20)]
                    
                    if pre_betas and post_betas:
                        pre_change_betas.append(np.mean(pre_betas))
                        post_change_betas.append(np.mean(post_betas[:10]))  # 最初の10ステップ
        
        if pre_change_betas and post_change_betas:
            positions = [1, 2]
            bp = ax3.boxplot([pre_change_betas, post_change_betas], positions=positions,
                            widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
            ax3.set_xticks(positions)
            ax3.set_xticklabels(['Pre-Change\n(10 steps)', 'Post-Change\n(10 steps)'])
            ax3.set_ylabel('β value')
            ax3.set_title(f'β Response to Change (Vol={vol_interval})')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 統計検定
            t_stat, p_val = stats.ttest_rel(pre_change_betas, post_change_betas)
            sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            ax3.text(0.5, 0.95, f'Paired t-test: p={p_val:.4f} {sig_marker}',
                    transform=ax3.transAxes, ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(f'{output_dir}/analysis1_beta_dynamics.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/analysis1_beta_dynamics.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/analysis1_beta_dynamics.png/pdf")
    plt.close()


def analyze_recovery_performance(data, output_dir):
    """環境変化後の回復性能を詳細分析"""
    conditions = data['conditions']
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    for idx, vol_interval in enumerate(volatility_intervals):
        vol_conditions = [c for c in conditions if c.volatility_interval == vol_interval]
        
        # 回復曲線（環境変化後の報酬推移）
        ax1 = fig.add_subplot(gs[idx // 3, idx % 3])
        
        adaptive_cond = next(c for c in vol_conditions if c.beta_type == 'adaptive')
        fixed_06_cond = next((c for c in vol_conditions if c.fixed_beta == 0.6), None)
        fixed_08_cond = next((c for c in vol_conditions if c.fixed_beta == 0.8), None)
        
        def get_post_change_curve(cond, window=30):
            all_curves = []
            for agent in cond.agent_performances:
                if not hasattr(agent, 'post_change_rewards'):
                    continue
                for post_rewards in agent.post_change_rewards:
                    if len(post_rewards) >= window:
                        all_curves.append(post_rewards[:window])
            
            if all_curves:
                curves_array = np.array(all_curves)
                mean_curve = np.mean(curves_array, axis=0)
                sem_curve = stats.sem(curves_array, axis=0)
                return mean_curve, sem_curve
            return None, None
        
        window_size = 30
        steps = range(window_size)
        
        # Adaptive
        mean_ada, sem_ada = get_post_change_curve(adaptive_cond, window_size)
        if mean_ada is not None:
            ax1.plot(steps, mean_ada, label='Adaptive β', linewidth=2, color='blue')
            ax1.fill_between(steps, mean_ada - sem_ada, mean_ada + sem_ada, 
                           alpha=0.3, color='blue')
        
        # Fixed 0.6
        if fixed_06_cond:
            mean_06, sem_06 = get_post_change_curve(fixed_06_cond, window_size)
            if mean_06 is not None:
                ax1.plot(steps, mean_06, label='Fixed β=0.6', linewidth=2, 
                        color='green', linestyle='--')
                ax1.fill_between(steps, mean_06 - sem_06, mean_06 + sem_06,
                               alpha=0.2, color='green')
        
        # Fixed 0.8
        if fixed_08_cond:
            mean_08, sem_08 = get_post_change_curve(fixed_08_cond, window_size)
            if mean_08 is not None:
                ax1.plot(steps, mean_08, label='Fixed β=0.8', linewidth=2,
                        color='orange', linestyle=':')
                ax1.fill_between(steps, mean_08 - sem_08, mean_08 + sem_08,
                               alpha=0.2, color='orange')
        
        ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Steps after environment change')
        ax1.set_ylabel('Average reward')
        ax1.set_title(f'Recovery Curve (Volatility={vol_interval})')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/analysis2_recovery_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/analysis2_recovery_curves.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/analysis2_recovery_curves.png/pdf")
    plt.close()


def analyze_learning_efficiency(data, output_dir):
    """学習効率とパフォーマンスの関係を分析"""
    conditions = data['conditions']
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    for idx, vol_interval in enumerate(volatility_intervals):
        ax = fig.add_subplot(gs[0, idx])
        
        vol_conditions = [c for c in conditions if c.volatility_interval == vol_interval]
        
        beta_values = []
        mean_rewards = []
        recovery_speeds = []
        labels = []
        colors = []
        
        for cond in sorted(vol_conditions, key=lambda x: (x.beta_type != 'adaptive', x.fixed_beta or -1)):
            if cond.beta_type == 'adaptive':
                beta_values.append(-0.1)  # 視覚的に左端に配置
                labels.append('Adaptive')
                colors.append('red')
            else:
                beta_values.append(cond.fixed_beta)
                labels.append(f'{cond.fixed_beta:.1f}')
                colors.append('blue')
            
            mean_rewards.append(cond.mean_total_reward)
            recovery_speeds.append(cond.mean_recovery_speed)
        
        # 散布図: β値 vs 総報酬
        scatter = ax.scatter(beta_values, mean_rewards, s=100, c=colors, alpha=0.7, edgecolors='black')
        
        # ラベル
        for i, (bv, mr, label) in enumerate(zip(beta_values, mean_rewards, labels)):
            ax.annotate(label, (bv, mr), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('β value (Adaptive at -0.1)')
        ax.set_ylabel('Mean Total Reward')
        ax.set_title(f'β vs Performance (Vol={vol_interval})')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Adaptiveを強調
        adaptive_idx = labels.index('Adaptive')
        ax.scatter([beta_values[adaptive_idx]], [mean_rewards[adaptive_idx]], 
                  s=200, c='red', marker='*', edgecolors='black', linewidth=2, zorder=10)
    
    plt.savefig(f'{output_dir}/analysis3_beta_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/analysis3_beta_performance.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/analysis3_beta_performance.png/pdf")
    plt.close()


def analyze_stability_vs_adaptability(data, output_dir):
    """安定性と適応性のトレードオフを分析"""
    conditions = data['conditions']
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    for idx, vol_interval in enumerate(volatility_intervals):
        ax = fig.add_subplot(gs[0, idx])
        
        vol_conditions = [c for c in conditions if c.volatility_interval == vol_interval]
        
        stability_scores = []  # 標準偏差の逆数（低いほど安定）
        adaptability_scores = []  # 回復速度
        labels = []
        colors = []
        
        for cond in vol_conditions:
            # 安定性: 分散の逆数（正規化）
            stability = 1000.0 / (cond.std_total_reward + 1.0)
            stability_scores.append(stability)
            
            # 適応性: 回復速度
            adaptability_scores.append(cond.mean_recovery_speed)
            
            if cond.beta_type == 'adaptive':
                labels.append('Adaptive')
                colors.append('red')
            else:
                labels.append(f'β={cond.fixed_beta:.1f}')
                colors.append('blue')
        
        # 散布図
        scatter = ax.scatter(stability_scores, adaptability_scores, s=100, 
                           c=colors, alpha=0.7, edgecolors='black')
        
        # ラベル
        for i, (stab, ada, label) in enumerate(zip(stability_scores, adaptability_scores, labels)):
            ax.annotate(label, (stab, ada), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Stability (1000/Std)')
        ax.set_ylabel('Adaptability (Recovery Speed)')
        ax.set_title(f'Stability vs Adaptability (Vol={vol_interval})')
        ax.grid(True, alpha=0.3)
        
        # Adaptiveを強調
        adaptive_idx = labels.index('Adaptive')
        ax.scatter([stability_scores[adaptive_idx]], [adaptability_scores[adaptive_idx]],
                  s=200, c='red', marker='*', edgecolors='black', linewidth=2, zorder=10)
    
    plt.savefig(f'{output_dir}/analysis4_stability_adaptability.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/analysis4_stability_adaptability.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/analysis4_stability_adaptability.png/pdf")
    plt.close()


def create_superiority_summary_table(data, output_dir):
    """優位性を示すサマリーテーブルを作成"""
    conditions = data['conditions']
    comparisons = data['comparisons']
    
    # CSVレポート
    summary_data = []
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    for vol_interval in volatility_intervals:
        vol_conditions = [c for c in conditions if c.volatility_interval == vol_interval]
        adaptive_cond = next(c for c in vol_conditions if c.beta_type == 'adaptive')
        
        # ランキング
        sorted_conds = sorted(vol_conditions, key=lambda x: x.mean_total_reward, reverse=True)
        adaptive_rank = sorted_conds.index(adaptive_cond) + 1
        
        # 勝利数
        wins = 0
        significant_wins = 0
        
        for comp in comparisons:
            if comp.condition_a == adaptive_cond.condition_name:
                if comp.mean_difference > 0:
                    wins += 1
                    if comp.significant:
                        significant_wins += 1
        
        summary_data.append({
            'Volatility_Interval': vol_interval,
            'Adaptive_Mean_Reward': adaptive_cond.mean_total_reward,
            'Adaptive_Std': adaptive_cond.std_total_reward,
            'Adaptive_Recovery_Speed': adaptive_cond.mean_recovery_speed,
            'Rank': f'{adaptive_rank}/{len(vol_conditions)}',
            'Total_Wins': f'{wins}/6',
            'Significant_Wins': f'{significant_wins}/6',
            'Best_Fixed_Beta': sorted_conds[0].fixed_beta if sorted_conds[0].beta_type == 'fixed' else 'N/A',
            'Best_Fixed_Reward': sorted_conds[0].mean_total_reward if sorted_conds[0].beta_type == 'fixed' else sorted_conds[1].mean_total_reward,
            'Advantage_vs_Best': adaptive_cond.mean_total_reward - (sorted_conds[0].mean_total_reward if sorted_conds[0].beta_type == 'fixed' else sorted_conds[1].mean_total_reward)
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{output_dir}/adaptive_superiority_summary.csv', index=False)
    print(f"\nSaved: {output_dir}/adaptive_superiority_summary.csv")
    print("\nAdaptive β Superiority Summary:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Analyze adaptive beta superiority')
    parser.add_argument('--input-dir', type=str, default='uncertainty_beta_results',
                       help='Input directory with experiment results')
    parser.add_argument('--output-dir', type=str, default='uncertainty_beta_results',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ADAPTIVE BETA SUPERIORITY ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    with open(f'{args.input_dir}/full_experiment_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data['conditions'])} conditions")
    
    # Run analyses
    print("\nAnalyzing beta dynamics...")
    analyze_beta_dynamics(data, args.output_dir)
    
    print("\nAnalyzing recovery performance...")
    analyze_recovery_performance(data, args.output_dir)
    
    print("\nAnalyzing learning efficiency...")
    analyze_learning_efficiency(data, args.output_dir)
    
    print("\nAnalyzing stability vs adaptability...")
    analyze_stability_vs_adaptability(data, args.output_dir)
    
    print("\nCreating superiority summary...")
    create_superiority_summary_table(data, args.output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nGenerated files in {args.output_dir}:")
    print("  - analysis1_beta_dynamics.png/pdf")
    print("  - analysis2_recovery_curves.png/pdf")
    print("  - analysis3_beta_performance.png/pdf")
    print("  - analysis4_stability_adaptability.png/pdf")
    print("  - adaptive_superiority_summary.csv")


if __name__ == "__main__":
    main()
