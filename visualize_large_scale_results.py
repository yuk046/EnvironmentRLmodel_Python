"""
Publication-Quality Visualization for Large-Scale Experiment
============================================================
Creates comprehensive figures for publication including:
- Performance comparison with error bars
- Statistical significance markers
- Learning curves
- Recovery speed analysis
- Effect size visualization
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
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
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5


def plot_main_results(conditions, comparisons, output_dir, num_steps=8000):
    """Main publication figure: Performance comparison across conditions"""
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, vol_interval in enumerate(volatility_intervals):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        # Get conditions for this volatility interval
        conds = [c for c in conditions if c.volatility_interval == vol_interval]
        conds_sorted = sorted(conds, key=lambda x: x.mean_total_reward, reverse=True)
        
        # Prepare data
        labels = []
        means = []
        sems = []
        colors = []
        
        for cond in conds_sorted:
            if cond.beta_type == "adaptive":
                labels.append("Adaptive β")
                colors.append("#2ecc71")
            else:
                labels.append(f"β={cond.fixed_beta:.1f}")
                if cond.fixed_beta == 0.0:
                    colors.append("#e74c3c")
                elif cond.fixed_beta == 1.0:
                    colors.append("#3498db")
                else:
                    colors.append("#95a5a6")
            
            means.append(cond.mean_total_reward)
            sems.append(cond.sem_total_reward)
        
        # Plot bars
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add significance stars
        adaptive_idx = next(i for i, c in enumerate(conds_sorted) if c.beta_type == "adaptive")
        adaptive_cond = conds_sorted[adaptive_idx]
        
        for i, cond in enumerate(conds_sorted):
            if cond.beta_type != "adaptive":
                # Find comparison
                comp = next((c for c in comparisons 
                           if c.condition_a == adaptive_cond.condition_name 
                           and c.condition_b == cond.condition_name), None)
                
                if comp:
                    # Add significance markers
                    y_pos = max(means[adaptive_idx], means[i]) + max(sems[adaptive_idx], sems[i]) + 10
                    
                    if comp.p_value < 0.001:
                        sig_text = "***"
                    elif comp.p_value < 0.01:
                        sig_text = "**"
                    elif comp.p_value < 0.05:
                        sig_text = "*"
                    else:
                        sig_text = "n.s."
                    
                    if sig_text != "n.s.":
                        # Draw comparison line
                        ax.plot([adaptive_idx, i], [y_pos, y_pos], 'k-', linewidth=1)
                        ax.text((adaptive_idx + i) / 2, y_pos + 5, sig_text,
                               ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Highlight best performer
        best_idx = 0
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(f'Total Reward ({num_steps} steps)', fontsize=11, fontweight='bold')
        ax.set_title(f'Volatility Interval: {vol_interval} steps', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add sample size
        n = conds_sorted[0].num_agents
        ax.text(0.02, 0.98, f'n={n} per condition', 
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend for significance
    fig.text(0.5, 0.02, '*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant\n'
                        'Error bars: SEM (Standard Error of Mean)',
             ha='center', fontsize=9, style='italic')
    
    plt.savefig(f'{output_dir}/figure1_main_performance.png', 
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/figure1_main_performance.png")
    plt.close()


def plot_learning_curves(conditions, output_dir):
    """Plot learning curves showing reward over time windows"""
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    fig, axes = plt.subplots(1, len(volatility_intervals), figsize=(15, 5))
    if len(volatility_intervals) == 1:
        axes = [axes]
    
    for idx, vol_interval in enumerate(volatility_intervals):
        ax = axes[idx]
        
        # Get conditions
        conds = [c for c in conditions if c.volatility_interval == vol_interval]
        
        # Plot adaptive beta
        adaptive = next(c for c in conds if c.beta_type == "adaptive")
        x = np.arange(1, len(adaptive.mean_rewards_per_window) + 1)
        ax.plot(x, adaptive.mean_rewards_per_window, 'o-', 
               color='#2ecc71', linewidth=2.5, markersize=6, 
               label='Adaptive β', zorder=10)
        
        # Shade error region
        if adaptive.std_rewards_per_window:
            lower = np.array(adaptive.mean_rewards_per_window) - np.array(adaptive.std_rewards_per_window)
            upper = np.array(adaptive.mean_rewards_per_window) + np.array(adaptive.std_rewards_per_window)
            ax.fill_between(x, lower, upper, color='#2ecc71', alpha=0.2)
        
        # Plot selected fixed betas
        selected_betas = [0.0, 0.5, 1.0]
        colors_fixed = {0.0: '#e74c3c', 0.5: '#f39c12', 1.0: '#3498db'}
        
        for beta_val in selected_betas:
            fixed = next((c for c in conds if c.beta_type == "fixed" and 
                         abs(c.fixed_beta - beta_val) < 0.01), None)
            if fixed is None:
                continue
            x_fixed = np.arange(1, len(fixed.mean_rewards_per_window) + 1)
            ax.plot(x_fixed, fixed.mean_rewards_per_window, 's--',
                   color=colors_fixed[beta_val], linewidth=1.5, 
                   markersize=4, alpha=0.7, label=f'Fixed β={beta_val:.1f}')
        
        # Mark environment changes
        num_changes = len(adaptive.mean_rewards_per_window) - 1
        for i in range(num_changes):
            ax.axvline(i + 1.5, color='red', linestyle=':', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Environment Window', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Volatility: {vol_interval} steps', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure2_learning_curves.png', 
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/figure2_learning_curves.png")
    plt.close()


def plot_effect_sizes(comparisons, output_dir):
    """Plot Cohen's d effect sizes"""
    
    # Group by volatility interval
    vol_intervals = sorted(list(set(int(c.condition_a.split('vol')[1]) 
                                   for c in comparisons)))
    
    fig, axes = plt.subplots(1, len(vol_intervals), figsize=(15, 5))
    if len(vol_intervals) == 1:
        axes = [axes]
    
    for idx, vol_interval in enumerate(vol_intervals):
        ax = axes[idx]
        
        # Get comparisons for this volatility
        comps = [c for c in comparisons if f'vol{vol_interval}' in c.condition_a]
        
        # Extract beta values and effect sizes
        beta_values = []
        cohens_d_values = []
        p_values = []
        
        for comp in comps:
            # Extract beta value from condition_b
            beta = float(comp.condition_b.split('β=')[1].split('_')[0])
            beta_values.append(beta)
            cohens_d_values.append(comp.cohens_d)
            p_values.append(comp.p_value)
        
        # Sort by beta value
        sorted_indices = np.argsort(beta_values)
        beta_values = [beta_values[i] for i in sorted_indices]
        cohens_d_values = [cohens_d_values[i] for i in sorted_indices]
        p_values = [p_values[i] for i in sorted_indices]
        
        # Color by significance
        colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in p_values]
        
        # Plot
        bars = ax.bar(range(len(beta_values)), cohens_d_values, 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add effect size interpretation lines
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.text(len(beta_values)-0.5, 0.2, 'Small', ha='right', va='bottom', 
               fontsize=8, style='italic', color='gray')
        ax.text(len(beta_values)-0.5, 0.5, 'Medium', ha='right', va='bottom',
               fontsize=8, style='italic', color='gray')
        ax.text(len(beta_values)-0.5, 0.8, 'Large', ha='right', va='bottom',
               fontsize=8, style='italic', color='gray')
        
        ax.set_xticks(range(len(beta_values)))
        ax.set_xticklabels([f'{b:.1f}' for b in beta_values], fontsize=10)
        ax.set_xlabel('Fixed β value', fontsize=11, fontweight='bold')
        ax.set_ylabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
        ax.set_title(f'Volatility: {vol_interval} steps\n(Adaptive β vs Fixed β)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add legend for colors
        sig_patch = mpatches.Patch(color='#2ecc71', label='p < 0.05 (significant)')
        ns_patch = mpatches.Patch(color='#95a5a6', label='p ≥ 0.05 (not significant)')
        ax.legend(handles=[sig_patch, ns_patch], loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure3_effect_sizes.png', 
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/figure3_effect_sizes.png")
    plt.close()


def plot_recovery_analysis(conditions, output_dir):
    """Analyze and plot recovery speeds after environment changes"""
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    fig, axes = plt.subplots(1, len(volatility_intervals), figsize=(15, 5))
    if len(volatility_intervals) == 1:
        axes = [axes]
    
    for idx, vol_interval in enumerate(volatility_intervals):
        ax = axes[idx]
        
        conds = [c for c in conditions if c.volatility_interval == vol_interval]
        
        # Prepare data for boxplot
        data_to_plot = []
        labels = []
        positions = []
        colors = []
        
        # Adaptive beta
        adaptive = next(c for c in conds if c.beta_type == "adaptive")
        all_speeds = []
        
        # 軽量モードの場合はall_recovery_speedsから取得
        if hasattr(adaptive, 'all_recovery_speeds') and adaptive.all_recovery_speeds:
            all_speeds = adaptive.all_recovery_speeds
        elif adaptive.agent_performances:
            for perf in adaptive.agent_performances:
                all_speeds.extend(perf.recovery_speeds)
        
        if all_speeds:
            data_to_plot.append(all_speeds)
            labels.append('Adaptive β')
            positions.append(0)
            colors.append('#2ecc71')
        
        # Fixed betas
        for i, beta_val in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            fixed = next((c for c in conds if c.beta_type == "fixed" and 
                         abs(c.fixed_beta - beta_val) < 0.01), None)
            if fixed is None:
                continue
            all_speeds = []
            
            # 軽量モードの場合はall_recovery_speedsから取得
            if hasattr(fixed, 'all_recovery_speeds') and fixed.all_recovery_speeds:
                all_speeds = fixed.all_recovery_speeds
            elif fixed.agent_performances:
                for perf in fixed.agent_performances:
                    all_speeds.extend(perf.recovery_speeds)
            
            if all_speeds:
                data_to_plot.append(all_speeds)
                labels.append(f'β={beta_val:.1f}')
                positions.append(i + 1)
                
                if beta_val < 0.01:
                    colors.append('#e74c3c')
                elif beta_val > 0.99:
                    colors.append('#3498db')
                else:
                    colors.append('#95a5a6')
        
        # データが空の場合はスキップ
        if not data_to_plot:
            ax.text(0.5, 0.5, 'No recovery data available\n(lightweight mode may not preserve this data)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Recovery After Env. Change\nVolatility: {vol_interval} steps',
                        fontsize=12, fontweight='bold')
            continue
        
        # Create violin plot
        parts = ax.violinplot(data_to_plot, positions=positions, 
                             showmeans=True, showmedians=True, widths=0.7)
        
        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Recovery Speed', fontsize=11, fontweight='bold')
        ax.set_title(f'Recovery After Env. Change\nVolatility: {vol_interval} steps',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # Add interpretation text
        ax.text(0.02, 0.98, 'Higher = Faster recovery\nto baseline performance',
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure4_recovery_analysis.png',
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/figure4_recovery_analysis.png")
    plt.close()


def plot_phase3_performance(conditions, comparisons, output_dir):
    """Plot performance for phase 3 (3rd environment window) only"""
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, vol_interval in enumerate(volatility_intervals):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        # Get conditions for this volatility interval
        conds = [c for c in conditions if c.volatility_interval == vol_interval]
        
        # Extract phase 3 (window index 2) rewards
        phase3_data = []
        labels = []
        colors = []
        
        for cond in conds:
            # 軽量モードではagent_performancesが空なので、mean_rewards_per_windowを使用
            if cond.agent_performances:
                # 詳細データがある場合
                phase3_rewards = []
                for perf in cond.agent_performances:
                    if len(perf.rewards_per_window) > 2:  # Ensure phase 3 exists
                        phase3_rewards.append(perf.rewards_per_window[2])
                
                if not phase3_rewards:
                    continue
                
                phase3_data.append({
                    'mean': np.mean(phase3_rewards),
                    'sem': stats.sem(phase3_rewards),
                    'cond': cond
                })
            elif len(cond.mean_rewards_per_window) > 2:
                # 軽量モードの場合、集計済みデータを使用
                # SEMは推定値として std/sqrt(num_agents) を使用
                phase3_data.append({
                    'mean': cond.mean_rewards_per_window[2],
                    'sem': cond.std_rewards_per_window[2] / np.sqrt(cond.num_agents) if len(cond.std_rewards_per_window) > 2 else 0,
                    'cond': cond
                })
        
        # Sort by mean performance
        phase3_data.sort(key=lambda x: x['mean'], reverse=True)
        
        # データが空の場合はスキップ
        if not phase3_data:
            ax.text(0.5, 0.5, 'No phase 3 data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Phase 3 Performance\nVolatility Interval: {vol_interval} steps', 
                        fontsize=12, fontweight='bold')
            continue
        
        # Prepare plot data
        means = [d['mean'] for d in phase3_data]
        sems = [d['sem'] for d in phase3_data]
        
        for d in phase3_data:
            cond = d['cond']
            if cond.beta_type == "adaptive":
                labels.append("Adaptive β")
                colors.append("#2ecc71")
            else:
                labels.append(f"β={cond.fixed_beta:.1f}")
                if cond.fixed_beta == 0.0:
                    colors.append("#e74c3c")
                elif cond.fixed_beta == 1.0:
                    colors.append("#3498db")
                else:
                    colors.append("#95a5a6")
        
        # Plot bars
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add significance stars (comparing to adaptive)
        adaptive_idx = next((i for i, d in enumerate(phase3_data) 
                            if d['cond'].beta_type == "adaptive"), None)
        
        if adaptive_idx is not None:
            adaptive_cond = phase3_data[adaptive_idx]['cond']
            for i, d in enumerate(phase3_data):
                if d['cond'].beta_type != "adaptive":
                    # Find comparison
                    comp = next((c for c in comparisons 
                               if c.condition_a == adaptive_cond.condition_name 
                               and c.condition_b == d['cond'].condition_name), None)
                    
                    if comp:
                        y_pos = max(means[adaptive_idx], means[i]) + max(sems[adaptive_idx], sems[i]) + 0.5
                        
                        if comp.p_value < 0.001:
                            sig_text = "***"
                        elif comp.p_value < 0.01:
                            sig_text = "**"
                        elif comp.p_value < 0.05:
                            sig_text = "*"
                        else:
                            sig_text = "n.s."
                        
                        if sig_text != "n.s.":
                            ax.plot([adaptive_idx, i], [y_pos, y_pos], 'k-', linewidth=1)
                            ax.text((adaptive_idx + i) / 2, y_pos + 0.1, sig_text,
                                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Highlight best performer
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(3)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Phase 3 Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Phase 3 Performance\nVolatility Interval: {vol_interval} steps', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add sample size
        if phase3_data:
            n = len([p for p in phase3_data[0]['cond'].agent_performances 
                    if len(p.rewards_per_window) > 2])
            ax.text(0.02, 0.98, f'n={n} agents\n(3rd env. window)', 
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend for significance
    fig.text(0.5, 0.02, '*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant\n'
                        'Error bars: SEM (Standard Error of Mean)\n'
                        'Phase 3: Rewards in the 3rd environment change window',
             ha='center', fontsize=9, style='italic')
    
    plt.savefig(f'{output_dir}/figure5_phase3_performance.png', 
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/figure5_phase3_performance.png")
    plt.close()


def plot_phase3_total_reward(conditions, comparisons, output_dir):
    """Plot total reward for phase 3 (3rd environment window) only"""
    
    volatility_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, vol_interval in enumerate(volatility_intervals):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        # Get conditions for this volatility interval
        conds = [c for c in conditions if c.volatility_interval == vol_interval]
        
        # Extract phase 3 total rewards
        phase3_data = []
        
        for cond in conds:
            # 軽量モードではpost_change_rewardsが空なので、mean_rewards_per_windowから推定
            if cond.agent_performances and any(len(p.post_change_rewards) > 2 for p in cond.agent_performances):
                # Collect phase 3 total rewards from all agents
                phase3_total_rewards = []
                for perf in cond.agent_performances:
                    # Use post_change_rewards[2] if available (3rd environment change)
                    if len(perf.post_change_rewards) > 2 and len(perf.post_change_rewards[2]) > 0:
                        total_reward = sum(perf.post_change_rewards[2])
                        phase3_total_rewards.append(total_reward)
                
                if not phase3_total_rewards:
                    continue
                
                phase3_data.append({
                    'mean': np.mean(phase3_total_rewards),
                    'sem': stats.sem(phase3_total_rewards),
                    'cond': cond
                })
            elif len(cond.mean_rewards_per_window) > 2:
                # 軽量モードの場合、平均報酬から推定
                # Phase 3の合計報酬 = 平均報酬 × ステップ数（推定）
                # volatility_intervalを使用して推定
                estimated_steps_in_phase3 = cond.volatility_interval
                estimated_total = cond.mean_rewards_per_window[2] * estimated_steps_in_phase3
                estimated_sem = (cond.std_rewards_per_window[2] / np.sqrt(cond.num_agents)) * estimated_steps_in_phase3 if len(cond.std_rewards_per_window) > 2 else 0
                
                phase3_data.append({
                    'mean': estimated_total,
                    'sem': estimated_sem,
                    'cond': cond
                })
        
        # Sort by mean performance
        phase3_data.sort(key=lambda x: x['mean'], reverse=True)
        
        # データが空の場合はスキップ
        if not phase3_data:
            ax.text(0.5, 0.5, 'No phase 3 data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Phase 3 Total Reward\nVolatility Interval: {vol_interval} steps', 
                        fontsize=12, fontweight='bold')
            continue
        
        # Prepare plot data
        means = [d['mean'] for d in phase3_data]
        sems = [d['sem'] for d in phase3_data]
        labels = []
        colors = []
        
        for d in phase3_data:
            cond = d['cond']
            if cond.beta_type == "adaptive":
                labels.append("Adaptive β")
                colors.append("#2ecc71")
            else:
                labels.append(f"β={cond.fixed_beta:.1f}")
                if cond.fixed_beta == 0.0:
                    colors.append("#e74c3c")
                elif cond.fixed_beta == 1.0:
                    colors.append("#3498db")
                else:
                    colors.append("#95a5a6")
        
        # Plot bars
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=5, 
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add significance stars (comparing to adaptive)
        adaptive_idx = next((i for i, d in enumerate(phase3_data) 
                            if d['cond'].beta_type == "adaptive"), None)
        
        if adaptive_idx is not None:
            adaptive_cond = phase3_data[adaptive_idx]['cond']
            for i, d in enumerate(phase3_data):
                if d['cond'].beta_type != "adaptive":
                    # Find comparison
                    comp = next((c for c in comparisons 
                               if c.condition_a == adaptive_cond.condition_name 
                               and c.condition_b == d['cond'].condition_name), None)
                    
                    if comp:
                        y_pos = max(means[adaptive_idx], means[i]) + max(sems[adaptive_idx], sems[i]) + 10
                        
                        if comp.p_value < 0.001:
                            sig_text = "***"
                        elif comp.p_value < 0.01:
                            sig_text = "**"
                        elif comp.p_value < 0.05:
                            sig_text = "*"
                        else:
                            sig_text = "n.s."
                        
                        if sig_text != "n.s.":
                            ax.plot([adaptive_idx, i], [y_pos, y_pos], 'k-', linewidth=1)
                            ax.text((adaptive_idx + i) / 2, y_pos + 2, sig_text,
                                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Highlight best performer
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(3)
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Phase 3 Total Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'Phase 3 Total Reward\nVolatility Interval: {vol_interval} steps', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Add sample size
        if phase3_data:
            n = len([p for p in phase3_data[0]['cond'].agent_performances 
                    if len(p.post_change_rewards) > 2 and len(p.post_change_rewards[2]) > 0])
            ax.text(0.02, 0.98, f'n={n} agents\n(3rd env. window)', 
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend for significance
    fig.text(0.5, 0.02, '*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant\n'
                        'Error bars: SEM (Standard Error of Mean)\n'
                        'Phase 3: Total cumulative reward in the 3rd environment change window',
             ha='center', fontsize=9, style='italic')
    
    plt.savefig(f'{output_dir}/figure6_phase3_total_reward.png', 
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_dir}/figure6_phase3_total_reward.png")
    plt.close()


def create_summary_table(conditions, comparisons, output_dir):
    """Create publication-ready summary table"""
    
    vol_intervals = sorted(list(set(c.volatility_interval for c in conditions)))
    
    with open(f'{output_dir}/table1_summary.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison of adaptive and fixed β strategies}\n")
        f.write("\\label{tab:performance}\n")
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\hline\n")
        f.write("Volatility & Strategy & Mean Reward & SEM & 95\\% CI & Recovery & vs Adaptive \\\\\n")
        f.write("\\hline\n")
        
        for vol_interval in vol_intervals:
            conds = [c for c in conditions if c.volatility_interval == vol_interval]
            conds_sorted = sorted(conds, key=lambda x: x.mean_total_reward, reverse=True)
            
            for i, cond in enumerate(conds_sorted):
                if i == 0:
                    vol_str = f"{vol_interval}"
                else:
                    vol_str = ""
                
                if cond.beta_type == "adaptive":
                    strategy = "Adaptive β"
                    vs_adaptive = "---"
                else:
                    strategy = f"Fixed β={cond.fixed_beta:.1f}"
                    
                    # Find comparison
                    adaptive = next(c for c in conds if c.beta_type == "adaptive")
                    comp = next((c for c in comparisons 
                               if c.condition_a == adaptive.condition_name 
                               and c.condition_b == cond.condition_name), None)
                    
                    if comp:
                        if comp.p_value < 0.001:
                            vs_adaptive = f"p<0.001***"
                        elif comp.p_value < 0.01:
                            vs_adaptive = f"p={comp.p_value:.3f}**"
                        elif comp.p_value < 0.05:
                            vs_adaptive = f"p={comp.p_value:.3f}*"
                        else:
                            vs_adaptive = f"p={comp.p_value:.3f}"
                    else:
                        vs_adaptive = "---"
                
                ci_str = f"[{cond.ci_95_lower:.1f}, {cond.ci_95_upper:.1f}]"
                
                f.write(f"{vol_str} & {strategy} & {cond.mean_total_reward:.2f} & "
                       f"{cond.sem_total_reward:.2f} & {ci_str} & "
                       f"{cond.mean_recovery_speed:.3f} & {vs_adaptive} \\\\\n")
            
            if vol_interval != vol_intervals[-1]:
                f.write("\\hline\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved: {output_dir}/table1_summary.tex")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Visualize large-scale experiment results')
    parser.add_argument('--input-dir', type=str, default='large_scale_results',
                       help='Input directory with experiment results')
    parser.add_argument('--output-dir', type=str, default='large_scale_results',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("PUBLICATION-QUALITY VISUALIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    with open(f'{args.input_dir}/full_experiment_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    conditions = data['conditions']
    comparisons = data['comparisons']
    params = data['parameters']
    
    print(f"Loaded {len(conditions)} conditions, {len(comparisons)} comparisons")
    
    # Generate all figures
    print("\nGenerating figures...")
    num_steps = params.get('num_steps', 8000)
    plot_main_results(conditions, comparisons, args.output_dir, num_steps)
    plot_learning_curves(conditions, args.output_dir)
    plot_effect_sizes(comparisons, args.output_dir)
    plot_recovery_analysis(conditions, args.output_dir)
    plot_phase3_performance(conditions, comparisons, args.output_dir)
    plot_phase3_total_reward(conditions, comparisons, args.output_dir)
    
    print("\nCreating summary table...")
    create_summary_table(conditions, comparisons, args.output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files in {args.output_dir}:")
    print("  - figure1_main_performance.png")
    print("  - figure2_learning_curves.png")
    print("  - figure3_effect_sizes.png")
    print("  - figure4_recovery_analysis.png")
    print("  - figure5_phase3_performance.png")
    print("  - figure6_phase3_total_reward.png")
    print("  - table1_summary.tex")


if __name__ == "__main__":
    main()
