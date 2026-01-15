"""
揮発性環境実験結果の可視化（英語ラベル版）
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# フォント設定（macOS対応）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_volatility_results_english(results_file: str, output_prefix: str = "volatility_english"):
    """英語ラベルで揮発性環境実験結果を可視化"""
    
    # 結果を読み込み
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    colors = {
        "fixed_0.0": "#e74c3c",  # 赤（MF重視）
        "fixed_0.5": "#f39c12",  # オレンジ
        "fixed_1.0": "#3498db",  # 青（MB重視）
        "adaptive": "#2ecc71",   # 緑（適応的）
    }
    
    labels = {
        "fixed_0.0": "Fixed β=0.0 (MF)",
        "fixed_0.5": "Fixed β=0.5",
        "fixed_1.0": "Fixed β=1.0 (MB)",
        "adaptive": "Adaptive β",
    }
    
    # ===== メインの比較グラフ =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) 累積報酬の比較
    ax1 = axes[0, 0]
    beta_types = list(results.keys())
    total_rewards = [results[bt]['total_reward'] for bt in beta_types]
    bars = ax1.bar(range(len(beta_types)), total_rewards, 
                   color=[colors[bt] for bt in beta_types], edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(beta_types)))
    ax1.set_xticklabels([labels[bt] for bt in beta_types], rotation=15, ha='right', fontsize=10)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Total Reward Comparison in Volatile Environment', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 値を表示
    for bar, val in zip(bars, total_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 30, 
                f'{val:.1f}', ha='center', va='top', fontweight='bold', fontsize=10, color='white')
    
    # 最高性能にマーク
    best_idx = np.argmax(total_rewards)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    ax1.annotate('BEST', xy=(best_idx, total_rewards[best_idx]),
                xytext=(best_idx, total_rewards[best_idx] + 50),
                ha='center', fontsize=12, fontweight='bold', color='gold',
                arrowprops=dict(arrowstyle='->', color='gold'))
    
    # (2) 環境変化区間ごとの報酬推移
    ax2 = axes[0, 1]
    for beta_type in results:
        rewards_per_window = results[beta_type]['rewards_per_window']
        if rewards_per_window:
            ax2.plot(range(1, len(rewards_per_window)+1), rewards_per_window, 
                    'o-', label=labels[beta_type], color=colors[beta_type], linewidth=2, markersize=4)
    ax2.set_xlabel('Environment Change Window', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Average Reward per Environment Window', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # (3) 適応速度の比較（箱ひげ図）
    ax3 = axes[1, 0]
    adaptation_data = []
    adaptation_labels = []
    for beta_type in results:
        speeds = results[beta_type]['adaptation_speed']
        if speeds:
            # 外れ値を除去（-10〜10の範囲に制限）
            speeds_clipped = [s for s in speeds if -10 < s < 10]
            if speeds_clipped:
                adaptation_data.append(speeds_clipped)
                adaptation_labels.append(labels[beta_type])
    
    if adaptation_data:
        bp = ax3.boxplot(adaptation_data, labels=adaptation_labels, patch_artist=True, widths=0.6)
        for patch, beta_type in zip(bp['boxes'], results.keys()):
            patch.set_facecolor(colors[beta_type])
            patch.set_alpha(0.7)
        ax3.set_ylabel('Adaptation Speed', fontsize=12)
        ax3.set_title('Adaptation Speed After Environment Change', fontsize=14, fontweight='bold')
        ax3.set_xticklabels(adaptation_labels, rotation=15, ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # (4) 平均報酬/ステップの比較
    ax4 = axes[1, 1]
    avg_rewards = [results[bt]['avg_reward_per_step'] for bt in beta_types]
    bars4 = ax4.bar(range(len(beta_types)), avg_rewards, 
                    color=[colors[bt] for bt in beta_types], edgecolor='black', linewidth=1)
    ax4.set_xticks(range(len(beta_types)))
    ax4.set_xticklabels([labels[bt] for bt in beta_types], rotation=15, ha='right', fontsize=10)
    ax4.set_ylabel('Average Reward per Step', fontsize=12)
    ax4.set_title('Average Reward per Step', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 値を表示
    for bar, val in zip(bars4, avg_rewards):
        y_pos = bar.get_height() - 0.02 if val < 0 else bar.get_height() + 0.01
        ax4.text(bar.get_x() + bar.get_width()/2, y_pos, 
                f'{val:.4f}', ha='center', va='top' if val < 0 else 'bottom', 
                fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_comparison.png")
    plt.close()
    
    # ===== サマリーグラフ =====
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # (1) 正規化性能比較
    ax2_1 = axes2[0]
    
    # 報酬を正規化（最も悪い結果を0、最も良い結果を1とする）
    min_reward = min(total_rewards)
    max_reward = max(total_rewards)
    reward_range = max_reward - min_reward if max_reward != min_reward else 1
    normalized_rewards = [(r - min_reward) / reward_range for r in total_rewards]
    
    bars_norm = ax2_1.barh(range(len(beta_types)), normalized_rewards,
                           color=[colors[bt] for bt in beta_types], edgecolor='black', linewidth=1)
    ax2_1.set_yticks(range(len(beta_types)))
    ax2_1.set_yticklabels([labels[bt] for bt in beta_types], fontsize=11)
    ax2_1.set_xlabel('Normalized Performance Score', fontsize=12)
    ax2_1.set_title('Performance Ranking in Volatile Environment', fontsize=14, fontweight='bold')
    ax2_1.set_xlim(0, 1.15)
    ax2_1.grid(True, alpha=0.3, axis='x')
    
    # 値とランキングを表示
    sorted_indices = np.argsort(normalized_rewards)[::-1]
    for i, (bar, val) in enumerate(zip(bars_norm, normalized_rewards)):
        rank = sorted_indices.tolist().index(i) + 1
        ax2_1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                  f'#{rank} ({val:.2f})', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # (2) 結果サマリー
    ax2_2 = axes2[1]
    ax2_2.axis('off')
    
    # 結果のサマリーを作成
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_reward'], reverse=True)
    
    summary_lines = [
        "EXPERIMENT RESULTS SUMMARY",
        "=" * 40,
        "",
        "Volatile Environment (changes every 100 steps)",
        "",
        "PERFORMANCE RANKING:",
        ""
    ]
    
    for i, (beta_type, result) in enumerate(sorted_results):
        rank = i + 1
        avg_speed = np.mean([s for s in result['adaptation_speed'] if -10 < s < 10])
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
        summary_lines.append(f"{medal} #{rank}: {labels[beta_type]}")
        summary_lines.append(f"     Total Reward: {result['total_reward']:.2f}")
        summary_lines.append(f"     Avg Reward/Step: {result['avg_reward_per_step']:.4f}")
        summary_lines.append(f"     Avg Adaptation Speed: {avg_speed:.3f}")
        summary_lines.append("")
    
    # 結論
    adaptive_rank = next(i for i, (bt, _) in enumerate(sorted_results) if bt == "adaptive") + 1
    
    summary_lines.append("=" * 40)
    summary_lines.append("CONCLUSION:")
    if adaptive_rank == 1:
        summary_lines.append("✓ Adaptive β achieved BEST performance!")
        summary_lines.append("  TD-error-based adaptation is effective")
        summary_lines.append("  in volatile environments.")
    else:
        winner = sorted_results[0][0]
        summary_lines.append(f"  Adaptive β ranked #{adaptive_rank}")
        summary_lines.append(f"  {labels[winner]} performed best.")
        summary_lines.append("  Further tuning may improve adaptive β.")
    
    summary_text = '\n'.join(summary_lines)
    ax2_2.text(0.05, 0.95, summary_text, transform=ax2_2.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange'))
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_summary.png")
    plt.close()
    
    # ===== 詳細分析グラフ =====
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    # (1) 環境変化後の報酬変動
    ax3_1 = axes3[0]
    for beta_type in results:
        rewards = results[beta_type]['rewards_per_window']
        # 変動率を計算
        changes = [rewards[i] - rewards[i-1] for i in range(1, len(rewards))]
        ax3_1.plot(range(2, len(rewards)+1), changes, 'o-', 
                  label=labels[beta_type], color=colors[beta_type], linewidth=1.5, markersize=3, alpha=0.7)
    ax3_1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3_1.set_xlabel('Environment Window', fontsize=12)
    ax3_1.set_ylabel('Reward Change', fontsize=12)
    ax3_1.set_title('Reward Change Between Windows', fontsize=14, fontweight='bold')
    ax3_1.legend(loc='best', fontsize=9)
    ax3_1.grid(True, alpha=0.3)
    
    # (2) 累積報酬の推移シミュレーション
    ax3_2 = axes3[1]
    for beta_type in results:
        rewards = results[beta_type]['rewards_per_window']
        cumulative = np.cumsum([r * 100 for r in rewards])  # 各区間100ステップと仮定
        ax3_2.plot(range(1, len(cumulative)+1), cumulative, 
                  'o-', label=labels[beta_type], color=colors[beta_type], linewidth=2, markersize=4)
    ax3_2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3_2.set_xlabel('Environment Window', fontsize=12)
    ax3_2.set_ylabel('Cumulative Reward', fontsize=12)
    ax3_2.set_title('Cumulative Reward Trajectory', fontsize=14, fontweight='bold')
    ax3_2.legend(loc='best', fontsize=9)
    ax3_2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_detailed.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_detailed.png")
    plt.close()
    
    print("\n" + "="*60)
    print("VOLATILITY EXPERIMENT - FINAL SUMMARY")
    print("="*60)
    for i, (beta_type, result) in enumerate(sorted_results):
        print(f"#{i+1}: {labels[beta_type]} | Total Reward: {result['total_reward']:.2f}")
    print("="*60)
    
    if adaptive_rank == 1:
        print("\n✓ SUCCESS: Adaptive β outperforms fixed β in volatile environment!")
        print("  The TD-error-based adaptation allows the agent to detect")
        print("  environment changes and quickly re-adapt by increasing MB reliance.")
    else:
        print(f"\n※ Adaptive β ranked #{adaptive_rank}.")
        print("  Consider tuning TD-error threshold or adaptation parameters.")


if __name__ == "__main__":
    plot_volatility_results_english("volatility_experiment_results.json", "volatility_result")
