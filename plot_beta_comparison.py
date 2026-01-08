#!/usr/bin/env python3
"""
β値比較実験の結果を可視化するスクリプト

各β値（固定モード）とβ学習モードの依存症率を比較し、
複数のグラフを生成します。
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_results(results_dir: Path):
    """結果ファイルを読み込む"""
    results = {}
    
    # 固定β値の結果を読み込む
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for beta in beta_values:
        # ファイル名のフォーマット: beta値が整数なら".0"を除く
        if beta == int(beta):
            file_path = results_dir / f"beta_{int(beta)}.json"
        else:
            file_path = results_dir / f"beta_{beta}.json"
            
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                results[f"β={beta}"] = {
                    'beta': beta,
                    'mode': 'fixed',
                    'mean_rate': data['addiction_rate'],
                    'run_rates': data['run_rates'],
                    'std': np.std(data['run_rates']),
                    'sem': np.std(data['run_rates']) / np.sqrt(len(data['run_rates'])),
                    # Reversal関連
                    'reversal_rate': data.get('reversal_adaptation_rate', 0.0),
                    'run_reversal_rates': data.get('run_reversal_rates', []),
                    'reversal_std': np.std(data.get('run_reversal_rates', [0])),
                    'reversal_sem': np.std(data.get('run_reversal_rates', [0])) / np.sqrt(max(1, len(data.get('run_reversal_rates', [0])))),
                    # 状態訪問統計
                    'addiction_state_ratios': data.get('addiction_state_ratios', [0.0] * 22),
                    'reversal_state_ratios': data.get('reversal_state_ratios', [0.0] * 22),
                }
        else:
            print(f"警告: {file_path} が見つかりません")
    
    # β学習モードの結果を読み込む
    learning_path = results_dir / "beta_learning.json"
    if learning_path.exists():
        with open(learning_path, 'r') as f:
            data = json.load(f)
            results["β=Learning"] = {
                'beta': None,
                'mode': 'learning',
                'mean_rate': data['addiction_rate'],
                'run_rates': data['run_rates'],
                'std': np.std(data['run_rates']),
                'sem': np.std(data['run_rates']) / np.sqrt(len(data['run_rates'])),
                # Reversal関連
                'reversal_rate': data.get('reversal_adaptation_rate', 0.0),
                'run_reversal_rates': data.get('run_reversal_rates', []),
                'reversal_std': np.std(data.get('run_reversal_rates', [0])),
                'reversal_sem': np.std(data.get('run_reversal_rates', [0])) / np.sqrt(max(1, len(data.get('run_reversal_rates', [0])))),
                # 状態訪問統計
                'addiction_state_ratios': data.get('addiction_state_ratios', [0.0] * 22),
                'reversal_state_ratios': data.get('reversal_state_ratios', [0.0] * 22),
            }
    else:
        print(f"警告: {learning_path} が見つかりません")
    
    return results


def plot_comparison(results, output_prefix="beta_comparison"):
    """比較グラフを作成"""
    
    # データの準備
    beta_labels = []
    mean_rates = []
    std_errors = []
    colors = []
    
    # 固定β値の結果
    fixed_betas = [k for k in results.keys() if k.startswith("β=") and k != "β=Learning"]
    fixed_betas.sort(key=lambda x: float(x.split("=")[1]))
    
    for label in fixed_betas:
        beta_labels.append(label)
        mean_rates.append(results[label]['mean_rate'])
        std_errors.append(results[label]['sem'])
        colors.append('#3498db')  # 青
    
    # β学習モードの結果
    if "β=Learning" in results:
        beta_labels.append("Learning\n(Adaptive)")
        mean_rates.append(results["β=Learning"]['mean_rate'])
        std_errors.append(results["β=Learning"]['sem'])
        colors.append('#e74c3c')  # 赤
    
    # ===== Figure 1: 棒グラフ（平均±標準誤差） =====
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    x_pos = np.arange(len(beta_labels))
    bars = ax1.bar(x_pos, mean_rates, yerr=std_errors, 
                   color=colors, alpha=0.7, capsize=10, 
                   edgecolor='black', linewidth=1.5)
    
    # 値をバーの上に表示
    for i, (bar, mean, sem) in enumerate(zip(bars, mean_rates, std_errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + sem + 1,
                f'{mean:.2f}%\n±{sem:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('β値 / モード', fontsize=14, fontweight='bold')
    ax1.set_ylabel('依存症率 (%)', fontsize=14, fontweight='bold')
    ax1.set_title('β値による依存症率の比較', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(beta_labels, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(mean_rates) + max(std_errors) + 10)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', label='固定β値'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='β学習モード')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bar.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_bar.png")
    plt.close()
    
    # ===== Figure 2: 折れ線グラフ（固定β値のみ） =====
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    fixed_beta_values = []
    fixed_means = []
    fixed_sems = []
    
    for label in fixed_betas:
        beta_val = results[label]['beta']
        fixed_beta_values.append(beta_val)
        fixed_means.append(results[label]['mean_rate'])
        fixed_sems.append(results[label]['sem'])
    
    ax2.errorbar(fixed_beta_values, fixed_means, yerr=fixed_sems,
                marker='o', markersize=10, linewidth=2.5, capsize=8,
                color='#2c3e50', ecolor='#95a5a6', markerfacecolor='#3498db',
                markeredgecolor='black', markeredgewidth=1.5)
    
    # β学習モードの結果を水平線で表示
    if "β=Learning" in results:
        learning_mean = results["β=Learning"]['mean_rate']
        learning_sem = results["β=Learning"]['sem']
        ax2.axhline(learning_mean, color='#e74c3c', linestyle='--', 
                   linewidth=2, label=f'Learning Mode: {learning_mean:.2f}% ±{learning_sem:.2f}')
        ax2.fill_between([0, 1], learning_mean - learning_sem, learning_mean + learning_sem,
                        alpha=0.2, color='#e74c3c')
    
    ax2.set_xlabel('β値 (0=Model-Free, 1=Model-Based)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('依存症率 (%)', fontsize=14, fontweight='bold')
    ax2.set_title('β値と依存症率の関係', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_line.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_line.png")
    plt.close()
    
    # ===== Figure 3: 箱ひげ図（各runの分布） =====
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    
    box_data = []
    box_labels = []
    box_colors = []
    
    for label in fixed_betas:
        box_data.append(results[label]['run_rates'])  # 既にパーセント値
        box_labels.append(label)
        box_colors.append('#3498db')
    
    if "β=Learning" in results:
        box_data.append(results["β=Learning"]['run_rates'])  # 既にパーセント値
        box_labels.append("Learning")
        box_colors.append('#e74c3c')
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True,
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # 箱の色を設定
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax3.set_xlabel('β値 / モード', fontsize=14, fontweight='bold')
    ax3.set_ylabel('依存症率 (%) - 各run', fontsize=14, fontweight='bold')
    ax3.set_title('β値による依存症率の分布（全run）', fontsize=16, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='中央値'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='平均値')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_boxplot.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_boxplot.png")
    plt.close()
    
    # ===== Figure 4: 統計サマリー表 =====
    fig4, ax4 = plt.subplots(figsize=(14, 8))
    ax4.axis('tight')
    ax4.axis('off')
    
    # テーブルデータの作成
    table_data = []
    headers = ['β値/モード', '平均 (%)', '標準偏差 (%)', '標準誤差 (%)', '最小 (%)', '最大 (%)']
    
    for label in fixed_betas + (["β=Learning"] if "β=Learning" in results else []):
        display_label = label.replace("β=", "")
        data = results[label]
        run_rates_pct = data['run_rates']  # 既にパーセント値
        
        row = [
            display_label,
            f"{data['mean_rate']:.2f}",
            f"{data['std']:.2f}",
            f"{data['sem']:.2f}",
            f"{min(run_rates_pct):.2f}",
            f"{max(run_rates_pct):.2f}"
        ]
        table_data.append(row)
    
    # テーブルの色設定
    cell_colors = []
    for i, label in enumerate(fixed_betas + (["β=Learning"] if "β=Learning" in results else [])):
        if label == "β=Learning":
            cell_colors.append(['#ffcccc'] * len(headers))
        else:
            cell_colors.append(['#cce5ff'] * len(headers))
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     cellColours=cell_colors,
                     colColours=['#4a7ba7'] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # ヘッダーのスタイル
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4a7ba7')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('β値比較実験 - 統計サマリー', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_table.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_table.png")
    plt.close()
    
    # ===== 統計情報を出力 =====
    print("\n" + "=" * 60)
    print("β値比較実験 - 統計サマリー")
    print("=" * 60)
    
    for label in fixed_betas + (["β=Learning"] if "β=Learning" in results else []):
        data = results[label]
        print(f"\n{label}:")
        print(f"  平均依存症率: {data['mean_rate']:.2f}% ± {data['sem']:.2f}%")
        print(f"  標準偏差: {data['std']:.2f}%")
        run_rates_pct = data['run_rates']  # 既にパーセント値
        print(f"  範囲: {min(run_rates_pct):.2f}% - {max(run_rates_pct):.2f}%")
        if 'reversal_rate' in data:
            print(f"  Reversal適応率: {data['reversal_rate']:.2f}% ± {data['reversal_sem']:.2f}%")
    
    print("\n" + "=" * 60)


def plot_reversal_analysis(results, output_prefix="reversal_analysis"):
    """Reversalフェーズの適応度分析グラフを作成"""
    
    # 固定β値とLearningモードの結果を分離
    fixed_betas = [k for k in results.keys() if k.startswith("β=") and k != "β=Learning"]
    fixed_betas.sort(key=lambda x: float(x.split("=")[1]))
    
    # ===== Figure 1: Reversal適応率の比較（棒グラフ） =====
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    labels = []
    reversal_rates = []
    reversal_sems = []
    colors = []
    
    for label in fixed_betas:
        labels.append(label)
        reversal_rates.append(results[label]['reversal_rate'])
        reversal_sems.append(results[label]['reversal_sem'])
        colors.append('#3498db')
    
    if "β=Learning" in results:
        labels.append("Learning\n(Adaptive)")
        reversal_rates.append(results["β=Learning"]['reversal_rate'])
        reversal_sems.append(results["β=Learning"]['reversal_sem'])
        colors.append('#e74c3c')
    
    x_pos = np.arange(len(labels))
    bars = ax1.bar(x_pos, reversal_rates, yerr=reversal_sems,
                   color=colors, alpha=0.7, capsize=10,
                   edgecolor='black', linewidth=1.5)
    
    for bar, rate, sem in zip(bars, reversal_rates, reversal_sems):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + sem + 1,
                f'{rate:.2f}%\n±{sem:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('β値 / モード', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Reversal適応率 (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Reversalフェーズ適応率の比較\n(環境変化への適応度)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(reversal_rates) + max(reversal_sems) + 15 if reversal_rates else 100)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', label='固定β値'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='β学習モード')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_reversal_bar.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_reversal_bar.png")
    plt.close()
    
    # ===== Figure 2: Addiction Rate vs Reversal Adaptation Rate =====
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    addiction_rates = []
    reversal_rates_list = []
    marker_labels = []
    marker_colors = []
    
    for label in fixed_betas:
        addiction_rates.append(results[label]['mean_rate'])
        reversal_rates_list.append(results[label]['reversal_rate'])
        marker_labels.append(label)
        marker_colors.append('#3498db')
    
    if "β=Learning" in results:
        addiction_rates.append(results["β=Learning"]['mean_rate'])
        reversal_rates_list.append(results["β=Learning"]['reversal_rate'])
        marker_labels.append("Learning")
        marker_colors.append('#e74c3c')
    
    # 散布図
    for i, (add_rate, rev_rate, label, color) in enumerate(zip(addiction_rates, reversal_rates_list, marker_labels, marker_colors)):
        marker = 's' if label == "Learning" else 'o'
        size = 200 if label == "Learning" else 150
        ax2.scatter(add_rate, rev_rate, c=color, s=size, marker=marker,
                   edgecolor='black', linewidth=1.5, alpha=0.8, zorder=3)
        ax2.annotate(label, (add_rate, rev_rate), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Addictionフェーズ 依存症率 (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Reversalフェーズ 適応率 (%)', fontsize=14, fontweight='bold')
    ax2.set_title('依存症率 vs 環境変化適応率', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    # 対角線（参考）
    max_val = max(max(addiction_rates), max(reversal_rates_list)) if addiction_rates else 100
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x (参考線)')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_addiction_vs_reversal.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_addiction_vs_reversal.png")
    plt.close()
    
    # ===== Figure 3: 状態0（Goal状態）訪問割合の比較 =====
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
    
    # Addictionフェーズでの状態0訪問率
    ax3_add = axes3[0]
    state0_addiction = []
    state0_labels = []
    state0_colors = []
    
    for label in fixed_betas:
        state0_addiction.append(results[label]['addiction_state_ratios'][0])
        state0_labels.append(label)
        state0_colors.append('#3498db')
    
    if "β=Learning" in results:
        state0_addiction.append(results["β=Learning"]['addiction_state_ratios'][0])
        state0_labels.append("Learning")
        state0_colors.append('#e74c3c')
    
    x_pos = np.arange(len(state0_labels))
    bars_add = ax3_add.bar(x_pos, state0_addiction, color=state0_colors, alpha=0.7,
                            edgecolor='black', linewidth=1.5)
    
    for bar, rate in zip(bars_add, state0_addiction):
        ax3_add.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3_add.set_xlabel('β値 / モード', fontsize=12, fontweight='bold')
    ax3_add.set_ylabel('状態0（Goal）訪問割合 (%)', fontsize=12, fontweight='bold')
    ax3_add.set_title('Addictionフェーズ\n状態0（Goal）訪問割合', fontsize=14, fontweight='bold')
    ax3_add.set_xticks(x_pos)
    ax3_add.set_xticklabels(state0_labels, fontsize=10)
    ax3_add.grid(True, alpha=0.3, axis='y')
    
    # Reversalフェーズでの状態0訪問率
    ax3_rev = axes3[1]
    state0_reversal = []
    
    for label in fixed_betas:
        state0_reversal.append(results[label]['reversal_state_ratios'][0])
    
    if "β=Learning" in results:
        state0_reversal.append(results["β=Learning"]['reversal_state_ratios'][0])
    
    bars_rev = ax3_rev.bar(x_pos, state0_reversal, color=state0_colors, alpha=0.7,
                            edgecolor='black', linewidth=1.5)
    
    for bar, rate in zip(bars_rev, state0_reversal):
        ax3_rev.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3_rev.set_xlabel('β値 / モード', fontsize=12, fontweight='bold')
    ax3_rev.set_ylabel('状態0（Goal）訪問割合 (%)', fontsize=12, fontweight='bold')
    ax3_rev.set_title('Reversalフェーズ\n状態0（Goal）訪問割合\n※Reversalでは高いほど依存的', fontsize=14, fontweight='bold')
    ax3_rev.set_xticks(x_pos)
    ax3_rev.set_xticklabels(state0_labels, fontsize=10)
    ax3_rev.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_state0_comparison.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_state0_comparison.png")
    plt.close()
    
    # ===== Figure 4: 状態訪問分布のヒートマップ比較 =====
    fig4, axes4 = plt.subplots(2, 1, figsize=(18, 12))
    
    # データ準備
    all_labels = fixed_betas + (["β=Learning"] if "β=Learning" in results else [])
    num_states = 22
    
    # Addictionフェーズ
    addiction_matrix = np.zeros((len(all_labels), num_states))
    for i, label in enumerate(all_labels):
        addiction_matrix[i, :] = results[label]['addiction_state_ratios']
    
    ax4_add = axes4[0]
    im_add = ax4_add.imshow(addiction_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    cbar_add = plt.colorbar(im_add, ax=ax4_add)
    cbar_add.set_label('訪問割合 (%)', fontsize=12)
    ax4_add.set_xlabel('状態番号', fontsize=12, fontweight='bold')
    ax4_add.set_ylabel('β値 / モード', fontsize=12, fontweight='bold')
    ax4_add.set_title('Addictionフェーズ - 状態訪問分布', fontsize=14, fontweight='bold')
    ax4_add.set_xticks(range(num_states))
    ax4_add.set_yticks(range(len(all_labels)))
    ax4_add.set_yticklabels([l.replace("β=", "") for l in all_labels])
    
    # 重要な状態をハイライト
    for s in [0, 3, 6, 7]:  # Goal, Start, Neutral最終, Drug
        ax4_add.axvline(s - 0.5, color='blue', linewidth=2, alpha=0.5)
        ax4_add.axvline(s + 0.5, color='blue', linewidth=2, alpha=0.5)
    
    # Reversalフェーズ
    reversal_matrix = np.zeros((len(all_labels), num_states))
    for i, label in enumerate(all_labels):
        reversal_matrix[i, :] = results[label]['reversal_state_ratios']
    
    ax4_rev = axes4[1]
    im_rev = ax4_rev.imshow(reversal_matrix, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    cbar_rev = plt.colorbar(im_rev, ax=ax4_rev)
    cbar_rev.set_label('訪問割合 (%)', fontsize=12)
    ax4_rev.set_xlabel('状態番号', fontsize=12, fontweight='bold')
    ax4_rev.set_ylabel('β値 / モード', fontsize=12, fontweight='bold')
    ax4_rev.set_title('Reversalフェーズ - 状態訪問分布', fontsize=14, fontweight='bold')
    ax4_rev.set_xticks(range(num_states))
    ax4_rev.set_yticks(range(len(all_labels)))
    ax4_rev.set_yticklabels([l.replace("β=", "") for l in all_labels])
    
    for s in [0, 3, 6, 7]:
        ax4_rev.axvline(s - 0.5, color='blue', linewidth=2, alpha=0.5)
        ax4_rev.axvline(s + 0.5, color='blue', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_state_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_state_heatmap.png")
    plt.close()
    
    # ===== Figure 5: 健康的状態 vs 依存的状態の訪問割合 =====
    fig5, axes5 = plt.subplots(1, 2, figsize=(16, 7))
    
    # Neutral状態(1-6)の合計 vs Drug/After状態(7-21)の合計
    # Addictionフェーズ
    neutral_addiction = []
    drug_addiction = []
    
    for label in all_labels:
        ratios = results[label]['addiction_state_ratios']
        neutral_addiction.append(sum(ratios[1:7]))  # 状態1-6
        drug_addiction.append(sum(ratios[7:]))       # 状態7-21
    
    ax5_add = axes5[0]
    x_pos = np.arange(len(all_labels))
    width = 0.35
    
    bars_neutral = ax5_add.bar(x_pos - width/2, neutral_addiction, width, 
                               label='Neutral区間 (1-6)', color='#2ecc71', alpha=0.7,
                               edgecolor='black', linewidth=1)
    bars_drug = ax5_add.bar(x_pos + width/2, drug_addiction, width,
                            label='Drug/After区間 (7-21)', color='#e74c3c', alpha=0.7,
                            edgecolor='black', linewidth=1)
    
    ax5_add.set_xlabel('β値 / モード', fontsize=12, fontweight='bold')
    ax5_add.set_ylabel('訪問割合 (%)', fontsize=12, fontweight='bold')
    ax5_add.set_title('Addictionフェーズ\nNeutral区間 vs Drug/After区間', fontsize=14, fontweight='bold')
    ax5_add.set_xticks(x_pos)
    ax5_add.set_xticklabels([l.replace("β=", "") for l in all_labels], fontsize=10)
    ax5_add.legend(fontsize=10)
    ax5_add.grid(True, alpha=0.3, axis='y')
    
    # Reversalフェーズ
    neutral_reversal = []
    drug_reversal = []
    
    for label in all_labels:
        ratios = results[label]['reversal_state_ratios']
        neutral_reversal.append(sum(ratios[1:7]))
        drug_reversal.append(sum(ratios[7:]))
    
    ax5_rev = axes5[1]
    
    bars_neutral_rev = ax5_rev.bar(x_pos - width/2, neutral_reversal, width,
                                   label='Neutral区間 (1-6) ※健康的', color='#2ecc71', alpha=0.7,
                                   edgecolor='black', linewidth=1)
    bars_drug_rev = ax5_rev.bar(x_pos + width/2, drug_reversal, width,
                                label='Drug/After区間 (7-21) ※依存的', color='#e74c3c', alpha=0.7,
                                edgecolor='black', linewidth=1)
    
    ax5_rev.set_xlabel('β値 / モード', fontsize=12, fontweight='bold')
    ax5_rev.set_ylabel('訪問割合 (%)', fontsize=12, fontweight='bold')
    ax5_rev.set_title('Reversalフェーズ\nNeutral区間 vs Drug/After区間\n(Reversalでは報酬が逆転)', fontsize=14, fontweight='bold')
    ax5_rev.set_xticks(x_pos)
    ax5_rev.set_xticklabels([l.replace("β=", "") for l in all_labels], fontsize=10)
    ax5_rev.legend(fontsize=10)
    ax5_rev.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_neutral_vs_drug.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_neutral_vs_drug.png")
    plt.close()
    
    # ===== Figure 6: 総合比較（Addiction Rate + Reversal Adaptation Rate） =====
    fig6, ax6 = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(all_labels))
    width = 0.35
    
    addiction_rates_all = [results[label]['mean_rate'] for label in all_labels]
    reversal_rates_all = [results[label]['reversal_rate'] for label in all_labels]
    addiction_sems = [results[label]['sem'] for label in all_labels]
    reversal_sems = [results[label]['reversal_sem'] for label in all_labels]
    
    colors_addiction = ['#3498db'] * len(fixed_betas) + (['#c0392b'] if "β=Learning" in results else [])
    colors_reversal = ['#2980b9'] * len(fixed_betas) + (['#e74c3c'] if "β=Learning" in results else [])
    
    bars_add = ax6.bar(x_pos - width/2, addiction_rates_all, width, yerr=addiction_sems,
                       label='Addiction依存症率', color='#e74c3c', alpha=0.7,
                       capsize=5, edgecolor='black', linewidth=1)
    bars_rev = ax6.bar(x_pos + width/2, reversal_rates_all, width, yerr=reversal_sems,
                       label='Reversal適応率', color='#2ecc71', alpha=0.7,
                       capsize=5, edgecolor='black', linewidth=1)
    
    ax6.set_xlabel('β値 / モード', fontsize=14, fontweight='bold')
    ax6.set_ylabel('割合 (%)', fontsize=14, fontweight='bold')
    ax6.set_title('β値別: 依存症率 vs 環境変化適応率\n(Learningモードは赤系統)', fontsize=16, fontweight='bold', pad=20)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([l.replace("β=", "") for l in all_labels], fontsize=11)
    ax6.legend(fontsize=12, loc='upper right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_combined_comparison.png", dpi=150, bbox_inches='tight')
    print(f"保存: {output_prefix}_combined_comparison.png")
    plt.close()
    
    # ===== Reversal統計サマリー出力 =====
    print("\n" + "=" * 60)
    print("Reversalフェーズ適応分析 - 統計サマリー")
    print("=" * 60)
    
    for label in all_labels:
        data = results[label]
        print(f"\n{label}:")
        print(f"  Reversal適応率: {data['reversal_rate']:.2f}% ± {data['reversal_sem']:.2f}%")
        print(f"  状態0訪問率 (Addiction): {data['addiction_state_ratios'][0]:.2f}%")
        print(f"  状態0訪問率 (Reversal): {data['reversal_state_ratios'][0]:.2f}%")
        print(f"  Drug区間訪問率 (Reversal): {sum(data['reversal_state_ratios'][7:]):.2f}%")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="β値比較実験の結果を可視化")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="beta_comparison_results",
        help="結果ファイルが保存されているディレクトリ"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="beta_comparison",
        help="出力ファイル名のプレフィックス"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"エラー: ディレクトリ {results_dir} が見つかりません")
        return
    
    print(f"結果ディレクトリ: {results_dir}")
    print("結果ファイルを読み込み中...")
    
    results = load_results(results_dir)
    
    if not results:
        print("エラー: 結果ファイルが見つかりません")
        return
    
    print(f"{len(results)} 個の結果を読み込みました")
    print("\n=== 基本比較グラフを作成中 ===")
    plot_comparison(results, args.output_prefix)
    
    print("\n=== Reversal適応分析グラフを作成中 ===")
    plot_reversal_analysis(results, args.output_prefix)
    
    print("\n完了！")


if __name__ == "__main__":
    main()