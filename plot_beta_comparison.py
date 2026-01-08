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
                    'std': np.std(data['run_rates']) * 100,
                    'sem': np.std(data['run_rates']) * 100 / np.sqrt(len(data['run_rates']))
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
                'std': np.std(data['run_rates']) * 100,
                'sem': np.std(data['run_rates']) * 100 / np.sqrt(len(data['run_rates']))
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
    print("\nグラフを作成中...")
    
    plot_comparison(results, args.output_prefix)
    
    print("\n完了！")


if __name__ == "__main__":
    main()
