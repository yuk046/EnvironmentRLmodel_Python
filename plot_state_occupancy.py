#!/usr/bin/env python3
"""
既存の結果データから状態占有率グラフを作成（簡易版）

既存のbeta_comparison_results/やbeta_reversal_results/から
addiction_state_ratiosとreversal_state_ratiosを使用して、
2パネル（Addiction Phase / Reversal Phase）のグラフを作成します。
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 状態定数
STATE_GOAL = 0
STATE_DRUG = 7


def load_results(results_dir: Path, beta_values: List[float]) -> Dict:
    """結果ファイルを読み込む"""
    results = {}
    
    for beta in beta_values:
        # ファイル名のフォーマット
        if beta == int(beta):
            file_path = results_dir / f"beta_{int(beta)}.json"
        else:
            file_path = results_dir / f"beta_{beta}.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                results[beta] = {
                    'addiction_state_ratios': data.get('addiction_state_ratios', [0.0] * 22),
                    'reversal_state_ratios': data.get('reversal_state_ratios', [0.0] * 22),
                }
        else:
            print(f"警告: {file_path} が見つかりません")
    
    return results


def plot_state_occupancy_comparison(results: Dict, 
                                   output_path: str = "state_occupancy_comparison.png",
                                   states_to_plot: List[int] = None):
    """
    2パネル（Addiction Phase / Reversal Phase）の状態占有率グラフを作成
    
    Args:
        results: β値ごとの結果データ
        output_path: 出力ファイルパス
        states_to_plot: プロットする状態のリスト（デフォルト: [0, 7]）
    """
    if states_to_plot is None:
        states_to_plot = [STATE_GOAL, STATE_DRUG]
    
    # β値をソート
    beta_values = sorted(results.keys())
    
    # β値ごとの色設定
    beta_colors = {
        0.0: '#666666',      # 濃いグレー
        0.2: '#888888',
        0.25: '#999999',
        0.4: '#AAAAAA',
        0.5: '#AAAAAA',
        0.6: '#BBBBBB',
        0.75: '#CCCCCC',
        0.8: '#DDDDDD',
        1.0: '#FF0000',      # 赤
    }
    
    beta_labels = {
        0.0: 'β=0 MF',
        0.2: 'β=0.2',
        0.25: 'β=0.25',
        0.4: 'β=0.4',
        0.5: 'β=0.5',
        0.6: 'β=0.6',
        0.75: 'β=0.75',
        0.8: 'β=0.8',
        1.0: 'β=1 MB',
    }
    
    # 状態の名前
    state_names = {
        0: 'Goal State',
        7: 'Drug State',
        1: 'Neutral State 1',
        14: 'Withdrawal State',
    }
    
    # グラフの作成
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('State Occupancy Comparison Across β Values', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    phases = [
        ('Addiction Phase', 'addiction_state_ratios'),
        ('Reversal Phase', 'reversal_state_ratios')
    ]
    panel_labels = ['A', 'B']
    
    for phase_idx, (ax, (phase_title, data_key), label) in enumerate(zip(axes, phases, panel_labels)):
        # パネルラベル
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
        
        # 状態ごとにサブグラフ（複数状態の場合）
        width = 0.8 / len(states_to_plot)
        
        for state_idx, state_id in enumerate(states_to_plot):
            x_positions = np.arange(len(beta_values)) + state_idx * width
            occupancies = []
            
            for beta in beta_values:
                state_ratios = results[beta][data_key]
                if state_id < len(state_ratios):
                    occupancies.append(state_ratios[state_id])
                else:
                    occupancies.append(0.0)
            
            # 状態名とβ値で色を変える
            colors = [beta_colors.get(beta, '#000000') for beta in beta_values]
            
            # 棒グラフ
            bars = ax.bar(x_positions, occupancies, width, 
                         label=state_names.get(state_id, f'State {state_id}'),
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # 各棒に色を設定
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)
        
        # 軸設定
        ax.set_title(phase_title, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('% Occupancy', fontsize=12)
        ax.set_xticks(np.arange(len(beta_values)) + width * (len(states_to_plot) - 1) / 2)
        ax.set_xticklabels([beta_labels.get(beta, f'β={beta}') for beta in beta_values])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=10)
        
        # X軸ラベルは最下部のみ
        if phase_idx == len(axes) - 1:
            ax.set_xlabel('β Value', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存: {output_path}")
    plt.close()


def plot_all_states_heatmap(results: Dict,
                            output_path: str = "state_occupancy_heatmap.png"):
    """
    全状態の占有率をヒートマップで可視化
    
    Args:
        results: β値ごとの結果データ
        output_path: 出力ファイルパス
    """
    beta_values = sorted(results.keys())
    num_states = 22
    
    # データの準備
    addiction_matrix = np.zeros((len(beta_values), num_states))
    reversal_matrix = np.zeros((len(beta_values), num_states))
    
    for i, beta in enumerate(beta_values):
        addiction_matrix[i, :] = results[beta]['addiction_state_ratios']
        reversal_matrix[i, :] = results[beta]['reversal_state_ratios']
    
    # グラフの作成
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('State Occupancy Heatmap Across β Values and States',
                 fontsize=16, fontweight='bold')
    
    phases = [
        ('Addiction Phase', addiction_matrix),
        ('Reversal Phase', reversal_matrix)
    ]
    
    for ax, (phase_title, matrix) in zip(axes, phases):
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # 軸設定
        ax.set_title(phase_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('State ID', fontsize=12)
        ax.set_ylabel('β Value', fontsize=12)
        
        ax.set_xticks(range(num_states))
        ax.set_xticklabels(range(num_states), fontsize=8)
        
        ax.set_yticks(range(len(beta_values)))
        ax.set_yticklabels([f'{beta:.1f}' for beta in beta_values])
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('% Occupancy', fontsize=10)
        
        # 重要な状態に注釈
        for state_id, name in [(0, 'Goal'), (7, 'Drug'), (14, 'Withdrawal')]:
            ax.axvline(state_id, color='blue', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(state_id, -0.5, name, ha='center', va='top', 
                   fontsize=9, color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='既存の結果から状態占有率グラフを作成'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='beta_comparison_results',
        help='結果ディレクトリ（デフォルト: beta_comparison_results）'
    )
    parser.add_argument(
        '--output-comparison',
        type=str,
        default='state_occupancy_comparison.png',
        help='比較グラフの出力ファイル名'
    )
    parser.add_argument(
        '--output-heatmap',
        type=str,
        default='state_occupancy_heatmap.png',
        help='ヒートマップの出力ファイル名'
    )
    parser.add_argument(
        '--beta-values',
        type=float,
        nargs='+',
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help='比較するβ値'
    )
    parser.add_argument(
        '--states',
        type=int,
        nargs='+',
        default=[0, 7],
        help='プロットする状態ID（デフォルト: 0 7 = Goal, Drug）'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"エラー: 結果ディレクトリ {results_dir} が見つかりません")
        return
    
    # 結果を読み込む
    print("結果データを読み込み中...")
    results = load_results(results_dir, args.beta_values)
    
    if not results:
        print("エラー: 有効なデータが見つかりませんでした")
        return
    
    print(f"読み込み完了: {len(results)} 個のβ値")
    
    # 比較グラフを作成
    print("\n比較グラフを作成中...")
    plot_state_occupancy_comparison(
        results,
        output_path=args.output_comparison,
        states_to_plot=args.states
    )
    
    # ヒートマップを作成
    print("ヒートマップを作成中...")
    plot_all_states_heatmap(
        results,
        output_path=args.output_heatmap
    )
    
    print("\n完了！")


if __name__ == '__main__':
    main()
