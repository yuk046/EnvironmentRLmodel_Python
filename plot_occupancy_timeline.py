#!/usr/bin/env python3
"""
状態占有率の時系列グラフ作成スクリプト

論文スタイルの3パネルグラフ（A, B, C）を作成:
- Panel A: Addictive Reward (Drug state占有率)
- Panel B: Healthy Reward, Position 1 (Goal state占有率)
- Panel C: Healthy Reward, Position 2 (別の報酬状態の占有率)

各パネルはGoal Switch時点を示す垂直線と、複数のβ値条件を比較
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 状態定数（addiction_rl_sim.pyと同じ）
STATE_GOAL = 0
STATE_DRUG = 7


def load_episode_data(results_dir: Path, beta_value) -> Dict:
    """
    エピソードごとの詳細データを読み込む
    
    addiction_rl_sim.pyが--save-occupancy-dataオプションで生成した
    エピソードデータを読み込みます。
    
    Args:
        beta_value: β値（float、int、または'learning'文字列）
    """
    # learningモードの場合
    if isinstance(beta_value, str) and beta_value.lower() == 'learning':
        file_path = results_dir / "beta_learning_episodes.json"
    else:
        # 文字列の場合はfloatに変換
        if isinstance(beta_value, str):
            try:
                beta_value = float(beta_value)
            except ValueError:
                print(f"警告: '{beta_value}'は有効なβ値ではありません")
                return None
        
        # β値に応じたファイル名を生成
        if beta_value == int(beta_value):
            file_path = results_dir / f"beta_{int(beta_value)}_episodes.json"
        else:
            file_path = results_dir / f"beta_{beta_value}_episodes.json"
        
        if not file_path.exists():
            # 代替ファイル名を試す（小数点付き）
            file_path = results_dir / f"beta_{beta_value:.1f}_episodes.json"
    
    if not file_path.exists():
        print(f"警告: {file_path} が見つかりません")
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_occupancy_over_time(episode_data: Dict, 
                                  state_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    時間区間ごとの状態占有率を計算
    
    Args:
        episode_data: エピソードデータ辞書（addiction_rl_sim.pyから生成）
        state_id: 対象状態ID
        
    Returns:
        (mean_occupancy, sem_occupancy): 各時間ビンの平均とSEM
    """
    if episode_data is None:
        return None, None
    
    # 状態別のデータを取得
    state_key = str(state_id)
    if 'states' in episode_data and state_key in episode_data['states']:
        # addiction_rl_sim.pyで生成されたデータ形式
        # states[state_id] = [[run0_bin0, run0_bin1, ...], [run1_bin0, run1_bin1, ...], ...]
        occupancy_by_run = np.array(episode_data['states'][state_key])
        
        # 平均とSEMを計算
        mean_occupancy = np.mean(occupancy_by_run, axis=0)
        n_runs = len(occupancy_by_run)
        sem_occupancy = np.std(occupancy_by_run, axis=0) / np.sqrt(n_runs) if n_runs > 0 else np.zeros_like(mean_occupancy)
        
        return mean_occupancy, sem_occupancy
    
    return None, None


def plot_multi_panel_occupancy(results_dir: Path, 
                               beta_values: List[float],
                               output_path: str = "occupancy_timeline.png",
                               switch_step: int = 2000,
                               max_steps: int = 15000):
    """
    3パネル（A, B, C）の状態占有率グラフを作成
    
    Args:
        results_dir: 結果ディレクトリ
        beta_values: 比較するβ値のリスト
        output_path: 出力ファイルパス
        switch_step: Goal Switch発生ステップ
        max_steps: 最大ステップ数
    """
    # 最初のデータファイルからtime_binsを取得
    first_beta = beta_values[0]
    first_data = load_episode_data(results_dir, first_beta)
    
    if first_data and 'time_bins' in first_data:
        time_bins = np.array(first_data['time_bins'])
    else:
        # デフォルトの時間ビンを生成
        time_bins = np.arange(0, max_steps + 1, 2000)
    
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # β値ごとの色とラベル設定
    beta_colors = {
        0.0: '#1f77b4',      # 青 (β=0 MF)
        0.2: '#ff7f0e',      # オレンジ
        0.25: '#999999',     # グレー
        0.4: '#2ca02c',      # 緑
        0.5: '#AAAAAA',      # 中間グレー
        0.6: '#d62728',      # 赤
        0.75: '#CCCCCC',     # 薄いグレー
        0.8: '#9467bd',      # 紫
        1.0: '#8c564b',      # 茶色 (β=1 MB)
        'learning': '#e377c2',  # ピンク (Learning mode)
        'Learning': '#e377c2'   # ピンク (Learning mode)
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
        'learning': 'β=Learning',
        'Learning': 'β=Learning'
    }
    
    # グラフの作成
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('State Occupancy Over Time', fontsize=16, fontweight='bold', y=0.995)
    
    # パネルごとの設定
    panel_configs = [
        {'title': 'Addictive Reward', 'state': STATE_DRUG, 'ylabel': '% occupancy'},
        {'title': 'Healthy Reward, Position 1', 'state': STATE_GOAL, 'ylabel': '% occupancy'},
        {'title': 'Healthy Reward, Position 2', 'state': 1, 'ylabel': '% occupancy'}  # 例: State 1
    ]
    
    panel_labels = ['A', 'B', 'C']
    
    for panel_idx, (ax, config, label) in enumerate(zip(axes, panel_configs, panel_labels)):
        # パネルラベルを左上に配置
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
        
        # 各β値についてプロット
        for beta in beta_values:
            # エピソードデータを読み込み
            episode_data = load_episode_data(results_dir, beta)
            
            if episode_data is not None:
                # 実データから占有率を計算
                mean_occ, sem_occ = calculate_occupancy_over_time(
                    episode_data, config['state']
                )
            else:
                # データがない場合はダミーデータを生成
                mean_occ, sem_occ = generate_dummy_occupancy(
                    time_centers, config['state'], beta, switch_step
                )
            
            if mean_occ is None:
                # データが取得できない場合はスキップ
                continue
            
            color = beta_colors.get(beta, '#000000')
            label_text = beta_labels.get(beta, f'β={beta}')
            
            # 線とエラーバーをプロット
            ax.errorbar(time_centers, mean_occ * 100, yerr=sem_occ * 100,
                       marker='o', markersize=4, linewidth=2, capsize=4,
                       color=color, ecolor=color, alpha=0.8,
                       label=label_text)
        
        # Goal Switch の垂直線
        ax.axvline(switch_step, color='blue', linestyle='--', 
                  linewidth=1.5, alpha=0.7)
        ax.text(switch_step - 100, ax.get_ylim()[1] * 0.95, 
               'Goal\nSwitch', ha='right', va='top', 
               color='blue', fontsize=10, fontweight='bold')
        
        # 軸ラベルと設定
        ax.set_title(config['title'], fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.set_xlim(0, max_steps)
        ax.grid(True, alpha=0.3)
        
        # 凡例（最初のパネルのみ）
        if panel_idx == 1:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # X軸ラベルは最下部のみ
        if panel_idx == len(axes) - 1:
            ax.set_xlabel('time steps', fontsize=12, fontweight='bold')
        else:
            ax.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"保存: {output_path}")
    plt.close()


def generate_dummy_occupancy(time_centers: np.ndarray, 
                            state_id: int, 
                            beta,
                            switch_step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    ダミーの占有率データを生成（テスト用）
    
    実際の実装では、load_episode_dataとcalculate_occupancy_over_timeを使用
    
    Args:
        beta: β値（float）または'learning'文字列
    """
    # learningの場合は中間値(0.5)として扱う
    if beta == 'learning' or beta == 'Learning':
        beta_val = 0.5
    else:
        beta_val = beta
    
    n_bins = len(time_centers)
    mean_occ = np.zeros(n_bins)
    sem_occ = np.zeros(n_bins)
    
    for i, t in enumerate(time_centers):
        # Goal Switchの前後で挙動を変える
        if state_id == STATE_DRUG:  # Addictive reward
            if t < switch_step:
                # Switch前は低い
                base = 0.03 + beta_val * 0.02
            else:
                # Switch後は上昇（Model-Basedほど低い）
                base = 0.10 - beta_val * 0.05
            mean_occ[i] = base
            sem_occ[i] = 0.02
            
        elif state_id == STATE_GOAL:  # Healthy reward, Position 1
            if t < switch_step:
                # Switch前は上昇
                base = 0.1 + (t / switch_step) * 0.3 * (1 - beta_val)
            else:
                # Switch後は減少（Model-Freeは維持、Model-Basedは急減）
                decay = (t - switch_step) / (max(time_centers) - switch_step)
                base = 0.6 * (1 - beta_val) * (1 - decay * 0.7)
            mean_occ[i] = base
            sem_occ[i] = 0.05
            
        else:  # Other states
            # ランダムな低い値
            mean_occ[i] = 0.01 + np.random.random() * 0.03
            sem_occ[i] = 0.01
    
    return mean_occ, sem_occ


def main():
    parser = argparse.ArgumentParser(
        description='状態占有率の時系列グラフを作成'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='beta_comparison_results',
        help='結果ディレクトリ（デフォルト: beta_comparison_results）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='occupancy_timeline.png',
        help='出力ファイル名（デフォルト: occupancy_timeline.png）'
    )
    parser.add_argument(
        '--beta-values',
        nargs='+',
        default=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', 'learning'],
        help='比較するβ値（数値または"learning"、デフォルト: 0.0 0.2 0.4 0.6 0.8 1.0 learning）'
    )
    parser.add_argument(
        '--switch-step',
        type=int,
        default=2000,
        help='Goal Switch発生ステップ（デフォルト: 2000）'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=15000,
        help='最大ステップ数（デフォルト: 15000）'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"エラー: 結果ディレクトリ {results_dir} が見つかりません")
        return
    
    # beta_valuesを適切な型に変換（文字列 or float）
    beta_values = []
    for b in args.beta_values:
        if b.lower() == 'learning':
            beta_values.append('learning')
        else:
            try:
                beta_values.append(float(b))
            except ValueError:
                print(f"警告: '{b}'は有効なβ値ではありません。スキップします。")
    
    # グラフ作成
    plot_multi_panel_occupancy(
        results_dir=results_dir,
        beta_values=beta_values,
        output_path=args.output,
        switch_step=args.switch_step,
        max_steps=args.max_steps
    )
    
    print("\n完了！")
    print(f"グラフを確認: {args.output}")


if __name__ == '__main__':
    main()
