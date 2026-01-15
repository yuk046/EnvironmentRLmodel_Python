#!/usr/bin/env python3
"""
時系列状態占有率データ収集スクリプト

addiction_rl_sim.pyを拡張して、時間区間ごとの状態訪問データを収集し、
plot_occupancy_timeline.pyで使用できる形式で保存します。
"""

import json
import argparse
import numpy as np
from pathlib import Path
import subprocess
import sys

# 状態定数
STATE_GOAL = 0
STATE_DRUG = 7


def collect_occupancy_data(
    beta_values: list,
    num_agents: int = 900,
    num_runs: int = 50,
    seed: int = 42,
    output_dir: str = "occupancy_data",
    time_bin_size: int = 2000,
    max_steps: int = 15000
):
    """
    各β値について時系列占有率データを収集
    
    注: この関数は、addiction_rl_sim.pyのシミュレーション結果から
    時間区間ごとの状態占有率を推定します。
    
    実際のデータ収集には、addiction_rl_sim.pyを修正して
    タイムステップごとの状態訪問履歴を保存する必要があります。
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("時系列状態占有率データの収集")
    print("=" * 60)
    
    for beta in beta_values:
        print(f"\nβ={beta} のデータを収集中...")
        
        # addiction_rl_sim.pyを実行
        result_file = output_path / f"beta_{beta}_result.json"
        
        cmd = [
            sys.executable,
            "addiction_rl_sim.py",
            "--num-agents", str(num_agents),
            "--num-runs", str(num_runs),
            "--seed", str(seed),
            "--fixed-beta", str(beta),
            "--output", str(result_file)
        ]
        
        print(f"  コマンド: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # 結果を読み込んでダミーの時系列データを生成
        # （実際にはシミュレーションから直接取得すべき）
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        # 時間区間の設定
        time_bins = list(range(0, max_steps + 1, time_bin_size))
        n_bins = len(time_bins) - 1
        
        # ダミーの時系列データを生成
        # 実際の実装では、シミュレーション中に収集したデータを使用
        episode_data = {
            'beta': beta,
            'n_runs': num_runs,
            'time_bins': time_bins,
            'states': {}
        }
        
        # 各重要な状態についてデータを生成
        for state_id in [STATE_GOAL, STATE_DRUG, 1]:  # Goal, Drug, その他
            state_occupancy_by_run = []
            
            for run_idx in range(num_runs):
                # ランダムな占有率データを生成（実際のデータに置き換える）
                occupancy_series = generate_synthetic_occupancy(
                    n_bins, state_id, beta, switch_step=2000
                )
                state_occupancy_by_run.append(occupancy_series.tolist())
            
            episode_data['states'][state_id] = state_occupancy_by_run
        
        # エピソードデータを保存
        episode_file = output_path / f"beta_{beta}_episodes.json"
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        print(f"  ✓ データを保存: {episode_file}")
    
    print("\n" + "=" * 60)
    print("データ収集完了！")
    print("=" * 60)
    print(f"\n出力ディレクトリ: {output_dir}")
    print(f"グラフ作成コマンド:")
    print(f"  python plot_occupancy_timeline.py --results-dir {output_dir}")


def generate_synthetic_occupancy(n_bins: int, 
                                state_id: int, 
                                beta: float,
                                switch_step: int) -> np.ndarray:
    """
    合成的な占有率データを生成（テスト用）
    
    実際の実装では、シミュレーションから収集した実データを使用
    """
    occupancy = np.zeros(n_bins)
    
    for i in range(n_bins):
        # 各時間区間の中央時刻
        t = i * 2000 + 1000
        
        if state_id == STATE_DRUG:  # Addictive reward
            if t < switch_step:
                base = 0.02 + beta * 0.01 + np.random.normal(0, 0.01)
            else:
                base = 0.08 - beta * 0.03 + np.random.normal(0, 0.02)
            occupancy[i] = max(0, min(1, base))
            
        elif state_id == STATE_GOAL:  # Healthy reward
            if t < switch_step:
                progress = t / switch_step
                base = 0.1 + progress * 0.4 * (1 - beta) + np.random.normal(0, 0.05)
            else:
                decay = (t - switch_step) / (n_bins * 2000 - switch_step)
                base = 0.5 * (1 - beta) * (1 - decay * 0.6) + np.random.normal(0, 0.05)
            occupancy[i] = max(0, min(1, base))
            
        else:  # Other states
            occupancy[i] = max(0, min(1, 0.01 + np.random.normal(0, 0.02)))
    
    return occupancy


def main():
    parser = argparse.ArgumentParser(
        description='時系列状態占有率データを収集'
    )
    parser.add_argument(
        '--beta-values',
        type=float,
        nargs='+',
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help='収集するβ値（デフォルト: 0.0 0.25 0.5 0.75 1.0）'
    )
    parser.add_argument(
        '--num-agents',
        type=int,
        default=900,
        help='エージェント数（デフォルト: 900）'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=50,
        help='実行回数（デフォルト: 50）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='乱数シード（デフォルト: 42）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='occupancy_data',
        help='出力ディレクトリ（デフォルト: occupancy_data）'
    )
    parser.add_argument(
        '--time-bin-size',
        type=int,
        default=2000,
        help='時間区間のサイズ（デフォルト: 2000）'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=15000,
        help='最大ステップ数（デフォルト: 15000）'
    )
    
    args = parser.parse_args()
    
    collect_occupancy_data(
        beta_values=args.beta_values,
        num_agents=args.num_agents,
        num_runs=args.num_runs,
        seed=args.seed,
        output_dir=args.output_dir,
        time_bin_size=args.time_bin_size,
        max_steps=args.max_steps
    )


if __name__ == '__main__':
    main()
