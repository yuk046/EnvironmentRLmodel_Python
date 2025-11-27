import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# ==========================================
# 1. 定数・環境設定 (論文準拠)
# ==========================================
NUM_STATES = 22
NUM_ACTIONS = 9

STATE_GOAL = 0        # Goal state
STATE_GOAL_ENTRY = 1  # Neutral state leading to goal
STATE_START = 3       # Start state (S0)
STATE_DRUG = 7        # Drug state
STATE_AW_SPECIAL = 14 # Special withdrawal state

# 状態セットの定義
STATE_NEUTRALS = tuple(range(1, 7))
STATE_NEUTRAL_SET = set(STATE_NEUTRALS)
NEUTRAL_MAX = STATE_NEUTRALS[-1]

STATE_AFTEREFFECTS = tuple(range(8, 22))
STATE_AFTER_SET = set(STATE_AFTEREFFECTS)
AFTER_MAX = STATE_AFTEREFFECTS[-1]

# ハイパーパラメータ
DISCOUNT = 0.9
ALPHA_MF = 0.05
MB_DECAY = 0.01      # 思考結果の減衰率
EPSILON = 0.1        # 探索率
N_PRIORITIZED_SWEEPS = 50
T_MB = 1.0           # Softmax temperature for planning

# 報酬設定
R_G = 1.0            # Goal reward
R_P = -4.0           # Punishment (shock)
R_SKIP_LONG = -0.3   # Cost of skipping states
AFTER_PHASE_REWARDS = (-0.3, -1.2) # Pre-drug / Addiction phase rewards in aftereffect

# アクション定義
ACTION_AS2 = 0
ACTION_AS3 = 1
ACTION_AS4 = 2
ACTION_AS5 = 3
ACTION_AS6 = 4
ACTION_AS7 = 5
ACTION_GOAL = 6
ACTION_DRUG = 7
ACTION_AW = 8

AS_ACTION_TARGETS = {
    ACTION_AS2: 1,
    ACTION_AS3: 2,
    ACTION_AS4: 3,
    ACTION_AS5: 4,
    ACTION_AS6: 5,
    ACTION_AS7: 6,
}

ACTION_NAMES = {
    ACTION_AS2: "as2",
    ACTION_AS3: "as3",
    ACTION_AS4: "as4",
    ACTION_AS5: "as5",
    ACTION_AS6: "as6",
    ACTION_AS7: "as7",
    ACTION_GOAL: "ag",
    ACTION_DRUG: "ad",
    ACTION_AW: "aw",
}

# 遷移確率テーブル (f1: Pre-drug, f2: Addiction)
NEUTRAL_MOVE_SUCCESS = [0.99, 0.99]
NEUTRAL_SKIP_SUCCESS = [0.0001, 0.0001]
AFTER_AG_EXIT = [0.001, 0.001]
AFTER_AS_EXIT = [0.001, 0.001]
AW_FORWARD = [0.4995, 0.4995]
AW_BACKWARD = [0.4995, 0.4995]
AW_SPECIAL_MOVE = [0.2, 0.2]
AW_SPECIAL_EXIT = [0.6, 0.6]
AD_FORWARD = [0.745, 0.745]
AD_BACKWARD = [0.245, 0.245]

# フェーズ定義: (名前, ステップ数, 薬物報酬)
PHASES: Tuple[Tuple[str, int, float], ...] = (
    ("pre-drug", 50, 0.0),
    ("addiction", 1000, 10.0),
)

# 報酬テーブルの事前構築
PHASE_REWARD_TABLE = np.zeros((len(PHASES), NUM_STATES))
PHASE_DRUG_REWARDS = np.array([phase[2] for phase in PHASES], dtype=np.float64)
for idx in range(len(PHASES)):
    PHASE_REWARD_TABLE[idx, list(STATE_AFTEREFFECTS)] = AFTER_PHASE_REWARDS[idx]


# ==========================================
# 2. Numba 高速化カーネル (Model-Based Planning)
# ==========================================
@jit(nopython=True, cache=True)
def run_prioritized_sweeping(
    q_mb, 
    model_counts, 
    model_rewards, 
    model_visits, 
    num_states, 
    num_actions, 
    n_sweeps, 
    t_mb, 
    discount, 
    decay
):
    """
    Model-Basedエージェントの思考プロセス (Prioritized Sweeping)
    毎回計算をリセットし、記憶に基づいて50回シミュレーションを行う。
    """
    # 経験がまだなければ何もしない
    # sum() は重いので簡易チェック
    has_experience = False
    for s in range(num_states):
        for a in range(num_actions):
            if model_visits[s, a] > 0:
                has_experience = True
                break
        if has_experience: break
    
    if not has_experience:
        return

    # --- 思考の初期化 (Reset) ---
    # 優先度キュー(H)と価値推定(V)を毎回ゼロからスタート
    H = np.zeros(num_states, dtype=np.float64)
    V = np.zeros(num_states, dtype=np.float64)
    
    # 作業用バッファ
    probs = np.zeros(num_states, dtype=np.float64)
    Q_vals = np.zeros(num_actions, dtype=np.float64)
    
    # --- Planning Loop ---
    # modelベースのQ値を優先度付きスイープで更新

    # n_sweeps回の思考
    for _ in range(n_sweeps):
        # 1. 思考する状態の選択 (Softmax on Priority H)
        max_h = np.max(H)
        # 数値安定性のため max_h を引く
        h_exp = np.exp((H - max_h) / t_mb)
        sum_h_exp = np.sum(h_exp)
        
        if sum_h_exp > 1e-9:
            probs = h_exp / sum_h_exp
        else:
            # 全て0なら均等確率
            probs[:] = 1.0 / num_states
            
        # 確率的選択 (CDF)
        rand_val = np.random.random()
        cumulative = 0.0
        s_tilde = num_states - 1
        for i in range(num_states):
            cumulative += probs[i]
            if rand_val <= cumulative:
                s_tilde = i
                break
        
        # 2. 選択した状態のQ値をモデルから再計算
        for a in range(num_actions):
            visits = model_visits[s_tilde, a]
            # 状態に訪れた経験がなければQ値は0
            if visits <= 0:
                Q_vals[a] = 0.0
                continue
            
            # R(s,a)
            r_expected = model_rewards[s_tilde, a] / visits
            
            # sum(P(s'|s,a) * V(s'))
            future_v = 0.0
            for next_s in range(num_states):
                count = model_counts[s_tilde, a, next_s]
                if count > 0:
                    prob_trans = count / visits
                    future_v += prob_trans * V[next_s]
            
            Q_vals[a] = r_expected + discount * future_v
            
        # 結果を保存
        q_mb[s_tilde] = Q_vals
        
        # 3. 優先度 H の更新
        # V(s_tilde) の更新幅 delta
        max_q = np.max(Q_vals)
        delta = np.abs(V[s_tilde] - max_q)
        V[s_tilde] = max_q
        
        # 全状態の優先度更新
        # 画像のアルゴリズム:
        # for all s: h(s) = delta * max_a P(s_tilde | s, a)
        # H(s_tilde) = h(s_tilde)
        # for all s != s_tilde: H(s) = max(h(s), H(s))
        
        for s in range(num_states):
            max_p = 0.0
            for a in range(num_actions):
                visits_s = model_visits[s, a]
                if visits_s <= 0:
                    continue
                
                # 遷移確率 P(s_tilde | s, a)
                cnt = model_counts[s, a, s_tilde]
                if cnt > 0:
                    p = cnt / visits_s
                    if p > max_p:
                        max_p = p
            
            h_s = delta * max_p
            
            if s == s_tilde:
                H[s] = h_s
            else:
                if h_s > H[s]:
                    H[s] = h_s

    # --- Decay (思考結果の減衰) ---
    if decay > 0:
        q_mb *= (1.0 - decay)


# ==========================================
# 3. クラス定義 (Environment / Agent)
# ==========================================
@dataclass
class PhaseResult:
    drug_choices: int = 0
    healthy_choices: int = 0

class AddictionEnvironment:
    def __init__(self, rng: np.random.Generator):
        self.state = STATE_START
        self.rng = rng

    def reset(self) -> int:
        self.state = STATE_START
        return self.state

    def step(self, action: int, phase_idx: int) -> Tuple[int, float]:
        curr = self.state
        next_state = curr
        
        if curr in STATE_NEUTRAL_SET:
            next_state = self._transition_neutral(curr, action, phase_idx)
        elif curr == STATE_DRUG or curr in STATE_AFTER_SET:
            next_state = self._transition_aftereffect(curr, action, phase_idx)
        elif curr == STATE_GOAL:
            # GoalからはStartへ戻るか、Goalに留まるか(行動による)
            next_state = STATE_START if action == ACTION_GOAL else STATE_GOAL
        else:
            # Start地点など
            next_state = curr

        reward = self._reward(curr, action, next_state, phase_idx)
        self.state = next_state
        return next_state, reward

    def _transition_neutral(self, state: int, action: int, phase_idx: int) -> int:
        # AS行動 (空間移動)
        if action in AS_ACTION_TARGETS:
            target = AS_ACTION_TARGETS[action]
            if target == state: return state
            
            dist = abs(target - state)
            success_prob = NEUTRAL_MOVE_SUCCESS[phase_idx] if dist == 1 else NEUTRAL_SKIP_SUCCESS[phase_idx]
            
            if self.rng.random() < success_prob:
                return target
            return state
            
        if action == ACTION_GOAL and state == STATE_GOAL_ENTRY:
            return STATE_GOAL
        if action == ACTION_DRUG and state == NEUTRAL_MAX:
            return STATE_DRUG
            
        return state

    def _transition_aftereffect(self, state: int, action: int, phase_idx: int) -> int:
        roll = self.rng.random()
        
        # 離脱 (Exit) の判定
        if action == ACTION_GOAL:
            return STATE_START if roll < AFTER_AG_EXIT[phase_idx] else state
        if action in AS_ACTION_TARGETS:
            return STATE_START if roll < AFTER_AS_EXIT[phase_idx] else state
            
        # 内部移動 (AW/AD)
        if action == ACTION_AW:
            # Special State (分岐点)
            if state == STATE_AW_SPECIAL:
                p_move = AW_SPECIAL_MOVE[phase_idx] / 2.0
                if roll < p_move:
                    return max(STATE_DRUG, state - 1)
                roll -= p_move
                if roll < p_move:
                    return min(AFTER_MAX, state + 1)
                roll -= p_move
                if roll < AW_SPECIAL_EXIT[phase_idx]:
                    return STATE_START
                return state
            
            # Normal Aftereffect State
            if roll < AW_FORWARD[phase_idx]:
                return min(AFTER_MAX, state + 1)
            roll -= AW_FORWARD[phase_idx]
            if roll < AW_BACKWARD[phase_idx]:
                return max(STATE_DRUG, state - 1)
            return state

        if action == ACTION_DRUG:
            if roll < AD_FORWARD[phase_idx]:
                return min(AFTER_MAX, state + 1)
            roll -= AD_FORWARD[phase_idx]
            if roll < AD_BACKWARD[phase_idx]:
                return max(STATE_DRUG, state - 1)
            return state
            
        return state

    def _reward(self, current_state: int, action: int, next_state: int, phase_idx: int) -> float:
        r = PHASE_REWARD_TABLE[phase_idx, next_state]

        if current_state == STATE_GOAL and action == ACTION_GOAL and next_state == STATE_START:
            r += R_G
        elif (current_state == NEUTRAL_MAX and action == ACTION_DRUG and next_state == STATE_DRUG):
            r += PHASE_DRUG_REWARDS[phase_idx]
        elif current_state == STATE_DRUG and next_state == STATE_DRUG:
            r += AFTER_PHASE_REWARDS[phase_idx]
        
        # Neutralエリアでのロングジャンプ失敗コストなどは簡略化のため省略せず実装
        if (current_state in STATE_NEUTRAL_SET and next_state in STATE_NEUTRAL_SET 
            and abs(next_state - current_state) > 1):
            r += R_SKIP_LONG
            
        # Drug/Aftereffect から Neutral/Goal への遷移 (離脱時の罰則)
        if ((current_state == STATE_DRUG or current_state in STATE_AFTER_SET) and 
            (next_state in STATE_NEUTRAL_SET or next_state == STATE_GOAL or next_state == STATE_GOAL_ENTRY)):
            r += R_P
            
        return r


class HybridAgent:
    def __init__(self, beta: float, rng: np.random.Generator, mb_forget: bool = False):
        self.beta = beta
        self.rng = rng
        self.mb_forget = mb_forget
        
        # Qテーブル (MF, MB)
        self.q_mf = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.q_mb = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        
        # メンタルモデル (Numba用にfloat64で定義)
        # model_counts: [state, action, next_state] -> count
        self.model_counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=np.float64)
        self.model_rewards = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.model_visits = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)

    def select_action(self, state: int) -> int:
        # MB Planning (JIT function call)
        self._plan_q_values()
        
        # ε-Greedy
        if self.rng.random() < EPSILON:
            return int(self.rng.integers(NUM_ACTIONS))
            
        # Hybrid Q-value
        q_mix = self.beta * self.q_mb[state] + (1.0 - self.beta) * self.q_mf[state]
        
        # Argmax (tie-break random)
        max_val = np.max(q_mix)
        best_actions = np.flatnonzero(np.isclose(q_mix, max_val,rtol=1e-08, atol=1e-12))
        return int(self.rng.choice(best_actions))

    def observe(self, state: int, action: int, reward: float, next_state: int, phase_idx: int):
        # MF Update (Q-Learning)
        td_target = reward + DISCOUNT * np.max(self.q_mf[next_state])
        self.q_mf[state, action] += ALPHA_MF * (td_target - self.q_mf[state, action])
        
        # MB Model Update
        self.model_counts[state, action, next_state] += 1.0
        self.model_rewards[state, action] += reward
        self.model_visits[state, action] += 1.0

    def _plan_q_values(self):
        """高速化カーネルを呼び出す"""
        if self.mb_forget:
            # 完全忘却モード: 毎回MB推定をゼロから再構築
            self.q_mb.fill(0.0)
            decay = MB_DECAY
        else:
            decay = MB_DECAY

        run_prioritized_sweeping(
            self.q_mb,
            self.model_counts,
            self.model_rewards,
            self.model_visits,
            NUM_STATES,
            NUM_ACTIONS,
            N_PRIORITIZED_SWEEPS,
            T_MB,
            DISCOUNT,
            decay
        )


# ==========================================
# 4. シミュレーション実行 & Main
# ==========================================
def simulate(
    beta: float,
    num_agents: int,
    num_runs: int,
    base_seed: int,
    progress_cb=None,
    mb_forget: bool = False,
    debug_episode: bool = False,
    debug_csv_path: Optional[str] = None,
    debug_txt_path: Optional[str] = None,
) -> float:
    
    addictions = 0
    total_agents = num_agents * num_runs
    debug_records = []
    
    # テキストログファイルを開く (追記モードではなく新規作成)
    log_file = None
    if debug_episode and debug_txt_path:
        log_file = open(debug_txt_path, "w", encoding="utf-8")

    try:
        # Numba内の乱数シード固定 (再現性のため)
        np.random.seed(base_seed)

        for seed_idx in range(num_runs):
            # Agent/Env用の乱数生成器
            current_seed = base_seed + seed_idx
            rng = np.random.default_rng(current_seed)
            
            for agent_idx in range(num_agents):
                env = AddictionEnvironment(rng)
                agent = HybridAgent(beta, rng, mb_forget=mb_forget)
                
                state = env.reset()
                counts = PhaseResult()
                
                # フェーズ実行
                for phase_idx, (_, length, _) in enumerate(PHASES):
                    report_points = {1, length // 2, length}
                    
                    for step_in_phase in range(length):
                        # Progress Log
                        if progress_cb and (step_in_phase + 1) in report_points:
                            progress_cb(beta, seed_idx, agent_idx, num_agents, num_runs, phase_idx, step_in_phase + 1, length)
                        
                        # 行動選択前にQ値を取得したいが、select_action内でMBのPlanningが走るため
                        # select_action後に取得すると、そのステップでのPlanning結果が反映された状態になる
                        # ここでは「行動選択に使われたQ値」に近いものを表示するため、select_action直後の値を参照する
                        
                        action = agent.select_action(state)
                        
                        # Debug出力用にQ値を取得 (現在の状態 state における全行動のQ値)
                        current_q_mf = agent.q_mf[state].copy()
                        current_q_mb = agent.q_mb[state].copy()
                        
                        next_state, reward = env.step(action, phase_idx)
                        agent.observe(state, action, reward, next_state, phase_idx)

                        if debug_episode and seed_idx == 0 and agent_idx == 0:
                            # ログメッセージの構築
                            q_mf_str = ", ".join([f"{x:.2f}" for x in current_q_mf])
                            q_mb_str = ", ".join([f"{x:.2f}" for x in current_q_mb])
                            
                            log_lines = [
                                f"[DEBUG] Beta={beta:.1f} Phase={PHASES[phase_idx][0]} Step={step_in_phase+1}",
                                f"  State: {state} -> Action: {ACTION_NAMES.get(action, str(action))} -> Next: {next_state} (Reward: {reward})",
                                f"  Q_MF: [{q_mf_str}]",
                                f"  Q_MB: [{q_mb_str}]",
                                "-" * 40
                            ]
                            
                            # コンソール出力
                            for line in log_lines:
                                print(line)
                            
                            # ファイル出力
                            if log_file:
                                for line in log_lines:
                                    log_file.write(line + "\n")

                            debug_records.append({
                                "phase_name": PHASES[phase_idx][0],
                                "phase_idx": phase_idx,
                                "step": step_in_phase + 1,
                                "state": state,
                                "action": action,
                                "action_name": ACTION_NAMES.get(action, str(action)),
                                "reward": reward,
                                "next_state": next_state,
                            })
                        
                        # 統計収集 (Addictionフェーズのみ)
                        if phase_idx == 1:
                            if next_state == STATE_DRUG and reward > 0:
                                counts.drug_choices += 1
                            elif state == STATE_GOAL and reward > 0:
                                counts.healthy_choices += 1
                                
                        state = next_state
                
                # 依存判定
                if counts.drug_choices > counts.healthy_choices:
                    addictions += 1

                if debug_episode and seed_idx == 0 and agent_idx == 0 and log_file:
                    log_file.write(f"Final Counts - Drug Choices: {counts.drug_choices}, Healthy Choices: {counts.healthy_choices}\n")
    finally:
        if log_file:
            log_file.close()
            print(f"Debug log (seed=0, agent=0) written to {debug_txt_path}")

    if debug_records:
        if debug_csv_path:
            csv_path = debug_csv_path
        else:
            csv_path = f"debug_episode_beta_{beta:.2f}.csv"
        fieldnames = [
            "phase_name",
            "phase_idx",
            "step",
            "state",
            "action",
            "action_name",
            "reward",
            "next_state",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(debug_records)
        print(f"Debug episode (seed=0, agent=0) written to {csv_path} ({len(debug_records)} steps)")

    return addictions / total_agents


def main():
    parser = argparse.ArgumentParser(description="Fast Hybrid RL Addiction Simulation")
    parser.add_argument("--num-agents", type=int, default=60, help="Agents per seed")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--mb-forget",
        action="store_true",
        help="Reset MB Q-values every planning call instead of gradual decay",
    )
    parser.add_argument(
        "--debug-episode",
        action="store_true",
        help="Print detailed transitions for the first simulated agent",
    )
    parser.add_argument(
        "--debug-csv",
        type=str,
        nargs="?",
        const=None,
        default=None,
        help="Path to write debug episode transitions as CSV (default: ./debug_episode_beta_<beta>.csv)",
    )
    args = parser.parse_args()

    # Beta値の範囲
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rates = []

    print(f"Simulation Start: Agents={args.num_agents}, Runs={args.num_runs}, Seed={args.seed}")
    print("-" * 60)

    def log_progress(beta, seed, agent, n_agents, n_runs, p_idx, step, length):
        # 進捗表示 (間引いて表示)
        if seed == 0 and agent == 0 and step % 100 == 0:
            p_name = PHASES[p_idx][0]
            print(f"[Beta={beta:.1f}] Phase: {p_name} Step: {step}/{length}")

    current_seed = args.seed
    for beta in beta_values:
        # デバッグログのファイル名を生成
        debug_txt_path = None
        if args.debug_episode:
            debug_txt_path = f"debug_log_beta_{beta:.2f}.txt"

        rate = simulate(
            beta,
            args.num_agents,
            args.num_runs,
            current_seed,
            log_progress,
            mb_forget=args.mb_forget,
            debug_episode=args.debug_episode,
            debug_csv_path=args.debug_csv,
            debug_txt_path=debug_txt_path,
        )
        current_seed += args.num_runs  # 次のBetaではシードをずらす
        rates.append(rate * 100)
        print(f"Result: Beta={beta:.1f} => Addiction Rate={rate*100:.2f}%")
        print("-" * 60)

    # グラフ描画
    plt.figure(figsize=(8, 5))
    plt.plot(beta_values, rates, marker="o", linewidth=2, label="No Treatment")
    plt.title(f"Transition to Addiction (N={args.num_agents * args.num_runs})")
    plt.xlabel("Degree of MB Control (Beta)")
    plt.ylabel("Addiction Rate (%)")
    plt.ylim(20, 60)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # 画像保存または表示
    # plt.savefig("addiction_result.png")
    plt.show()

if __name__ == "__main__":
    main()