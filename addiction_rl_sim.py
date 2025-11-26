import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ----- 定数定義: 論文 Table から写した環境仕様 -----
NUM_STATES = 22
NUM_ACTIONS = 9
STATE_GOAL = 0  # corresponds to state 1 in the paper
STATE_START = 3  # corresponds to S0 = 4
STATE_GOAL_ENTRY = 1  # neutral state that leads to the goal
STATE_DRUG = 7  # corresponds to state 8
STATE_NEUTRALS = tuple(range(1, 7))
STATE_NEUTRAL_SET = set(STATE_NEUTRALS)
NEUTRAL_MIN = STATE_NEUTRALS[0]
NEUTRAL_MAX = STATE_NEUTRALS[-1]
STATE_AFTEREFFECTS = tuple(range(8, 22))
STATE_AFTER_SET = set(STATE_AFTEREFFECTS)
AFTER_MIN = STATE_AFTEREFFECTS[0]
AFTER_MAX = STATE_AFTEREFFECTS[-1]
STATE_DRUG_AREA = (STATE_DRUG, *STATE_AFTEREFFECTS)
STATE_DRUG_AREA_SET = set(STATE_DRUG_AREA)
STATE_AW_SPECIAL = 14  # corresponds to paper state 15 (code index)
assert len(STATE_AFTEREFFECTS) == 14, "Aftereffect span must match 14 states (paper states 9-22)."
assert STATE_AW_SPECIAL in STATE_AFTER_SET, "Special aw state must fall inside aftereffect range."
DISCOUNT = 0.9
ALPHA_MF = 0.05
MB_DECAY = 0.01
EPSILON = 0.1
N_PRIORITIZED_SWEEPS = 50
R_G = 1.0
R_P = -4.0
R_SKIP_LONG = -0.3
AFTER_PHASE_REWARDS = (-0.3, -1.2)

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
# Each as^k action aims for a specific neutral state; success depends on distance.

# Table 1: transition probabilities (columns f1,f2）
NEUTRAL_MOVE_SUCCESS = [0.99, 0.99]
NEUTRAL_MOVE_FAIL = [0.01, 0.01]
NEUTRAL_SKIP_SUCCESS = [0.0001, 0.0001]
NEUTRAL_SKIP_FAIL = [0.9999, 0.9999]
AFTER_AG_EXIT = [0.001, 0.001]
AFTER_AS_EXIT = [0.001, 0.001]
AW_FORWARD = [0.4995, 0.4995]
AW_BACKWARD = [0.4995, 0.4995]
AW_STAY = [0.001, 0.001]
AW_SPECIAL_MOVE = [0.2, 0.2]
AW_SPECIAL_EXIT = [0.6, 0.6]
AW_SPECIAL_STAY = [0.2, 0.2]
AD_FORWARD = [0.745, 0.745]
AD_BACKWARD = [0.245, 0.245]
AD_STAY = [0.01, 0.01]


# ----- フェーズ（プレドラッグ→中毒）の台本 -----
PHASES: Tuple[Tuple[str, int, float], ...] = (
    ("pre-drug", 50, 0.0),
    ("addiction", 1000, 10.0),
)
table = np.zeros((len(PHASES), NUM_STATES))
table[:, STATE_GOAL] = R_G
for idx, (_, _, drug_reward) in enumerate(PHASES):
    table[idx, STATE_DRUG] = drug_reward
    table[idx, list(STATE_AFTEREFFECTS)] = AFTER_PHASE_REWARDS[idx]
PHASE_REWARD_TABLE: np.ndarray = table


@dataclass
class PhaseResult:
    drug_choices: int = 0
    healthy_choices: int = 0


class AddictionEnvironment:
    def __init__(self, rng: np.random.Generator):
        # 状態遷移に使う乱数生成器と現在状態を保持
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
            next_state = STATE_START if action == ACTION_GOAL else STATE_GOAL
        else:
            next_state = curr
        reward = self._reward(curr, next_state, phase_idx)
        self.state = next_state
        return next_state, reward

    def _transition_neutral(self, state: int, action: int, phase_idx: int) -> int:
        # 中立エリアの遷移。左右移動・目標到達・ドラッグ突入など
        if action in AS_ACTION_TARGETS:
            target = AS_ACTION_TARGETS[action]
            if target == state:
                return state
            distance = abs(target - state)
            success_prob = (
                NEUTRAL_MOVE_SUCCESS[phase_idx]
                if distance == 1
                else NEUTRAL_SKIP_SUCCESS[phase_idx]
            )
            if self.rng.random() < success_prob:
                return target
            return state
        if action == ACTION_GOAL and state == STATE_GOAL_ENTRY:
            return STATE_GOAL
        if action == ACTION_DRUG and state == NEUTRAL_MAX:
            return STATE_DRUG
        if action == ACTION_AW:
            return state
        return state

    def _transition_aftereffect(self, state: int, action: int, phase_idx: int) -> int:
        # ドラッグ/アフターエフェクト領域内の確率遷移（Table 1 準拠）
        roll = self.rng.random()
        if action == ACTION_GOAL:
            return STATE_START if roll < AFTER_AG_EXIT[phase_idx] else state
        if action in AS_ACTION_TARGETS:
            return STATE_START if roll < AFTER_AS_EXIT[phase_idx] else state
        if action == ACTION_AW:
            if state == STATE_AW_SPECIAL:
                move_prob = AW_SPECIAL_MOVE[phase_idx] / 2.0  # split 20% equally to neighbors
                if roll < move_prob:
                    return max(STATE_DRUG, state - 1)
                roll -= move_prob
                if roll < move_prob:
                    return min(AFTER_MAX, state + 1)
                roll -= move_prob
                if roll < AW_SPECIAL_EXIT[phase_idx]:
                    return STATE_START
                return state
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

    def _reward(self, current_state: int, next_state: int, phase_idx: int) -> float:
        # フェーズごとの報酬表を参照し、後遺症→中立へ出る際の罰則も加算
        reward = PHASE_REWARD_TABLE[phase_idx, next_state]
        if (
            current_state in STATE_NEUTRAL_SET
            and next_state in STATE_NEUTRAL_SET
            and abs(next_state - current_state) > 1
        ):
            reward += R_SKIP_LONG
        if (current_state == STATE_DRUG or current_state in STATE_AFTER_SET) and (
            next_state in STATE_NEUTRAL_SET.union({STATE_GOAL, STATE_GOAL_ENTRY})
        ):
            reward += R_P
        return reward


class HybridAgent:
    def __init__(self, beta: float, rng: np.random.Generator):
        # βで MB/MF の重みを決め、両方の Q テーブルと内部モデルを初期化
        self.beta = beta
        self.rng = rng
        self.q_mf = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.q_mb = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.model_counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
        self.model_rewards = np.zeros((NUM_STATES, NUM_ACTIONS))
        self.model_visits = np.zeros((NUM_STATES, NUM_ACTIONS))

    def select_action(self, state: int) -> int:
        # ε-greedy 方策で行動を決める（βで統合した価値を使用）
        self._plan_q_values()
        if self.rng.random() < EPSILON:
            return int(self.rng.integers(NUM_ACTIONS))
        q_mix = self.beta * self.q_mb[state] + (1.0 - self.beta) * self.q_mf[state]
        best_actions = np.flatnonzero(np.isclose(q_mix, np.max(q_mix)))
        return int(self.rng.choice(best_actions))

    def observe(self, state: int, action: int, reward: float, next_state: int, phase_idx: int):
        # 経験を使って MF Q を更新し、MB 側の内部モデルを学習
        alpha_mf = ALPHA_MF
        td_target = reward + DISCOUNT * np.max(self.q_mf[next_state])
        self.q_mf[state, action] += alpha_mf * (td_target - self.q_mf[state, action])
        self.model_counts[state, action, next_state] += 1
        self.model_rewards[state, action] += reward
        self.model_visits[state, action] += 1

    def _plan_q_values(self):
        # 有界合理性: 各ステップで優先度付きスイーピングを実行後リセット
        self.q_mb.fill(0.0)
        visited_pairs = np.argwhere(self.model_visits > 0)
        if visited_pairs.size == 0:
            return
        best_next = np.zeros(NUM_STATES)
        priorities = []
        for s, a in visited_pairs:
            total = self.model_visits[s, a]
            probs = self.model_counts[s, a] / total
            expected_reward = self.model_rewards[s, a] / total
            target = expected_reward + DISCOUNT * np.dot(probs, best_next)
            diff = abs(target - self.q_mb[s, a])
            priorities.append(diff)
        order = np.argsort(priorities)[::-1]
        updates = min(N_PRIORITIZED_SWEEPS, len(order))
        for idx in order[:updates]:
            s, a = visited_pairs[idx]
            total = self.model_visits[s, a]
            probs = self.model_counts[s, a] / total
            expected_reward = self.model_rewards[s, a] / total
            target = expected_reward + DISCOUNT * np.dot(probs, best_next)
            self.q_mb[s, a] += MB_DECAY * (target - self.q_mb[s, a])
            best_next = np.max(self.q_mb, axis=1)


def simulate(
    beta: float,
    runs: int,
    rng: np.random.Generator,
    progress_cb=None,
) -> float:
    """指定した β で複数エージェントを走らせ、平均的な依存率を返す。"""
    addictions = 0
    for run_idx in range(runs):
        env = AddictionEnvironment(rng)
        agent = HybridAgent(beta, rng)
        state = env.reset()
        counts = PhaseResult()
        for phase_idx, (_, length, _) in enumerate(PHASES):
            report_points = {
                1,
                max(1, length // 2),
                length,
            }
            for step_in_phase in range(length):
                step_number = step_in_phase + 1
                if (
                    progress_cb is not None
                    and step_number in report_points
                ):
                    progress_cb(
                        beta,
                        run_idx,
                        runs,
                        phase_idx,
                        step_number,
                        length,
                    )
                action = agent.select_action(state)
                next_state, reward = env.step(action, phase_idx)
                agent.observe(state, action, reward, next_state, phase_idx)
                if phase_idx == 1:
                    # 中毒フェーズのみ: 報酬ベースでカウント
                    if next_state == STATE_DRUG and reward > 0:
                        counts.drug_choices += 1
                    elif next_state == STATE_GOAL and reward > 0:
                        counts.healthy_choices += 1
                state = next_state
        if counts.drug_choices > counts.healthy_choices:
            addictions += 1
    return addictions / runs


def main():
    # ---- コマンドライン引数の設定 ----
    parser = argparse.ArgumentParser(description="Hybrid MB/MF addiction simulation")
    parser.add_argument(
        "--runs",
        type=int,
        default=60,
        help="Number of agents per beta value",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    runs_per_beta = args.runs
    plt.figure(figsize=(7, 4))
    def log_progress(beta, run_idx, total_runs, phase_idx, step_in_phase, phase_len):
        # 実行状況を標準出力に流す（長時間計算の見える化）
        phase_name = PHASES[phase_idx][0]
        print(
            f"[progress] beta={beta:.1f} run={run_idx + 1}/{total_runs} "
            f"phase={phase_name} step={step_in_phase}/{phase_len}",
            flush=True,
        )

    rates = []
    for beta in beta_values:
        rate = simulate(beta, runs_per_beta, rng, log_progress)
        rates.append(rate * 100)
    for beta, rate in zip(beta_values, rates):
        print(f"beta={beta:.1f} addiction rate={rate:.2f}%")
    plt.plot(beta_values, rates, marker="o", label="No treatment baseline")
    plt.xlabel("β (MB dominance)")
    plt.ylabel("Percentage of agents developing addiction")
    if rates:
        span_min = max(0.0, min(rates) - 5.0)
        span_max = min(100.0, max(rates) + 5.0)
        if span_min >= span_max:
            span_max = span_min + 5.0
        plt.ylim(span_min, span_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
