"""
揮発性環境（Volatility）実験
========================================
報酬の場所（Goal/Drug）や薬物の罰則コストを100ステップごとにランダムに入れ替えて、
固定βと可変β（TD誤差に応じた適応的β学習）の性能差を検証する。

仮説:
- 固定βでは前の環境のMF価値が残ってしまい適応が遅れる
- 可変βが「TD誤差（予測誤差）」に応じて即座にβを引き上げることができれば、
  新しい環境への再適応速度で圧倒できる
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numba import jit

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 定数・環境設定
# ==========================================
NUM_STATES = 22
NUM_ACTIONS = 9

# 状態の初期定義（環境変化により変わる可能性あり）
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

STATE_DRUG_AFTEREFFECT_SET = {STATE_DRUG} | STATE_AFTER_SET

# アフターエフェクト区間でのループ
def aftereffect_forward_state(state: int) -> int:
    if state >= AFTER_MAX:
        return STATE_DRUG
    return state + 1

def aftereffect_backward_state(state: int) -> int:
    if state <= STATE_DRUG:
        return AFTER_MAX
    return state - 1

# ハイパーパラメータ
DISCOUNT = 0.9
ALPHA_MF = 0.05
MODEL_DECAY = 0.01
INITIAL_TRANSITION_COUNT = 5.0
EPSILON = 0.1
N_PRIORITIZED_SWEEPS = 50
T_MB = 1.0

# β学習用パラメータ
BETA_VALUES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
ALPHA_BETA = 0.1
EPSILON_BETA = 0.1

# 報酬設定
R_G = 1.0            # Goal reward
R_P = -4.0           # Punishment (shock)
R_SKIP_LONG = -0.3

# 揮発性環境用のパラメータ
VOLATILITY_INTERVAL = 100  # 環境変化の間隔（ステップ数）

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
    ACTION_AS2: "as2", ACTION_AS3: "as3", ACTION_AS4: "as4",
    ACTION_AS5: "as5", ACTION_AS6: "as6", ACTION_AS7: "as7",
    ACTION_GOAL: "ag", ACTION_DRUG: "ad", ACTION_AW: "aw",
}

# 遷移確率テーブル
NEUTRAL_MOVE_SUCCESS = [0.99, 0.99]
NEUTRAL_SKIP_SUCCESS = [0.0001, 0.0001]
AFTER_AG_EXIT = [0.001, 0.001]
AFTER_AS_EXIT = [0.001, 0.001]
AW_FORWARD = [0.4995, 0.4995]
AW_BACKWARD = [0.4995, 0.4995]
AW_SPECIAL_MOVE = [0.4, 0.4]
AW_SPECIAL_EXIT = [0.6, 0.6]
AD_FORWARD = [0.745, 0.745]
AD_BACKWARD = [0.245, 0.245]


# ==========================================
# 2. Numba 高速化カーネル
# ==========================================
@jit(nopython=True, cache=False)
def run_prioritized_sweeping(
    q_mb, model_counts, model_rewards,
    num_states, num_actions, n_sweeps, t_mb, discount, rand_vals
):
    H = np.zeros(num_states, dtype=np.float64)
    V = np.zeros(num_states, dtype=np.float64)
    q_temp = np.zeros(num_actions, dtype=np.float64)
    probs = np.zeros(num_states, dtype=np.float64)
    exp_vals = np.zeros(num_states, dtype=np.float64)

    steps = 0
    while steps < n_sweeps:
        inv_t = 1.0 / t_mb
        sum_exp = 0.0
        for i in range(num_states):
            val = math.exp(H[i] * inv_t)
            exp_vals[i] = val
            sum_exp += val

        if sum_exp <= 0.0:
            for i in range(num_states):
                probs[i] = 1.0 / num_states
        else:
            for i in range(num_states):
                probs[i] = exp_vals[i] / sum_exp

        r = rand_vals[steps]
        cum = 0.0
        s_tilde = 0
        for i in range(num_states):
            cum += probs[i]
            if r < cum:
                s_tilde = i
                break

        for a in range(num_actions):
            q_val = 0.0
            denom = 0.0
            for sp in range(num_states):
                denom += model_counts[s_tilde, a, sp]

            if denom <= 0.0:
                q_temp[a] = 0.0
                continue

            for sp in range(num_states):
                cnt = model_counts[s_tilde, a, sp]
                if cnt <= 0.0:
                    continue
                p = cnt / denom
                r_avg = model_rewards[s_tilde, a, sp] / cnt
                q_val += p * (r_avg + discount * V[sp])
            q_temp[a] = q_val

        for a in range(num_actions):
            q_mb[s_tilde, a] = q_temp[a]

        M = q_temp[0]
        for a in range(1, num_actions):
            if q_temp[a] > M:
                M = q_temp[a]

        delta = V[s_tilde] - M
        if delta < 0.0:
            delta = -delta
        V[s_tilde] = M

        h = np.zeros(num_states, dtype=np.float64)
        for s in range(num_states):
            max_p = 0.0
            for a in range(num_actions):
                denom_sa = 0.0
                for sp in range(num_states):
                    denom_sa += model_counts[s, a, sp]
                if denom_sa <= 0.0:
                    continue
                p_s_to_st = model_counts[s, a, s_tilde] / denom_sa
                if p_s_to_st > max_p:
                    max_p = p_s_to_st
            h[s] = delta * max_p

        H[s_tilde] = h[s_tilde]
        for s in range(num_states):
            if s == s_tilde:
                continue
            if h[s] > H[s]:
                H[s] = h[s]

        steps += 1


# ==========================================
# 3. 揮発性環境クラス
# ==========================================
@dataclass
class VolatileEnvironmentConfig:
    """揮発性環境の設定を保持するクラス"""
    goal_reward: float = 1.0         # Goal報酬（変動可能）
    drug_reward: float = 10.0        # 薬物報酬（変動可能）
    aftereffect_penalty: float = -1.2  # アフターエフェクト区間のペナルティ（変動可能）
    shock_penalty: float = -4.0      # ショックペナルティ（変動可能）
    # どの報酬が「良い」選択かを示すフラグ（報酬の符号が入れ替わった場合）
    goal_is_better: bool = True


class VolatileAddictionEnvironment:
    """揮発性を持つ依存症環境"""
    def __init__(self, rng: np.random.Generator, volatility_interval: int = VOLATILITY_INTERVAL):
        self.state = STATE_START
        self.rng = rng
        self.volatility_interval = volatility_interval
        self.step_count = 0
        self.config_history: List[VolatileEnvironmentConfig] = []
        
        # 初期設定
        self.config = VolatileEnvironmentConfig()
        self.config_history.append(VolatileEnvironmentConfig())
        
    def reset(self) -> int:
        self.state = STATE_START
        return self.state
    
    def maybe_change_environment(self):
        """100ステップごとに環境をランダムに変化させる"""
        if self.step_count > 0 and self.step_count % self.volatility_interval == 0:
            # 環境変化のタイプをランダム選択
            change_type = self.rng.choice(['swap_rewards', 'vary_penalties', 'mixed'])
            
            if change_type == 'swap_rewards':
                # Goal報酬と薬物報酬の相対的な魅力度を入れ替え
                old_goal = self.config.goal_reward
                old_drug = self.config.drug_reward
                
                # ランダムな新しい報酬値を生成
                self.config.goal_reward = self.rng.uniform(0.5, 2.0)
                self.config.drug_reward = self.rng.uniform(5.0, 15.0)
                
                # 50%の確率でどちらが「良い」選択かを入れ替え
                if self.rng.random() < 0.5:
                    self.config.goal_is_better = not self.config.goal_is_better
                    
            elif change_type == 'vary_penalties':
                # ペナルティの強度を変更
                self.config.aftereffect_penalty = self.rng.uniform(-2.0, -0.5)
                self.config.shock_penalty = self.rng.uniform(-6.0, -2.0)
                
            else:  # mixed
                # すべてをランダムに変更
                self.config.goal_reward = self.rng.uniform(0.5, 2.0)
                self.config.drug_reward = self.rng.uniform(5.0, 15.0)
                self.config.aftereffect_penalty = self.rng.uniform(-2.0, -0.5)
                self.config.shock_penalty = self.rng.uniform(-6.0, -2.0)
                if self.rng.random() < 0.5:
                    self.config.goal_is_better = not self.config.goal_is_better
            
            self.config_history.append(VolatileEnvironmentConfig(
                goal_reward=self.config.goal_reward,
                drug_reward=self.config.drug_reward,
                aftereffect_penalty=self.config.aftereffect_penalty,
                shock_penalty=self.config.shock_penalty,
                goal_is_better=self.config.goal_is_better
            ))
            return True
        return False

    def step(self, action: int, phase_idx: int = 1) -> Tuple[int, float, bool]:
        """環境のステップ実行（環境変化フラグも返す）"""
        self.step_count += 1
        env_changed = self.maybe_change_environment()
        
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

        reward = self._reward(curr, action, next_state, phase_idx)
        self.state = next_state
        return next_state, reward, env_changed

    def _transition_neutral(self, state: int, action: int, phase_idx: int) -> int:
        if action in AS_ACTION_TARGETS:
            target = AS_ACTION_TARGETS[action]
            if target == state:
                return state
            
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
        
        if action == ACTION_GOAL:
            return STATE_START if roll < AFTER_AG_EXIT[phase_idx] else state
        if action in AS_ACTION_TARGETS:
            return STATE_START if roll < AFTER_AS_EXIT[phase_idx] else state
            
        if action == ACTION_AW:
            if state == STATE_AW_SPECIAL:
                p_move_each = 0.2
                if roll < p_move_each:
                    return max(STATE_DRUG, state - 1)
                roll -= p_move_each
                if roll < p_move_each:
                    return min(AFTER_MAX, state + 1)
                return STATE_START
            
            if roll < AW_FORWARD[phase_idx]:
                return aftereffect_forward_state(state)
            roll -= AW_FORWARD[phase_idx]
            if roll < AW_BACKWARD[phase_idx]:
                return aftereffect_backward_state(state)
            return STATE_START

        if action == ACTION_DRUG:
            if roll < AD_FORWARD[phase_idx]:
                return aftereffect_forward_state(state)
            roll -= AD_FORWARD[phase_idx]
            if roll < AD_BACKWARD[phase_idx]:
                return aftereffect_backward_state(state)
            return STATE_START
            
        return state

    def _reward(self, current_state: int, action: int, next_state: int, phase_idx: int) -> float:
        r = 0.0
        
        # Goal報酬（現在の設定値を使用）
        if current_state == STATE_GOAL and action == ACTION_GOAL and next_state == STATE_START:
            r += self.config.goal_reward
        # 薬物報酬（現在の設定値を使用）
        elif current_state == NEUTRAL_MAX and action == ACTION_DRUG and next_state == STATE_DRUG:
            r += self.config.drug_reward
        
        # Neutralエリアでのロングジャンプコスト
        if (current_state in STATE_NEUTRAL_SET and next_state in STATE_NEUTRAL_SET 
            and abs(next_state - current_state) > 1):
            r += R_SKIP_LONG
        
        # アフターエフェクト区間での報酬設計（現在の設定値を使用）
        if current_state in STATE_DRUG_AFTEREFFECT_SET:
            if next_state == STATE_START:
                r += self.config.shock_penalty
            elif next_state in STATE_DRUG_AFTEREFFECT_SET:
                r += self.config.aftereffect_penalty
            
        return r


# ==========================================
# 4. 適応的β学習エージェント（TD誤差に応じた調整）
# ==========================================
class AdaptiveBetaAgent:
    """TD誤差に応じてβを適応的に調整するエージェント"""
    def __init__(self, rng: np.random.Generator, fixed_beta: Optional[float] = None):
        self.rng = rng
        self.fixed_beta = fixed_beta
        
        # Qテーブル
        self.q_mf = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.q_mb = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        
        # β学習用
        self.q_beta = np.zeros(len(BETA_VALUES), dtype=np.float64)
        self.current_beta_idx = 0
        self.prev_beta_idx = None
        
        # TD誤差の履歴（適応的β調整用）
        self.td_error_history: List[float] = []
        self.recent_td_errors: List[float] = []  # 直近のTD誤差
        self.td_error_window = 10  # TD誤差の移動平均ウィンドウ
        
        # 環境変化検出用
        self.env_change_detected = False
        self.td_error_threshold = 2.0  # TD誤差がこれを超えると環境変化と判断
        
        if self.fixed_beta is not None:
            self.current_beta_idx = np.argmin(np.abs(BETA_VALUES - self.fixed_beta))
        
        # メンタルモデル
        self.model_counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=np.float64)
        self.model_rewards = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=np.float64)
        self.model_visits = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.model_observed = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=bool)
        
        # 初期モデル
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                self.model_counts[s, a, s] = 1.0
                self.model_visits[s, a] = 1.0
                self.model_observed[s, a, s] = True

    def select_beta(self) -> int:
        """β値を選択（TD誤差に応じた適応的調整を含む）"""
        if self.fixed_beta is not None:
            return self.current_beta_idx
        
        # TD誤差が大きい場合、MB（高いβ）を優先
        if len(self.recent_td_errors) >= 3:
            avg_td_error = np.mean(np.abs(self.recent_td_errors[-3:]))
            if avg_td_error > self.td_error_threshold:
                # 環境変化を検出 -> 高いβ（MB重視）を選択しやすくする
                self.env_change_detected = True
                # Q_betaを高いβ側にバイアス
                biased_q = self.q_beta.copy()
                for i, beta in enumerate(BETA_VALUES):
                    biased_q[i] += beta * avg_td_error * 0.5
                max_val = np.max(biased_q)
                best_betas = np.flatnonzero(np.isclose(biased_q, max_val, rtol=1e-08, atol=1e-12))
                return int(self.rng.choice(best_betas))
        
        # 通常のε-greedy選択
        if self.rng.random() < EPSILON_BETA:
            return int(self.rng.integers(len(BETA_VALUES)))
        
        max_val = np.max(self.q_beta)
        best_betas = np.flatnonzero(np.isclose(self.q_beta, max_val, rtol=1e-08, atol=1e-12))
        return int(self.rng.choice(best_betas))
    
    def select_action(self, state: int) -> int:
        self.current_beta_idx = self.select_beta()
        current_beta = BETA_VALUES[self.current_beta_idx]
        
        self._plan_q_values()
        
        if self.rng.random() < EPSILON:
            return int(self.rng.integers(NUM_ACTIONS))
            
        q_mix = current_beta * self.q_mb[state] + (1.0 - current_beta) * self.q_mf[state]
        max_val = np.max(q_mix)
        best_actions = np.flatnonzero(np.isclose(q_mix, max_val, rtol=1e-08, atol=1e-12))
        return int(self.rng.choice(best_actions))

    def observe(self, state: int, action: int, reward: float, next_state: int):
        # TD誤差計算
        td_target = reward + DISCOUNT * np.max(self.q_mf[next_state])
        td_error = td_target - self.q_mf[state, action]
        
        # TD誤差履歴に追加
        self.td_error_history.append(td_error)
        self.recent_td_errors.append(td_error)
        if len(self.recent_td_errors) > self.td_error_window:
            self.recent_td_errors.pop(0)
        
        # β学習の更新
        if self.fixed_beta is None and self.prev_beta_idx is not None:
            td_target_beta = reward + DISCOUNT * np.max(self.q_beta)
            self.q_beta[self.prev_beta_idx] += ALPHA_BETA * (
                td_target_beta - self.q_beta[self.prev_beta_idx]
            )
        
        if self.fixed_beta is None:
            self.prev_beta_idx = self.current_beta_idx
        
        # MF更新
        self.q_mf[state, action] += ALPHA_MF * td_error
        
        # モデル更新
        self.model_counts *= (1.0 - MODEL_DECAY)
        self.model_rewards *= (1.0 - MODEL_DECAY)
        self.model_visits *= (1.0 - MODEL_DECAY)

        if not self.model_observed[state, action, next_state]:
            self.model_counts[state, action, next_state] = INITIAL_TRANSITION_COUNT
            self.model_rewards[state, action, next_state] = reward * INITIAL_TRANSITION_COUNT
            self.model_observed[state, action, next_state] = True
            self.model_visits[state, action] += INITIAL_TRANSITION_COUNT
        else:
            self.model_counts[state, action, next_state] += 1.0
            self.model_rewards[state, action, next_state] += reward
            self.model_visits[state, action] += 1.0

    def _plan_q_values(self):
        rand_vals = self.rng.random(N_PRIORITIZED_SWEEPS)
        run_prioritized_sweeping(
            self.q_mb, self.model_counts, self.model_rewards,
            NUM_STATES, NUM_ACTIONS, N_PRIORITIZED_SWEEPS,
            T_MB, DISCOUNT, rand_vals
        )
    
    def get_current_beta(self) -> float:
        return BETA_VALUES[self.current_beta_idx]


# ==========================================
# 5. 実験実行
# ==========================================
@dataclass
class VolatilityExperimentResult:
    """揮発性環境実験の結果"""
    beta_type: str  # "fixed_0.0", "fixed_0.5", "fixed_1.0", "adaptive"
    total_reward: float
    avg_reward_per_step: float
    rewards_per_window: List[float]  # 各環境変化区間での平均報酬
    adaptation_speed: List[float]  # 各環境変化後の適応速度
    beta_history: List[float]  # β値の履歴
    td_error_history: List[float]  # TD誤差の履歴
    env_change_points: List[int]  # 環境変化が起きたステップ
    optimal_choice_rate: List[float]  # 各区間での最適選択率


def run_volatility_experiment(
    num_agents: int = 100,
    num_steps: int = 2000,
    base_seed: int = 42,
    volatility_interval: int = VOLATILITY_INTERVAL,
) -> Dict[str, VolatilityExperimentResult]:
    """揮発性環境での実験を実行"""
    
    results = {}
    beta_configs = [
        ("fixed_0.0", 0.0),
        ("fixed_0.5", 0.5),
        ("fixed_1.0", 1.0),
        ("adaptive", None),
    ]
    
    for beta_name, fixed_beta in beta_configs:
        print(f"\n{'='*50}")
        print(f"Running experiment: {beta_name}")
        print(f"{'='*50}")
        
        all_rewards = []
        all_rewards_per_window = []
        all_adaptation_speeds = []
        all_beta_histories = []
        all_td_errors = []
        all_env_changes = []
        all_optimal_rates = []
        
        for agent_idx in range(num_agents):
            if (agent_idx + 1) % 20 == 0:
                print(f"  Agent {agent_idx + 1}/{num_agents}")
            
            seed = base_seed + agent_idx
            rng = np.random.default_rng(seed)
            
            env = VolatileAddictionEnvironment(rng, volatility_interval)
            agent = AdaptiveBetaAgent(rng, fixed_beta=fixed_beta)
            
            state = env.reset()
            
            rewards = []
            beta_history = []
            env_change_points = []
            window_rewards = []
            current_window_rewards = []
            
            for step in range(num_steps):
                action = agent.select_action(state)
                beta_history.append(agent.get_current_beta())
                
                next_state, reward, env_changed = env.step(action)
                agent.observe(state, action, reward, next_state)
                
                rewards.append(reward)
                current_window_rewards.append(reward)
                
                if env_changed:
                    env_change_points.append(step)
                    window_rewards.append(np.mean(current_window_rewards))
                    current_window_rewards = []
                
                state = next_state
            
            # 最後のウィンドウを追加
            if current_window_rewards:
                window_rewards.append(np.mean(current_window_rewards))
            
            # 適応速度を計算（環境変化後10ステップでの報酬回復率）
            adaptation_speeds = []
            for change_point in env_change_points:
                if change_point + 20 < num_steps:
                    pre_change = np.mean(rewards[max(0, change_point-10):change_point])
                    post_change_early = np.mean(rewards[change_point:change_point+10])
                    post_change_late = np.mean(rewards[change_point+10:change_point+20])
                    if abs(pre_change - post_change_early) > 0.01:
                        speed = (post_change_late - post_change_early) / abs(pre_change - post_change_early + 0.01)
                        adaptation_speeds.append(speed)
            
            all_rewards.append(sum(rewards))
            all_rewards_per_window.append(window_rewards)
            all_adaptation_speeds.append(adaptation_speeds)
            all_beta_histories.append(beta_history)
            all_td_errors.append(agent.td_error_history)
            all_env_changes.append(env_change_points)
        
        # 結果を集計
        avg_rewards_per_window = []
        max_windows = max(len(w) for w in all_rewards_per_window)
        for i in range(max_windows):
            window_vals = [w[i] for w in all_rewards_per_window if len(w) > i]
            avg_rewards_per_window.append(np.mean(window_vals))
        
        avg_adaptation_speed = []
        max_changes = max(len(a) for a in all_adaptation_speeds)
        for i in range(max_changes):
            speed_vals = [a[i] for a in all_adaptation_speeds if len(a) > i]
            if speed_vals:
                avg_adaptation_speed.append(np.mean(speed_vals))
        
        results[beta_name] = VolatilityExperimentResult(
            beta_type=beta_name,
            total_reward=np.mean(all_rewards),
            avg_reward_per_step=np.mean(all_rewards) / num_steps,
            rewards_per_window=avg_rewards_per_window,
            adaptation_speed=avg_adaptation_speed,
            beta_history=all_beta_histories[0] if all_beta_histories else [],
            td_error_history=all_td_errors[0] if all_td_errors else [],
            env_change_points=all_env_changes[0] if all_env_changes else [],
            optimal_choice_rate=[],
        )
        
        print(f"  Total reward: {results[beta_name].total_reward:.2f}")
        print(f"  Avg reward/step: {results[beta_name].avg_reward_per_step:.4f}")
    
    return results


# ==========================================
# 6. 可視化
# ==========================================
def plot_volatility_results(results: Dict[str, VolatilityExperimentResult], output_prefix: str = "volatility"):
    """揮発性環境実験の結果を可視化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {
        "fixed_0.0": "#e74c3c",  # 赤（MF重視）
        "fixed_0.5": "#f39c12",  # オレンジ
        "fixed_1.0": "#3498db",  # 青（MB重視）
        "adaptive": "#2ecc71",   # 緑（適応的）
    }
    
    labels = {
        "fixed_0.0": "固定β=0.0 (MF)",
        "fixed_0.5": "固定β=0.5",
        "fixed_1.0": "固定β=1.0 (MB)",
        "adaptive": "可変β（適応的）",
    }
    
    # (1) 累積報酬の比較
    ax1 = axes[0, 0]
    beta_types = list(results.keys())
    total_rewards = [results[bt].total_reward for bt in beta_types]
    bars = ax1.bar(range(len(beta_types)), total_rewards, 
                   color=[colors[bt] for bt in beta_types])
    ax1.set_xticks(range(len(beta_types)))
    ax1.set_xticklabels([labels[bt] for bt in beta_types], rotation=15, ha='right')
    ax1.set_ylabel('累積報酬', fontsize=12)
    ax1.set_title('累積報酬の比較', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 値を表示
    for bar, val in zip(bars, total_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # (2) 環境変化区間ごとの報酬推移
    ax2 = axes[0, 1]
    for beta_type in results:
        rewards_per_window = results[beta_type].rewards_per_window
        if rewards_per_window:
            ax2.plot(range(1, len(rewards_per_window)+1), rewards_per_window, 
                    'o-', label=labels[beta_type], color=colors[beta_type], linewidth=2)
    ax2.set_xlabel('環境変化区間', fontsize=12)
    ax2.set_ylabel('平均報酬', fontsize=12)
    ax2.set_title('各環境変化区間での平均報酬', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # (3) 適応速度の比較
    ax3 = axes[0, 2]
    adaptation_data = []
    adaptation_labels = []
    for beta_type in results:
        speeds = results[beta_type].adaptation_speed
        if speeds:
            adaptation_data.append(speeds)
            adaptation_labels.append(labels[beta_type])
    
    if adaptation_data:
        bp = ax3.boxplot(adaptation_data, labels=adaptation_labels, patch_artist=True)
        for patch, beta_type in zip(bp['boxes'], results.keys()):
            patch.set_facecolor(colors[beta_type])
            patch.set_alpha(0.7)
        ax3.set_ylabel('適応速度', fontsize=12)
        ax3.set_title('環境変化後の適応速度', fontsize=14, fontweight='bold')
        ax3.set_xticklabels(adaptation_labels, rotation=15, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # (4) β値の時系列（適応的エージェント）
    ax4 = axes[1, 0]
    adaptive_result = results.get("adaptive")
    if adaptive_result and adaptive_result.beta_history:
        beta_hist = adaptive_result.beta_history
        ax4.plot(beta_hist, 'g-', alpha=0.7, linewidth=1)
        
        # 環境変化点を縦線で表示
        for cp in adaptive_result.env_change_points:
            ax4.axvline(cp, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax4.set_xlabel('ステップ', fontsize=12)
        ax4.set_ylabel('選択されたβ値', fontsize=12)
        ax4.set_title('適応的β値の時系列変化', fontsize=14, fontweight='bold')
        ax4.set_ylim(-0.05, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # 凡例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='g', linewidth=2, label='β値'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='環境変化点')
        ]
        ax4.legend(handles=legend_elements, loc='upper right')
    
    # (5) TD誤差の時系列
    ax5 = axes[1, 1]
    if adaptive_result and adaptive_result.td_error_history:
        td_errors = adaptive_result.td_error_history
        # 移動平均を計算
        window = 20
        if len(td_errors) > window:
            moving_avg = np.convolve(np.abs(td_errors), np.ones(window)/window, mode='valid')
            ax5.plot(range(window-1, len(td_errors)), moving_avg, 'b-', linewidth=1, alpha=0.8)
        
        # 環境変化点を縦線で表示
        for cp in adaptive_result.env_change_points:
            ax5.axvline(cp, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax5.set_xlabel('ステップ', fontsize=12)
        ax5.set_ylabel('|TD誤差| (移動平均)', fontsize=12)
        ax5.set_title('TD誤差の時系列変化（環境変化検出の指標）', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # (6) 環境変化直後の報酬回復曲線
    ax6 = axes[1, 2]
    # 環境変化後20ステップの報酬推移を比較
    # これは詳細な分析が必要なため、シンプルな説明を追加
    ax6.text(0.5, 0.5, '環境変化後の\n報酬回復パターン\n\n（詳細分析は\n別途実施）', 
             ha='center', va='center', fontsize=14, transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('環境変化後の回復パターン', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_prefix}_comparison.png")
    plt.close()
    
    # 追加のサマリーグラフ
    plot_volatility_summary(results, output_prefix)


def plot_volatility_summary(results: Dict[str, VolatilityExperimentResult], output_prefix: str):
    """揮発性環境実験のサマリーグラフ"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        "fixed_0.0": "#e74c3c",
        "fixed_0.5": "#f39c12",
        "fixed_1.0": "#3498db",
        "adaptive": "#2ecc71",
    }
    
    labels = {
        "fixed_0.0": "固定β=0.0\n(MF重視)",
        "fixed_0.5": "固定β=0.5\n(中間)",
        "fixed_1.0": "固定β=1.0\n(MB重視)",
        "adaptive": "可変β\n(適応的)",
    }
    
    # (1) 総合性能比較（レーダーチャート風の棒グラフ）
    ax1 = axes[0]
    metrics = ['累積報酬', '適応速度']
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, beta_type in enumerate(results):
        result = results[beta_type]
        avg_adaptation = np.mean(result.adaptation_speed) if result.adaptation_speed else 0
        values = [
            result.total_reward / max(r.total_reward for r in results.values()),  # 正規化
            (avg_adaptation + 1) / 2  # -1〜1を0〜1に正規化
        ]
        ax1.bar(x + i * width, values, width, label=labels[beta_type], color=colors[beta_type], alpha=0.7)
    
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel('正規化スコア', fontsize=12)
    ax1.set_title('総合性能比較（揮発性環境）', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # (2) 結論サマリー
    ax2 = axes[1]
    ax2.axis('off')
    
    # 結果のサマリーテキストを作成
    sorted_results = sorted(results.items(), key=lambda x: x[1].total_reward, reverse=True)
    
    summary_text = "【実験結果サマリー】\n\n"
    summary_text += "■ 揮発性環境（100ステップごとに変化）での性能比較\n\n"
    
    for i, (beta_type, result) in enumerate(sorted_results):
        rank = i + 1
        avg_speed = np.mean(result.adaptation_speed) if result.adaptation_speed else 0
        summary_text += f"{rank}位: {labels[beta_type].replace(chr(10), ' ')}\n"
        summary_text += f"   累積報酬: {result.total_reward:.2f}\n"
        summary_text += f"   平均適応速度: {avg_speed:.3f}\n\n"
    
    # 可変βが最高か確認
    adaptive_rank = next(i for i, (bt, _) in enumerate(sorted_results) if bt == "adaptive") + 1
    
    if adaptive_rank == 1:
        summary_text += "✓ 可変β（適応的）が最高性能を達成！\n"
        summary_text += "  → TD誤差に基づく適応が揮発性環境で有効"
    else:
        summary_text += f"※ 可変β（適応的）は{adaptive_rank}位\n"
        summary_text += "  → パラメータ調整で改善の余地あり"
    
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_summary.png")
    plt.close()


# ==========================================
# 7. メイン実行
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='揮発性環境での可変β vs 固定β実験')
    parser.add_argument('--num-agents', type=int, default=100, help='エージェント数')
    parser.add_argument('--num-steps', type=int, default=2000, help='ステップ数')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--volatility-interval', type=int, default=100, help='環境変化間隔')
    parser.add_argument('--output-prefix', type=str, default='volatility_experiment', help='出力ファイル名のプレフィックス')
    
    args = parser.parse_args()
    
    print("="*60)
    print("揮発性環境（Volatility）実験")
    print("="*60)
    print(f"エージェント数: {args.num_agents}")
    print(f"ステップ数: {args.num_steps}")
    print(f"環境変化間隔: {args.volatility_interval}ステップ")
    print(f"乱数シード: {args.seed}")
    print("="*60)
    
    # 実験実行
    results = run_volatility_experiment(
        num_agents=args.num_agents,
        num_steps=args.num_steps,
        base_seed=args.seed,
        volatility_interval=args.volatility_interval,
    )
    
    # 結果を可視化
    plot_volatility_results(results, args.output_prefix)
    
    # 結果をJSONで保存
    results_dict = {}
    for beta_type, result in results.items():
        results_dict[beta_type] = {
            'total_reward': result.total_reward,
            'avg_reward_per_step': result.avg_reward_per_step,
            'rewards_per_window': result.rewards_per_window,
            'adaptation_speed': result.adaptation_speed,
            'env_change_points': result.env_change_points,
        }
    
    with open(f"{args.output_prefix}_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output_prefix}_results.json")
    
    print("\n" + "="*60)
    print("実験完了!")
    print("="*60)


if __name__ == "__main__":
    main()
