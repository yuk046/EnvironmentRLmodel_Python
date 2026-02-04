"""
Large-Scale Volatility Experiment for Publication
===================================================
Comprehensive comparison of adaptive β vs fixed β strategies
in volatile environments with rigorous statistical analysis.

実験条件:
- エージェント数: 200 per condition
- 固定β: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- 適応β: TD-error based adaptation
- フェーズ: 50 (安全学習) → 1000 (高報酬・高リスク) → 8000 (動的ボラティリティ)
- 環境変化頻度: [200, 500, 1000 steps] in dynamic phase
- 総ステップ数: 9050 steps (デフォルト)
- 統計分析: t-test, effect size, confidence intervals
"""

import argparse
import json
import pickle
import math
import time
import gc
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from numba import jit
import warnings
warnings.filterwarnings('ignore')

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")

# ==========================================
# Memory Monitoring
# ==========================================
def get_memory_usage():
    """Get current memory usage in MB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0

def print_memory_status(prefix=""):
    """Print current memory usage"""
    if PSUTIL_AVAILABLE:
        mem_mb = get_memory_usage()
        print(f"{prefix}Memory usage: {mem_mb:.1f} MB")

# ==========================================
# Constants and Environment Setup
# ==========================================
NUM_STATES = 22
NUM_ACTIONS = 9

STATE_GOAL = 0
STATE_GOAL_ENTRY = 1
STATE_START = 3
STATE_DRUG = 7
STATE_AW_SPECIAL = 14

STATE_NEUTRALS = tuple(range(1, 7))
STATE_NEUTRAL_SET = set(STATE_NEUTRALS)
NEUTRAL_MAX = STATE_NEUTRALS[-1]

STATE_AFTEREFFECTS = tuple(range(8, 22))
STATE_AFTER_SET = set(STATE_AFTEREFFECTS)
AFTER_MAX = STATE_AFTEREFFECTS[-1]

STATE_DRUG_AFTEREFFECT_SET = {STATE_DRUG} | STATE_AFTER_SET

def aftereffect_forward_state(state: int) -> int:
    if state >= AFTER_MAX:
        return STATE_DRUG
    return state + 1

def aftereffect_backward_state(state: int) -> int:
    if state <= STATE_DRUG:
        return AFTER_MAX
    return state - 1

# Hyperparameters
DISCOUNT = 0.9
ALPHA_MF = 0.05
MODEL_DECAY = 0.01
INITIAL_TRANSITION_COUNT = 5.0
EPSILON = 0.1
N_PRIORITIZED_SWEEPS = 50
T_MB = 1.0

# Beta learning parameters
BETA_VALUES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
ALPHA_BETA = 0.1
EPSILON_BETA = 0.1

# Reward settings
R_G = 1.0
R_P = -4.0
R_SKIP_LONG = -0.3

# Phase schedule (safe pre-training → risky high-reward → volatile dynamic)
PHASE_STEPS = [50, 1000, 8000]
TOTAL_PHASE_STEPS = sum(PHASE_STEPS)

# Actions
ACTION_AS2, ACTION_AS3, ACTION_AS4 = 0, 1, 2
ACTION_AS5, ACTION_AS6, ACTION_AS7 = 3, 4, 5
ACTION_GOAL, ACTION_DRUG, ACTION_AW = 6, 7, 8

AS_ACTION_TARGETS = {
    ACTION_AS2: 1, ACTION_AS3: 2, ACTION_AS4: 3,
    ACTION_AS5: 4, ACTION_AS6: 5, ACTION_AS7: 6,
}

# Transition probabilities
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
# Numba JIT Functions
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
# Environment and Agent Classes
# ==========================================
@dataclass
class VolatileEnvironmentConfig:
    goal_reward: float = 1.0
    drug_reward: float = 10.0
    aftereffect_penalty: float = -1.2
    shock_penalty: float = -4.0
    goal_is_better: bool = True


class VolatileAddictionEnvironment:
    def __init__(self, rng: np.random.Generator, volatility_interval: int = 100):
        self.state = STATE_START
        self.rng = rng
        self.volatility_interval = volatility_interval
        self.step_count = 0
        self.config_history: List[VolatileEnvironmentConfig] = []
        self.config = VolatileEnvironmentConfig()
        self.config_history.append(VolatileEnvironmentConfig())
        
    def reset(self) -> int:
        self.state = STATE_START
        return self.state
    
    def maybe_change_environment(self, allow_volatility: bool = True):
        if not allow_volatility:
            return False
        if self.step_count > 0 and self.step_count % self.volatility_interval == 0:
            change_type = self.rng.choice([
                'swap_rewards',        # mild reshuffle
                'vary_penalties',      # heavier punishment variance
                'mixed',               # both rewards and penalties shift
                'punish_drug'          # make drug temporarily bad
            ])
            
            if change_type == 'swap_rewards':
                self.config.goal_reward = self.rng.uniform(0.8, 3.0)
                self.config.drug_reward = self.rng.uniform(5.0, 18.0)
                # occasional inversion: drug becomes unattractive
                if self.rng.random() < 0.3:
                    self.config.drug_reward = self.rng.uniform(-8.0, -2.0)
                
            elif change_type == 'vary_penalties':
                self.config.aftereffect_penalty = self.rng.uniform(-3.0, -0.3)
                self.config.shock_penalty = self.rng.uniform(-10.0, -3.0)
                
            elif change_type == 'punish_drug':
                # Strongly discourage habitual drug-seeking; goal modest
                self.config.goal_reward = self.rng.uniform(1.0, 3.0)
                self.config.drug_reward = self.rng.uniform(-10.0, -3.0)
                self.config.aftereffect_penalty = self.rng.uniform(-3.5, -0.8)
                self.config.shock_penalty = self.rng.uniform(-10.0, -4.0)
                
            else:  # mixed
                self.config.goal_reward = self.rng.uniform(0.8, 3.0)
                self.config.drug_reward = self.rng.uniform(5.0, 18.0)
                if self.rng.random() < 0.5:
                    self.config.drug_reward = self.rng.uniform(-9.0, -2.5)
                self.config.aftereffect_penalty = self.rng.uniform(-3.0, -0.3)
                self.config.shock_penalty = self.rng.uniform(-10.0, -3.0)
            
            self.config_history.append(VolatileEnvironmentConfig(
                goal_reward=self.config.goal_reward,
                drug_reward=self.config.drug_reward,
                aftereffect_penalty=self.config.aftereffect_penalty,
                shock_penalty=self.config.shock_penalty,
                goal_is_better=self.config.goal_is_better
            ))
            return True
        return False

    def step(self, action: int, phase_idx: int = 1, allow_volatility: bool = True) -> Tuple[int, float, bool]:
        self.step_count += 1
        env_changed = self.maybe_change_environment(allow_volatility)
        
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
        
        if current_state == STATE_GOAL and action == ACTION_GOAL and next_state == STATE_START:
            r += self.config.goal_reward
        elif current_state == NEUTRAL_MAX and action == ACTION_DRUG and next_state == STATE_DRUG:
            r += self.config.drug_reward
        
        if (current_state in STATE_NEUTRAL_SET and next_state in STATE_NEUTRAL_SET 
            and abs(next_state - current_state) > 1):
            r += R_SKIP_LONG
        
        if current_state in STATE_DRUG_AFTEREFFECT_SET:
            if next_state == STATE_START:
                r += self.config.shock_penalty
            elif next_state in STATE_DRUG_AFTEREFFECT_SET:
                r += self.config.aftereffect_penalty
            
        return r


class AdaptiveBetaAgent:
    def __init__(self, rng: np.random.Generator, fixed_beta: Optional[float] = None, 
                 td_threshold: float = 2.0, use_uncertainty_beta: bool = False):
        self.rng = rng
        self.fixed_beta = fixed_beta
        self.td_error_threshold = td_threshold
        self.use_uncertainty_beta = use_uncertainty_beta  # 不確実性ベースのβ適応を使用
        
        self.q_mf = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.q_mb = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        
        self.q_beta = np.zeros(len(BETA_VALUES), dtype=np.float64)
        self.current_beta_idx = 0
        self.prev_beta_idx = None
        
        self.td_error_history: List[float] = []
        self.recent_td_errors: List[float] = []
        self.td_error_window = 10
        self.beta_history: List[float] = []
        self.td_bias_scale = 0.5  # scales TD-error-driven MB bias
        self.epsilon_beta_local = EPSILON_BETA
        
        # 不確実性ベースβ適応用パラメータ
        self.td_window_size = 20
        self.td_error_ma = 0.0
        self.uncertainty_threshold_low = 0.1
        self.uncertainty_threshold_high = 0.5
        self.current_beta_value = 0.5
        
        if self.fixed_beta is not None:
            self.current_beta_idx = np.argmin(np.abs(BETA_VALUES - self.fixed_beta))
        else:
            # If uncertainty-based β is disabled, strengthen TD-based adaptation
            if not self.use_uncertainty_beta:
                self.td_error_threshold = max(0.5, self.td_error_threshold * 0.5)
                self.td_bias_scale = 1.0
                self.epsilon_beta_local = EPSILON_BETA * 0.5
        
        self.model_counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=np.float64)
        self.model_rewards = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=np.float64)
        self.model_visits = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.model_observed = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=bool)
        
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                self.model_counts[s, a, s] = 1.0
                self.model_visits[s, a] = 1.0
                self.model_observed[s, a, s] = True

    def select_beta(self) -> int:
        if self.fixed_beta is not None:
            return self.current_beta_idx
        
        # 不確実性ベースのβ適応
        if self.use_uncertainty_beta:
            return self._compute_uncertainty_based_beta()
        
        # 従来のTD誤差バイアス方式
        if len(self.recent_td_errors) >= 3:
            avg_td_error = np.mean(np.abs(self.recent_td_errors[-3:]))
            if avg_td_error > self.td_error_threshold:
                biased_q = self.q_beta.copy()
                for i, beta in enumerate(BETA_VALUES):
                    biased_q[i] += beta * avg_td_error * self.td_bias_scale
                max_val = np.max(biased_q)
                best_betas = np.flatnonzero(np.isclose(biased_q, max_val, rtol=1e-08, atol=1e-12))
                return int(self.rng.choice(best_betas))
        
        if self.rng.random() < self.epsilon_beta_local:
            return int(self.rng.integers(len(BETA_VALUES)))
        
        max_val = np.max(self.q_beta)
        best_betas = np.flatnonzero(np.isclose(self.q_beta, max_val, rtol=1e-08, atol=1e-12))
        return int(self.rng.choice(best_betas))
    
    def _compute_uncertainty_based_beta(self) -> int:
        """不確実性（TD誤差）に基づいてβ値を計算"""
        if self.td_error_ma < self.uncertainty_threshold_low:
            # 低不確実性: MF優先 (β = 0.2-0.4)
            self.current_beta_value = 0.2 + 0.2 * (self.td_error_ma / self.uncertainty_threshold_low)
        elif self.td_error_ma > self.uncertainty_threshold_high:
            # 高不確実性: MB優先 (β = 0.6-1.0)
            excess = min(self.td_error_ma - self.uncertainty_threshold_high, self.uncertainty_threshold_high)
            self.current_beta_value = 0.6 + 0.4 * (excess / self.uncertainty_threshold_high)
        else:
            # 中間不確実性: 線形補間 (β = 0.4-0.6)
            range_size = self.uncertainty_threshold_high - self.uncertainty_threshold_low
            position = (self.td_error_ma - self.uncertainty_threshold_low) / range_size
            self.current_beta_value = 0.4 + 0.2 * position
        
        self.current_beta_value = np.clip(self.current_beta_value, 0.0, 1.0)
        return int(np.argmin(np.abs(BETA_VALUES - self.current_beta_value)))
    
    def select_action(self, state: int) -> int:
        self.current_beta_idx = self.select_beta()
        current_beta = BETA_VALUES[self.current_beta_idx]
        self.beta_history.append(current_beta)
        
        self._plan_q_values()
        
        if self.rng.random() < EPSILON:
            return int(self.rng.integers(NUM_ACTIONS))
            
        q_mix = current_beta * self.q_mb[state] + (1.0 - current_beta) * self.q_mf[state]
        max_val = np.max(q_mix)
        best_actions = np.flatnonzero(np.isclose(q_mix, max_val, rtol=1e-08, atol=1e-12))
        return int(self.rng.choice(best_actions))

    def observe(self, state: int, action: int, reward: float, next_state: int):
        td_target = reward + DISCOUNT * np.max(self.q_mf[next_state])
        td_error = td_target - self.q_mf[state, action]
        
        self.td_error_history.append(td_error)
        self.recent_td_errors.append(td_error)
        if len(self.recent_td_errors) > self.td_error_window:
            self.recent_td_errors.pop(0)
        
        # 不確実性ベースβ適応用のTD誤差移動平均を更新
        if self.use_uncertainty_beta:
            abs_td_error = abs(td_error)
            # td_error_historyを移動平均ウィンドウとして使用
            if len(self.td_error_history) > self.td_window_size:
                # 最古のものを削除（recent_td_errorsとは別管理）
                recent_for_ma = self.td_error_history[-self.td_window_size:]
                self.td_error_ma = np.mean(np.abs(recent_for_ma))
            else:
                self.td_error_ma = np.mean(np.abs(self.td_error_history))
        
        if self.fixed_beta is None and self.prev_beta_idx is not None:
            td_target_beta = reward + DISCOUNT * np.max(self.q_beta)
            self.q_beta[self.prev_beta_idx] += ALPHA_BETA * (
                td_target_beta - self.q_beta[self.prev_beta_idx]
            )
        
        if self.fixed_beta is None:
            self.prev_beta_idx = self.current_beta_idx
        
        self.q_mf[state, action] += ALPHA_MF * td_error
        
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
# Experiment Data Structures
# ==========================================
@dataclass
class AgentPerformance:
    """Single agent's performance metrics"""
    agent_id: int
    beta_type: str
    fixed_beta: Optional[float]
    total_reward: float
    avg_reward_per_step: float
    rewards_per_window: List[float]
    td_errors: List[float]
    beta_history: List[float]
    env_change_points: List[int]
    
    # Recovery metrics (reward change after environment shift)
    recovery_speeds: List[float]  # How fast reward returns to baseline
    post_change_rewards: List[List[float]]  # Rewards in windows after each change
    phase_total_rewards: List[float]


@dataclass
class ExperimentCondition:
    """Results for one experimental condition"""
    condition_name: str
    volatility_interval: int
    beta_type: str
    fixed_beta: Optional[float]
    num_agents: int
    
    # Aggregate statistics
    mean_total_reward: float
    std_total_reward: float
    sem_total_reward: float
    ci_95_lower: float
    ci_95_upper: float
    
    mean_reward_per_step: float
    std_reward_per_step: float
    
    # Per-agent data
    agent_performances: List[AgentPerformance]
    
    # Window-level statistics
    mean_rewards_per_window: List[float]
    std_rewards_per_window: List[float]
    
    # Recovery statistics
    mean_recovery_speed: float
    std_recovery_speed: float
    all_recovery_speeds: List[float] = field(default_factory=list)  # 全recovery_speedsのリスト

    # Phase-specific totals
    mean_phase3_total_reward: float = 0.0
    std_phase3_total_reward: float = 0.0
    sem_phase3_total_reward: float = 0.0


@dataclass
class StatisticalComparison:
    """Statistical comparison between two conditions"""
    condition_a: str
    condition_b: str
    t_statistic: float
    p_value: float
    cohens_d: float
    mean_difference: float
    ci_95_difference: Tuple[float, float]
    significant: bool


# ==========================================
# Experiment Runner
# ==========================================
def run_single_agent(
    agent_id: int,
    beta_type: str,
    fixed_beta: Optional[float],
    volatility_interval: int,
    num_steps: int,
    seed: int,
    uncertainty_based_beta: bool = False,
    lightweight: bool = True
) -> AgentPerformance:
    """Run a single agent and collect performance metrics"""
    
    rng = np.random.default_rng(seed)
    env = VolatileAddictionEnvironment(rng, volatility_interval)
    
    # Use uncertainty-based beta for adaptive agents
    use_uncertainty = (beta_type == 'adaptive' and uncertainty_based_beta)
    agent = AdaptiveBetaAgent(rng, fixed_beta=fixed_beta, use_uncertainty_beta=use_uncertainty)
    
    def set_phase_config(phase_idx: int):
        """Configure environment for each phase before stepping."""
        if phase_idx == 0:
            # Safe pre-training: no addictive reward, penalties remain
            env.config.goal_reward = R_G
            env.config.drug_reward = 0.0
            env.config.aftereffect_penalty = -1.2
            env.config.shock_penalty = R_P
            env.config.goal_is_better = True
        elif phase_idx == 1:
            # High-reward/high-risk: strong addictive reward, penalties stay
            env.config.goal_reward = R_G
            env.config.drug_reward = 12.0
            env.config.aftereffect_penalty = -1.2
            env.config.shock_penalty = R_P
            env.config.goal_is_better = False
        else:
            # Dynamic phase baseline before volatility kicks in
            env.config.goal_reward = R_G
            env.config.drug_reward = 10.0
            env.config.aftereffect_penalty = -1.2
            env.config.shock_penalty = R_P
            env.config.goal_is_better = True
        env.config_history.append(VolatileEnvironmentConfig(
            goal_reward=env.config.goal_reward,
            drug_reward=env.config.drug_reward,
            aftereffect_penalty=env.config.aftereffect_penalty,
            shock_penalty=env.config.shock_penalty,
            goal_is_better=env.config.goal_is_better
        ))

    state = env.reset()
    rewards: List[float] = []
    env_change_points: List[int] = []
    window_rewards: List[float] = []
    current_window_rewards: List[float] = []
    post_change_rewards: List[List[float]] = []
    phase_total_rewards: List[float] = []

    phase_lengths = PHASE_STEPS
    total_steps = sum(phase_lengths)
    if num_steps != total_steps:
        # Respect explicit override while still following phase ordering proportionally
        scale = num_steps / total_steps
        phase_lengths = [max(1, int(round(p * scale))) for p in phase_lengths]
        total_steps = sum(phase_lengths)

    global_step = 0

    for phase_idx, phase_len in enumerate(phase_lengths):
        set_phase_config(phase_idx)
        if phase_idx == 2:
            # Start volatility clock for phase 3
            env.step_count = 0

        phase_reward_accum = 0.0
        for _ in range(phase_len):
            action = agent.select_action(state)
            allow_volatility = (phase_idx == 2)
            next_state, reward, env_changed = env.step(
                action,
                phase_idx=1,
                allow_volatility=allow_volatility
            )
            agent.observe(state, action, reward, next_state)

            rewards.append(reward)
            current_window_rewards.append(reward)
            phase_reward_accum += reward

            if env_changed:
                env_change_points.append(global_step)
                if current_window_rewards:
                    window_rewards.append(np.mean(current_window_rewards))
                current_window_rewards = []

                # Collect post-change rewards for recovery analysis
                post_window: List[float] = []
                post_change_rewards.append(post_window)

            if env_change_points and global_step > env_change_points[-1]:
                if len(post_change_rewards) > 0:
                    post_change_rewards[-1].append(reward)

            state = next_state
            global_step += 1

        phase_total_rewards.append(phase_reward_accum)

        # Flush window at phase boundary to avoid mixing phases
        if current_window_rewards:
            window_rewards.append(np.mean(current_window_rewards))
            current_window_rewards = []
    
    if current_window_rewards:
        window_rewards.append(np.mean(current_window_rewards))
    
    # Calculate recovery speeds
    recovery_speeds = []
    for i, change_point in enumerate(env_change_points):
        # 最低でも10ステップ分のpost-changeデータがあればrecovery計算可能
        if change_point + 10 < total_steps:
            pre_change = np.mean(rewards[max(0, change_point-10):change_point])
            post_10 = np.mean(rewards[change_point:change_point+10])
            
            # 20ステップ分のデータがあれば使用、なければ残りのステップを使用
            if change_point + 20 < total_steps:
                post_20 = np.mean(rewards[change_point+10:change_point+20])
            else:
                # 残りのステップを使用
                remaining_steps = total_steps - change_point - 10
                if remaining_steps > 0:
                    post_20 = np.mean(rewards[change_point+10:total_steps])
                else:
                    continue
            
            if abs(pre_change - post_10) > 0.01:
                speed = (post_20 - post_10) / (abs(pre_change - post_10) + 0.01)
                if -10 < speed < 10:  # Filter outliers
                    recovery_speeds.append(speed)
    
    # 軽量モードでは履歴データを保存しない（メモリ節約）
    # ただしrecovery_speedsは統計に重要なので保持
    if lightweight:
        return AgentPerformance(
            agent_id=agent_id,
            beta_type=beta_type,
            fixed_beta=fixed_beta,
            total_reward=sum(rewards),
            avg_reward_per_step=np.mean(rewards),
            rewards_per_window=window_rewards,
            td_errors=[],  # 空リスト
            beta_history=[],  # 空リスト
            env_change_points=env_change_points,
            recovery_speeds=recovery_speeds,  # 保持
            post_change_rewards=[],  # 空リスト
            phase_total_rewards=phase_total_rewards
        )
    else:
        return AgentPerformance(
            agent_id=agent_id,
            beta_type=beta_type,
            fixed_beta=fixed_beta,
            total_reward=sum(rewards),
            avg_reward_per_step=np.mean(rewards),
            rewards_per_window=window_rewards,
            td_errors=agent.td_error_history,
            beta_history=agent.beta_history,
            env_change_points=env_change_points,
            recovery_speeds=recovery_speeds,
            post_change_rewards=post_change_rewards,
            phase_total_rewards=phase_total_rewards
        )


def run_experiment_condition(
    condition_name: str,
    beta_type: str,
    fixed_beta: Optional[float],
    volatility_interval: int,
    num_agents: int,
    num_steps: int,
    base_seed: int,
    num_runs: int = 1,
    uncertainty_based_beta: bool = False,
    max_workers: int = None,
    batch_size: int = 100,
    lightweight: bool = True
) -> ExperimentCondition:
    """Run full experiment for one condition with memory-efficient batch processing"""
    
    print(f"\nRunning: {condition_name}")
    print(f"  Beta: {beta_type}, Volatility: {volatility_interval}, Agents: {num_agents}, Runs: {num_runs}")
    print(f"  Parallel workers: {max_workers if max_workers else 'auto'}, Batch size: {batch_size}")
    print(f"  Lightweight mode: {lightweight}")
    print_memory_status("  Initial ")
    
    start_time = time.time()
    
    # Prepare all agent tasks
    tasks = []
    for run in range(num_runs):
        run_seed_offset = run * 100000
        for i in range(num_agents):
            agent_idx = run * num_agents + i
            seed = base_seed + run_seed_offset + i
            tasks.append((
                agent_idx,
                beta_type,
                fixed_beta,
                volatility_interval,
                num_steps,
                seed,
                uncertainty_based_beta,
                lightweight
            ))
    
    # 集計用の変数（メモリ効率化のため段階的に集計）
    total_rewards = []
    reward_per_steps = []
    all_recovery_speeds = []
    window_rewards_accumulator = []
    phase3_total_rewards = []
    
    total_tasks = len(tasks)
    completed = 0
    
    # バッチ処理で実行
    for batch_start in range(0, total_tasks, batch_size):
        batch_end = min(batch_start + batch_size, total_tasks)
        batch_tasks = tasks[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//batch_size + 1}/{(total_tasks + batch_size - 1)//batch_size}...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch tasks
            future_to_task = {executor.submit(run_single_agent, *task): task for task in batch_tasks}
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                perf = future.result()
                
                # 即座に集計（メモリ節約）
                total_rewards.append(perf.total_reward)
                reward_per_steps.append(perf.avg_reward_per_step)
                all_recovery_speeds.extend(perf.recovery_speeds)
                window_rewards_accumulator.append(perf.rewards_per_window)
                if perf.phase_total_rewards and len(perf.phase_total_rewards) >= 3:
                    phase3_total_rewards.append(perf.phase_total_rewards[2])
                
                completed += 1
                
                if completed % 50 == 0 or completed == total_tasks:
                    elapsed = time.time() - start_time
                    print(f"    Progress: {completed}/{total_tasks} agents completed ({elapsed:.1f}s)")
                    print_memory_status("    ")
        
        # バッチ終了後にガベージコレクション
        gc.collect()
    
    # Aggregate statistics from accumulated data
    mean_total = np.mean(total_rewards)
    std_total = np.std(total_rewards, ddof=1)
    sem_total = stats.sem(total_rewards)
    ci_95 = stats.t.interval(0.95, len(total_rewards)-1, loc=mean_total, scale=sem_total)
    
    actual_num_agents = len(total_rewards)
    
    # Window-level aggregation
    max_windows = max(len(w) for w in window_rewards_accumulator) if window_rewards_accumulator else 0
    mean_rewards_per_window = []
    std_rewards_per_window = []
    
    for w in range(max_windows):
        window_vals = [rewards[w] for rewards in window_rewards_accumulator 
                      if len(rewards) > w]
        if window_vals:
            mean_rewards_per_window.append(np.mean(window_vals))
            std_rewards_per_window.append(np.std(window_vals, ddof=1))
    
    # Recovery statistics
    mean_recovery = np.mean(all_recovery_speeds) if all_recovery_speeds else 0
    std_recovery = np.std(all_recovery_speeds, ddof=1) if len(all_recovery_speeds) > 1 else 0

    # Phase 3 total reward statistics (volatile phase)
    if phase3_total_rewards:
        mean_phase3_total = np.mean(phase3_total_rewards)
        std_phase3_total = np.std(phase3_total_rewards, ddof=1) if len(phase3_total_rewards) > 1 else 0
        sem_phase3_total = stats.sem(phase3_total_rewards) if len(phase3_total_rewards) > 1 else 0
    else:
        mean_phase3_total = 0.0
        std_phase3_total = 0.0
        sem_phase3_total = 0.0
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s | Mean reward: {mean_total:.2f} ± {sem_total:.2f}")
    print_memory_status("  Final ")
    
    # 注意: agent_performancesはrecovery分析等で必要なので、軽量モードでも保持
    # （ただし各AgentPerformanceの履歴データは空にしてメモリ節約）
    # agent_performancesはバッチ処理中に構築されていないので、ここで再構築する必要がある
    # しかし、既にデータを集計してしまっているので、空リストのままにする
    # visualization側で対処する
    agent_performances = [] if lightweight else None
    
    # ガベージコレクション
    gc.collect()
    
    return ExperimentCondition(
        condition_name=condition_name,
        volatility_interval=volatility_interval,
        beta_type=beta_type,
        fixed_beta=fixed_beta,
        num_agents=actual_num_agents,
        mean_total_reward=mean_total,
        std_total_reward=std_total,
        sem_total_reward=sem_total,
        ci_95_lower=ci_95[0],
        ci_95_upper=ci_95[1],
        mean_reward_per_step=np.mean(reward_per_steps),
        std_reward_per_step=np.std(reward_per_steps, ddof=1),
        agent_performances=agent_performances,
        mean_rewards_per_window=mean_rewards_per_window,
        std_rewards_per_window=std_rewards_per_window,
        mean_recovery_speed=mean_recovery,
        std_recovery_speed=std_recovery,
        all_recovery_speeds=all_recovery_speeds,  # visualization用に保存
        mean_phase3_total_reward=mean_phase3_total,
        std_phase3_total_reward=std_phase3_total,
        sem_phase3_total_reward=sem_phase3_total
    )


def compute_statistical_comparison(cond_a: ExperimentCondition, 
                                   cond_b: ExperimentCondition,
                                   use_summary_stats: bool = True,
                                   metric: str = "overall") -> StatisticalComparison:
    """Compute statistical comparison between two conditions"""
    
    # 軽量モードの場合は、要約統計量から近似計算
    if use_summary_stats or not cond_a.agent_performances:
        # 要約統計量を使用した近似t検定
        n_a = cond_a.num_agents
        n_b = cond_b.num_agents
        if metric == "phase3":
            mean_a = cond_a.mean_phase3_total_reward
            mean_b = cond_b.mean_phase3_total_reward
            std_a = cond_a.std_phase3_total_reward
            std_b = cond_b.std_phase3_total_reward
        else:
            mean_a = cond_a.mean_total_reward
            mean_b = cond_b.mean_total_reward
            std_a = cond_a.std_total_reward
            std_b = cond_b.std_total_reward
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / (n_a + n_b - 2))
        if pooled_std < 1e-8:
            pooled_std = 1e-8
        
        # t-statistic
        se_diff = pooled_std * np.sqrt(1/n_a + 1/n_b)
        if se_diff < 1e-8:
            t_stat = 0.0
            p_val = 1.0
            cohens_d = 0.0
            mean_diff = mean_a - mean_b
            ci_diff = (0.0, 0.0)
            return StatisticalComparison(
                condition_a=cond_a.condition_name,
                condition_b=cond_b.condition_name,
                t_statistic=t_stat,
                p_value=p_val,
                cohens_d=cohens_d,
                mean_difference=mean_diff,
                ci_95_difference=ci_diff,
                significant=False
            )
        t_stat = (mean_a - mean_b) / se_diff
        
        # p-value
        df = n_a + n_b - 2
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # Cohen's d
        cohens_d = (mean_a - mean_b) / pooled_std
        
        # Mean difference
        mean_diff = mean_a - mean_b
        
        # 95% CI for difference
        t_crit = stats.t.ppf(0.975, df)
        ci_diff = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)
        
        return StatisticalComparison(
            condition_a=cond_a.condition_name,
            condition_b=cond_b.condition_name,
            t_statistic=t_stat,
            p_value=p_val,
            cohens_d=cohens_d,
            mean_difference=mean_diff,
            ci_95_difference=ci_diff,
            significant=(p_val < 0.05)
        )
    
    # 詳細データがある場合は従来の方法
    if metric == "phase3":
        rewards_a = [p.phase_total_rewards[2] for p in cond_a.agent_performances if len(p.phase_total_rewards) >= 3]
        rewards_b = [p.phase_total_rewards[2] for p in cond_b.agent_performances if len(p.phase_total_rewards) >= 3]
    else:
        rewards_a = [p.total_reward for p in cond_a.agent_performances]
        rewards_b = [p.total_reward for p in cond_b.agent_performances]
    
    # t-test
    t_stat, p_val = stats.ttest_ind(rewards_a, rewards_b)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(rewards_a)-1)*np.var(rewards_a, ddof=1) + 
                          (len(rewards_b)-1)*np.var(rewards_b, ddof=1)) / 
                         (len(rewards_a) + len(rewards_b) - 2))
    if pooled_std < 1e-8:
        pooled_std = 1e-8
    cohens_d = (np.mean(rewards_a) - np.mean(rewards_b)) / pooled_std
    
    # Mean difference
    mean_diff = np.mean(rewards_a) - np.mean(rewards_b)
    
    # 95% CI for difference
    se_diff = np.sqrt(np.var(rewards_a, ddof=1)/len(rewards_a) + 
                     np.var(rewards_b, ddof=1)/len(rewards_b))
    df = len(rewards_a) + len(rewards_b) - 2
    t_crit = stats.t.ppf(0.975, df)
    ci_diff = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)
    
    return StatisticalComparison(
        condition_a=cond_a.condition_name,
        condition_b=cond_b.condition_name,
        t_statistic=t_stat,
        p_value=p_val,
        cohens_d=cohens_d,
        mean_difference=mean_diff,
        ci_95_difference=ci_diff,
        significant=(p_val < 0.05)
    )


# ==========================================
# Main Experiment
# ==========================================
def run_large_scale_experiment(
    num_agents: int = 200,
    num_steps: int = TOTAL_PHASE_STEPS,
    base_seed: int = 42,
    num_runs: int = 1,
    volatility_intervals: List[int] = [200, 1000, 500],
    output_dir: str = "large_scale_results",
    max_workers: int = None,
    batch_size: int = 100,
    lightweight: bool = True,
    use_uncertainty_beta: bool = True
):
    """Run comprehensive large-scale experiment with memory-efficient processing"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("LARGE-SCALE VOLATILITY EXPERIMENT")
    print("="*70)
    print(f"Agents per condition: {num_agents}")
    print(f"Steps per agent: {num_steps}")
    print(f"Phase schedule: {PHASE_STEPS} (safe → risky → volatile)")
    print(f"Number of runs: {num_runs}")
    print(f"Volatility intervals: {volatility_intervals} (shorter to stress rapid adaptation)")
    print(f"Fixed betas: {BETA_VALUES}")
    print(f"Total conditions: {len(volatility_intervals) * (len(BETA_VALUES) + 1)}")
    print(f"Total agent runs per run: {num_agents * len(volatility_intervals) * (len(BETA_VALUES) + 1)}")
    print(f"Total agent runs (all runs): {num_agents * len(volatility_intervals) * (len(BETA_VALUES) + 1) * num_runs}")
    print(f"Batch size: {batch_size}")
    print(f"Lightweight mode: {lightweight}")
    print(f"Adaptive β uncertainty-based update: {use_uncertainty_beta}")
    print_memory_status("Initial ")
    print("="*70)
    
    all_conditions = []
    
    for vol_interval in volatility_intervals:
        print(f"\n{'='*70}")
        print(f"VOLATILITY INTERVAL: {vol_interval} steps")
        print(f"{'='*70}")
        
        # Run adaptive beta
        cond = run_experiment_condition(
            condition_name=f"Adaptive_β_vol{vol_interval}",
            beta_type="adaptive",
            fixed_beta=None,
            volatility_interval=vol_interval,
            num_agents=num_agents,
            num_steps=num_steps,
            base_seed=base_seed + vol_interval * 10000,
            num_runs=num_runs,
            uncertainty_based_beta=use_uncertainty_beta,
            max_workers=max_workers,
            batch_size=batch_size,
            lightweight=lightweight
        )
        all_conditions.append(cond)
        gc.collect()  # 条件間でガベージコレクション
        
        # Run all fixed betas
        for beta_val in BETA_VALUES:
            cond = run_experiment_condition(
                condition_name=f"Fixed_β={beta_val:.1f}_vol{vol_interval}",
                beta_type="fixed",
                fixed_beta=beta_val,
                volatility_interval=vol_interval,
                num_agents=num_agents,
                num_steps=num_steps,
                base_seed=base_seed + vol_interval * 10000 + int(beta_val * 1000),
                num_runs=num_runs,
                max_workers=max_workers,
                batch_size=batch_size,
                lightweight=lightweight
            )
            all_conditions.append(cond)
            gc.collect()  # 条件間でガベージコレクション
    
    # Statistical comparisons
    print(f"\n{'='*70}")
    print("STATISTICAL COMPARISONS")
    print(f"{'='*70}")
    
    comparisons = []
    for vol_interval in volatility_intervals:
        adaptive_cond = next(c for c in all_conditions if c.beta_type == "adaptive" and 
                           c.volatility_interval == vol_interval)
        
        print(f"\nVolatility interval: {vol_interval}")
        print(f"{'Comparison':<40} {'t-stat':<10} {'p-value':<12} {'Cohen d':<10} {'Sig':<5}")
        print("-" * 80)
        
        for beta_val in BETA_VALUES:
            fixed_cond = next(c for c in all_conditions if c.beta_type == "fixed" and 
                            c.fixed_beta == beta_val and c.volatility_interval == vol_interval)
            
            comp = compute_statistical_comparison(adaptive_cond, fixed_cond, metric="phase3")
            comparisons.append(comp)
            
            sig_marker = "***" if comp.p_value < 0.001 else ("**" if comp.p_value < 0.01 else ("*" if comp.p_value < 0.05 else ""))
            comparison_label = f"Adaptive vs Fixed β={beta_val:.1f}"
            print(f"{comparison_label:<40} {comp.t_statistic:>9.3f} {comp.p_value:>11.4f} {comp.cohens_d:>9.3f} {sig_marker:<5}")
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save condition summaries
    summary_data = []
    for cond in all_conditions:
        summary_data.append({
            'condition_name': cond.condition_name,
            'volatility_interval': cond.volatility_interval,
            'beta_type': cond.beta_type,
            'fixed_beta': cond.fixed_beta,
            'num_agents': cond.num_agents,
            'mean_total_reward': cond.mean_total_reward,
            'std_total_reward': cond.std_total_reward,
            'sem_total_reward': cond.sem_total_reward,
            'ci_95_lower': cond.ci_95_lower,
            'ci_95_upper': cond.ci_95_upper,
            'mean_reward_per_step': cond.mean_reward_per_step,
            'mean_recovery_speed': cond.mean_recovery_speed,
            'mean_phase3_total_reward': cond.mean_phase3_total_reward,
            'std_phase3_total_reward': cond.std_phase3_total_reward,
            'sem_phase3_total_reward': cond.sem_phase3_total_reward,
        })
    
    import pandas as pd
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    print(f"Saved: {output_dir}/summary_statistics.csv")
    
    # Save comparison results
    comp_data = []
    for comp in comparisons:
        comp_data.append({
            'condition_a': comp.condition_a,
            'condition_b': comp.condition_b,
            't_statistic': comp.t_statistic,
            'p_value': comp.p_value,
            'cohens_d': comp.cohens_d,
            'mean_difference': comp.mean_difference,
            'ci_95_lower': comp.ci_95_difference[0],
            'ci_95_upper': comp.ci_95_difference[1],
            'significant': comp.significant
        })
    
    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(f"{output_dir}/statistical_comparisons.csv", index=False)
    print(f"Saved: {output_dir}/statistical_comparisons.csv")
    
    # Save full data with pickle
    with open(f"{output_dir}/full_experiment_data.pkl", 'wb') as f:
        pickle.dump({
            'conditions': all_conditions,
            'comparisons': comparisons,
            'parameters': {
                'num_agents': num_agents,
                'num_steps': num_steps,
                'num_runs': num_runs,
                'base_seed': base_seed,
                'volatility_intervals': volatility_intervals,
                'phase_steps': PHASE_STEPS,
                'phase3_steps': PHASE_STEPS[2],
                'use_uncertainty_beta': use_uncertainty_beta
            }
        }, f)
    print(f"Saved: {output_dir}/full_experiment_data.pkl")
    
    return all_conditions, comparisons


# ==========================================
# Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Large-scale volatility experiment')
    parser.add_argument('--num-agents', type=int, default=1000, 
                       help='Number of agents per condition per run')
    parser.add_argument('--num-steps', type=int, default=8000,
                       help='Number of steps per agent')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of runs with different seeds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed')
    parser.add_argument('--output-dir', type=str, default='large_scale_results',
                       help='Output directory')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: CPU count)')
    parser.add_argument('--batch-size', type=int, default=200,
                       help='Number of agents to process in each batch (default: 100)')
    parser.add_argument('--no-lightweight', action='store_true',
                       help='Disable lightweight mode (saves full agent histories)')
    parser.add_argument('--no-uncertainty-beta', action='store_true',
                       help='Disable uncertainty-based beta adaptation for adaptive agents')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    conditions, comparisons = run_large_scale_experiment(
        num_agents=args.num_agents,
        num_steps=args.num_steps,
        num_runs=args.num_runs,
        base_seed=args.seed,
        volatility_intervals=[200, 1000, 500],
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        lightweight=not args.no_lightweight,
        use_uncertainty_beta=not args.no_uncertainty_beta
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETED in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - summary_statistics.csv")
    print("  - statistical_comparisons.csv")
    print("  - full_experiment_data.pkl")


if __name__ == "__main__":
    main()
