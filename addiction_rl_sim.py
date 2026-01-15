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

# Drug/After-effect区間（State 7を含む）
STATE_DRUG_AFTEREFFECT_SET = {STATE_DRUG} | STATE_AFTER_SET

# アフターエフェクト区間でのループを定義
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
MODEL_DECAY = 0.01   # モデルカウントの減衰率 (毎ステップ)
INITIAL_TRANSITION_COUNT = 5.0  # 新規遷移観測時の初期カウント
EPSILON = 0.1        # 探索率
N_PRIORITIZED_SWEEPS = 50 #思考回数
T_MB = 1.0           # Softmax temperature for planning

# β学習用のパラメータ
BETA_VALUES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)  # β候補値
ALPHA_BETA = 0.05     # βのQ学習率
EPSILON_BETA = 0.1   # β選択時の探索率

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

# 状態遷移
AS_ACTION_TARGETS = {
    ACTION_AS2: 1,
    ACTION_AS3: 2,
    ACTION_AS4: 3,
    ACTION_AS5: 4,
    ACTION_AS6: 5,
    ACTION_AS7: 6,
}

# debug用の名前
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

# 遷移確率テーブル (f1: Pre-drug, f2: Addiction, f3: Reversal)
NEUTRAL_MOVE_SUCCESS = [0.99, 0.99, 0.99] #隣り合う状態遷移
NEUTRAL_SKIP_SUCCESS = [0.0001, 0.0001, 0.0001] #離れた状態遷移
# アフターエフェクト区間の状態遷移
AFTER_AG_EXIT = [0.001, 0.001, 0.001]
AFTER_AS_EXIT = [0.001, 0.001, 0.001]
AW_FORWARD = [0.4995, 0.4995, 0.4995]
AW_BACKWARD = [0.4995, 0.4995, 0.4995]
# 状態15(14)での特別遷移
AW_SPECIAL_MOVE = [0.4, 0.4, 0.4]
AW_SPECIAL_EXIT = [0.6, 0.6, 0.6]
AD_FORWARD = [0.745, 0.745, 0.745]
AD_BACKWARD = [0.245, 0.245, 0.245]

# フェーズ定義: (名前, ステップ数, 薬物報酬, Goal報酬)
# reversal フェーズでは報酬が逆転: Drug=1.0(健康的), Goal=10.0(依存的)
PHASES: Tuple[Tuple[str, int, float, float], ...] = (
    ("pre-drug", 50, 0.0, 1.0),      # Drug報酬=0, Goal報酬=1
    ("addiction", 1000, 10.0, 1.0),   # Drug報酬=10, Goal報酬=1
    ("reversal", 1000, 1.0, 10.0),    # Drug報酬=1, Goal報酬=10 (逆転)
)

# 報酬テーブルの事前構築
PHASE_REWARD_TABLE = np.zeros((len(PHASES), NUM_STATES)) #3x22
PHASE_DRUG_REWARDS = np.array([phase[2] for phase in PHASES], dtype=np.float64) #[0.0, 10.0, 1.0]
PHASE_GOAL_REWARDS = np.array([phase[3] for phase in PHASES], dtype=np.float64) #[1.0, 1.0, 10.0]
# アフターエフェクト区間の報酬は_reward関数内で動的に計算するため、ここでは設定しない

# フェーズ3(reversal)用のアフターエフェクト報酬 (Goal版)
# reversalフェーズではGoal報酬獲得後にGoal版アフターエフェクト区間に入る
AFTER_PHASE_REWARDS_REVERSAL = -1.2  # reversalフェーズでのGoal版アフターエフェクト罰則


# ==========================================
# 2. Numba 高速化カーネル (Model-Based Planning)
# ==========================================
@jit(nopython=True, cache=False)
def run_prioritized_sweeping(
    q_mb, 
    model_counts, 
    model_rewards,
    num_states, 
    num_actions, 
    n_sweeps, 
    t_mb,
    discount,  # 割引率
    rand_vals  # 事前生成された乱数配列 (長さ = n_sweeps)
):
    """
    Numba対応版 Early Interrupted Stochastic Prioritized Sweeping.
    - q_mb: (num_states, num_actions) に対して、選ばれた状態のみバックアップを書き込む
    - model_counts, model_rewards: shape (num_states, num_actions, num_states)
    - discount: 割引率 (gamma)
    - rand_vals: 一様乱数 [0,1) の配列（長さ >= n_sweeps）
    """
    # 局所変数初期化
    H = np.zeros(num_states, dtype=np.float64)   # 優先度
    V = np.zeros(num_states, dtype=np.float64)   # 状態価値（max_a Q）
    # 一時配列
    q_temp = np.zeros(num_actions, dtype=np.float64)
    probs = np.zeros(num_states, dtype=np.float64)  # softmax 分布
    exp_vals = np.zeros(num_states, dtype=np.float64)

    # 事前に next-state確率の分母（countsの和）を使うための一時変数
    # main loop: n_sweeps 回だけ "早期打ち切り" で1状態ずつ処理
    steps = 0
    while steps < n_sweeps:
        # ---------------------------
        # 1) softmax(H / t_mb) による状態サンプリング
        # ---------------------------
        # Compute exp(H / t_mb) safely (ここでは t_mb > 0 を想定)
        inv_t = 1.0 / t_mb
        sum_exp = 0.0
        for i in range(num_states):
            # e^{H[i]/T}
            val = math.exp(H[i] * inv_t)
            exp_vals[i] = val
            sum_exp += val

        # 正規化して累積でサンプリング
        # sum_exp が 0 になることはほぼない（exp >= 0）だが安全対策
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

        # ---------------------------
        # 2) 選ばれた状態 s_tilde の Q(s_tilde, a) をモデルから計算
        #    Q(s,a) = sum_{s'} P(s'|s,a) [ R(s,a,s') + V[s'] ]
        # ---------------------------
        for a in range(num_actions):
            q_val = 0.0
            # denom = sum_s' counts[s_tilde,a,s']
            denom = 0.0
            for sp in range(num_states):
                denom += model_counts[s_tilde, a, sp]

            if denom <= 0.0:
                # その行動は観測されていない -> 期待値0として扱う（あるいは既存 q_mb を利用する選択肢もある）
                q_temp[a] = 0.0
                continue

            # accumulate expectation
            for sp in range(num_states):
                cnt = model_counts[s_tilde, a, sp]
                if cnt <= 0.0:
                    continue
                p = cnt / denom
                # 期待報酬: model_rewards / cnt (累積報酬をカウントで割る)
                r_avg = model_rewards[s_tilde, a, sp] / cnt
                q_val += p * (r_avg + discount * V[sp])
            q_temp[a] = q_val

        # 書き込み：q_mb の s_tilde 行だけ更新（論文疑似コードに準拠）
        for a in range(num_actions):
            q_mb[s_tilde, a] = q_temp[a]

        # ---------------------------
        # 3) V の更新と Δ 計算
        # ---------------------------
        # M = max_a Q(s_tilde, a)
        M = q_temp[0]
        for a in range(1, num_actions):
            if q_temp[a] > M:
                M = q_temp[a]

        delta = V[s_tilde] - M
        if delta < 0.0:
            delta = -delta
        V[s_tilde] = M

        # ---------------------------
        # 4) 逆伝播 h(s) = Δ * max_a P(s_tilde | s, a)
        #    ただし、自己遷移(s == s_tilde)は除外する
        # ---------------------------
        h = np.zeros(num_states, dtype=np.float64)
        # For each predecessor state s, find max_a P(s_tilde | s, a)
        for s in range(num_states):
                
            max_p = 0.0
            for a in range(num_actions):
                # denom for (s,a)
                denom_sa = 0.0
                for sp in range(num_states):
                    denom_sa += model_counts[s, a, sp]
                if denom_sa <= 0.0:
                    continue
                p_s_to_st = model_counts[s, a, s_tilde] / denom_sa
                if p_s_to_st > max_p:
                    max_p = p_s_to_st
            h[s] = delta * max_p

        # ---------------------------
        # 5) H の更新（s_tilde は上書き、他は max で蓄積）
        # ---------------------------
        H[s_tilde] = h[s_tilde]
        for s in range(num_states):
            if s == s_tilde:
                continue
            # H[s] = max(h[s], H[s])
            if h[s] > H[s]:
                H[s] = h[s]

        steps += 1

    # end while
    return  # q_mb は参照渡しで更新される


# ==========================================
# 3. クラス定義 (Environment / Agent)
# ==========================================
@dataclass
class PhaseResult:
    drug_choices: int = 0
    healthy_choices: int = 0
    # reversalフェーズ用: Goalが依存的、Drugが健康的に逆転
    reversal_goal_choices: int = 0    # reversalでのGoal選択（依存的）
    reversal_drug_choices: int = 0    # reversalでのDrug選択（健康的）
    
    # フェーズ別状態訪問カウント（分析用）
    # addiction_phase_state_visits[state] = そのフェーズで状態stateを訪問した回数
    addiction_phase_state_visits: Optional[np.ndarray] = None
    reversal_phase_state_visits: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.addiction_phase_state_visits is None:
            self.addiction_phase_state_visits = np.zeros(NUM_STATES, dtype=np.int64)
        if self.reversal_phase_state_visits is None:
            self.reversal_phase_state_visits = np.zeros(NUM_STATES, dtype=np.int64)

@dataclass
class BetaStatistics:
    """β選択の統計情報（全エージェント）"""
    beta_history_all: List[List[float]]  # 各エージェントの各ステップでのβ値
    state_history_all: List[List[int]]   # 各エージェントの各ステップでの状態
    phase_history_all: List[List[int]]   # 各エージェントの各ステップでのフェーズ
    step_history: List[int]              # グローバルステップ番号（全エージェント共通）
    addiction_status: List[bool]         # 各エージェントが依存症になったか
    num_agents: int                      # 収集したエージェント数

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
        
        # エージェントの現在地によって適応される行動ルールが異なる
        if curr in STATE_NEUTRAL_SET:
            next_state = self._transition_neutral(curr, action, phase_idx)
        elif curr == STATE_DRUG or curr in STATE_AFTER_SET:
            next_state = self._transition_aftereffect(curr, action, phase_idx)
        elif curr == STATE_GOAL:
            # reversalフェーズではGoal行動でアフターエフェクト区間(STATE_DRUG)へ
            # 通常フェーズではGoal行動でSTARTへ
            if action == ACTION_GOAL:
                if phase_idx == 2:  # reversalフェーズ
                    next_state = STATE_DRUG  # Goal報酬後にアフターエフェクト区間へ
                else:
                    next_state = STATE_START
            else:
                next_state = STATE_GOAL
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
            # reversalフェーズではDrug行動は健康的（STARTへ直接戻る）
            if phase_idx == 2:  # reversalフェーズ
                return STATE_START  # 健康的報酬を得てSTARTに戻る
            else:
                return STATE_DRUG  # 通常通りアフターエフェクト区間へ
            
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
                # 論文準拠: 左右への移動確率を計0.4 (片側0.2) に変更
                # 元コード: p_move = AW_SPECIAL_MOVE[phase_idx] / 2.0 (0.1)
                
                p_move_each = 0.2  # ここを 0.1 から 0.2 に変更
                
                if roll < p_move_each:
                    return max(STATE_DRUG, state - 1)
                roll -= p_move_each
                if roll < p_move_each:
                    return min(AFTER_MAX, state + 1)
                # 残り (0.6) は全て脱出。留まる確率は0にする。
                return STATE_START
            
            # Normal Aftereffect State
            if roll < AW_FORWARD[phase_idx]:
                return aftereffect_forward_state(state)
            roll -= AW_FORWARD[phase_idx]
            if roll < AW_BACKWARD[phase_idx]:
                return aftereffect_backward_state(state)
            # 残り0.1%でlocation 4 (STATE_START) へ抜ける
            return STATE_START

        if action == ACTION_DRUG:
            if roll < AD_FORWARD[phase_idx]:
                return aftereffect_forward_state(state)
            roll -= AD_FORWARD[phase_idx]
            if roll < AD_BACKWARD[phase_idx]:
                return aftereffect_backward_state(state)
            # 残り1%でlocation 4 (STATE_START) へ抜ける
            return STATE_START
            
        return state

    def _reward(self, current_state: int, action: int, next_state: int, phase_idx: int) -> float:
        r = PHASE_REWARD_TABLE[phase_idx, next_state]

        # === reversalフェーズ（phase_idx=2）の報酬ロジック ===
        if phase_idx == 2:
            # reversal: Goal報酬=10(依存的), Drug報酬=1(健康的)
            # Goal状態(0)でGoal行動→Drug状態(7)へ（Goal報酬獲得後アフターエフェクト）
            if current_state == STATE_GOAL and action == ACTION_GOAL and next_state == STATE_DRUG:
                r += PHASE_GOAL_REWARDS[phase_idx]  # Goal報酬=10.0
            # Neutral最終(6)でDrug行動→START(3)へ（健康的報酬）
            elif current_state == NEUTRAL_MAX and action == ACTION_DRUG and next_state == STATE_START:
                r += PHASE_DRUG_REWARDS[phase_idx]  # Drug報酬=1.0（健康的）
            # アフターエフェクト区間での罰則（Goal版アフターエフェクト）
            if current_state in STATE_DRUG_AFTEREFFECT_SET:
                if next_state == STATE_START:  # 状態4への遷移
                    r += R_P  # -4 (Goal版アフターエフェクトからの脱出罰則)
                elif next_state in STATE_DRUG_AFTEREFFECT_SET:  # アフターエフェクト区間内での遷移
                    r += AFTER_PHASE_REWARDS_REVERSAL  # -1.2 (Goal版アフターエフェクト罰則)
        else:
            # === 通常フェーズ（pre-drug, addiction）の報酬ロジック ===
            # 状態0でagした際（Goal報酬）
            if current_state == STATE_GOAL and action == ACTION_GOAL and next_state == STATE_START:
                r += PHASE_GOAL_REWARDS[phase_idx]  # Goal報酬=1.0
            # 状態6でadした際（薬物報酬）
            elif (current_state == NEUTRAL_MAX and action == ACTION_DRUG and next_state == STATE_DRUG):
                r += PHASE_DRUG_REWARDS[phase_idx]
            
            # アフターエフェクト区間での報酬設計
            if current_state in STATE_DRUG_AFTEREFFECT_SET:
                if next_state == STATE_START:  # 状態4への遷移
                    r += R_P  # -4 (f1, f2共通)
                elif next_state in STATE_DRUG_AFTEREFFECT_SET:  # アフターエフェクト区間内での遷移
                    r += AFTER_PHASE_REWARDS[phase_idx]  # f1: -0.3, f2: -1.2
        
        # Neutralエリアでのロングジャンプ失敗コスト（全フェーズ共通）
        if (current_state in STATE_NEUTRAL_SET and next_state in STATE_NEUTRAL_SET 
            and abs(next_state - current_state) > 1):
            r += R_SKIP_LONG
            
        return r


class HybridAgent:
    def __init__(self, rng: np.random.Generator, mb_forget: bool = False, fixed_beta: Optional[float] = None):
        self.rng = rng
        self.mb_forget = mb_forget
        self.fixed_beta = fixed_beta  # 固定β値(Noneの場合は学習モード)
        
        # Qテーブル (MF, MB)
        self.q_mf = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        self.q_mb = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        
        # β学習用のQ値テーブル (どのβ値を選ぶべきかをグローバルに学習)
        self.q_beta = np.zeros(len(BETA_VALUES), dtype=np.float64)
        self.current_beta_idx = 0  # 現在選択されているβのインデックス
        self.prev_beta_idx = None  # β選択のQ学習用に前のβインデックスを記憶
        
        # 固定β値モードの場合、対応するインデックスを設定
        if self.fixed_beta is not None:
            self.current_beta_idx = np.argmin(np.abs(BETA_VALUES - self.fixed_beta))
        
        # メンタルモデル (Numba用にfloat64で定義)
        # [今の状態, 行動, 次の状態] に何回遷移したかを記録する3次元配列
        self.model_counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=np.float64)
        # そこで得られた報酬の累積値
        self.model_rewards = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=np.float64)
        # その状態・行動を何回試したかの合計
        self.model_visits = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
        # 観測済みの遷移を記録するフラグ(減衰後の再初期化を防ぐため)
        self.model_observed = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=bool)
        
        # 初期モデル: 全ての行動が自己遷移すると仮定 (論文準拠)
        # "The initial model assumes that transitions bring the agent deterministically to the same state"
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                self.model_counts[s, a, s] = 1.0  # 自己遷移のカウント
                self.model_visits[s, a] = 1.0
                self.model_observed[s, a, s] = True  # 初期状態も観測済みとしてマーク

    def select_beta(self) -> int:
        """β値を選択する（ε-greedy or 固定値）"""
        # 固定β値モードの場合は常に同じインデックスを返す
        if self.fixed_beta is not None:
            return self.current_beta_idx
        
        # ε-Greedy でβを選択
        if self.rng.random() < EPSILON_BETA:
            return int(self.rng.integers(len(BETA_VALUES)))
        
        # Q値が最大のβを選択
        max_val = np.max(self.q_beta)
        best_betas = np.flatnonzero(np.isclose(self.q_beta, max_val, rtol=1e-08, atol=1e-12))
        return int(self.rng.choice(best_betas))
    
    def select_action(self, state: int) -> int:
        # βを選択（状態に関係なくグローバルに選択）
        self.current_beta_idx = self.select_beta()
        current_beta = BETA_VALUES[self.current_beta_idx]
        
        # MB Planning (JIT function call)
        self._plan_q_values()
        
        # ε-Greedy
        if self.rng.random() < EPSILON:
            return int(self.rng.integers(NUM_ACTIONS))
            
        # Hybrid Q-value (選択されたβを使用)
        q_mix = current_beta * self.q_mb[state] + (1.0 - current_beta) * self.q_mf[state]
        
        # 最も価値が高い行動を選ぶ (Argmax)
        max_val = np.max(q_mix)
        # 同着1位が複数ある場合に備えて、最大値に近い行動を全てリストアップ
        best_actions = np.flatnonzero(np.isclose(q_mix, max_val,rtol=1e-08, atol=1e-12))
        # 同着の中からランダムに1つ選んで返す
        return int(self.rng.choice(best_actions))

    # モデルの学習機構
    def observe(self, state: int, action: int, reward: float, next_state: int, phase_idx: int):
        # --- β学習の更新(固定β値モードではスキップ) ---
        if self.fixed_beta is None and self.prev_beta_idx is not None:
            # 前のステップで選択したβのQ値を更新
            # TD学習: Q(β) ← Q(β) + α * (R + γ*max_β' Q(β') - Q(β))
            td_target_beta = reward + DISCOUNT * np.max(self.q_beta)
            self.q_beta[self.prev_beta_idx] += ALPHA_BETA * (
                td_target_beta - self.q_beta[self.prev_beta_idx]
            )
        
        # 次回のβ更新のために現在のβインデックスを記憶(固定β値モードではスキップ)
        if self.fixed_beta is None:
            self.prev_beta_idx = self.current_beta_idx
        
        # --- Model-Free (直感) の更新 ---
        # Q学習の式: Q(s,a) ← Q(s,a) + α * (R + γ*maxQ(s') - Q(s,a))
        td_target = reward + DISCOUNT * np.max(self.q_mf[next_state])
        self.q_mf[state, action] += ALPHA_MF * (td_target - self.q_mf[state, action])
        # MB Model Update (論文準拠)
        # 1. カウント減衰
        self.model_counts *= (1.0 - MODEL_DECAY)
        self.model_rewards *= (1.0 - MODEL_DECAY)
        self.model_visits *= (1.0 - MODEL_DECAY)

        # 2. 新規遷移の検出と初期カウント設定
        # "The first time a new transition is observed an initial count is set to 5"
        # 論文の "while keeping a degree of uncertainty" を守るため、
        # 観測済みフラグをチェックして、減衰後の再初期化を防ぐ
        if not self.model_observed[state, action, next_state]:
            # 真の新規遷移: 初期カウントを設定
            self.model_counts[state, action, next_state] = INITIAL_TRANSITION_COUNT
            # 報酬もカウントに合わせてスケールして記録
            self.model_rewards[state, action, next_state] = reward * INITIAL_TRANSITION_COUNT
            # 観測済みとしてマーク
            self.model_observed[state, action, next_state] = True
            # visits も初期カウント分増やす
            self.model_visits[state, action] += INITIAL_TRANSITION_COUNT
        else:
            # 既に観測済みの遷移: カウントを +1 するだけ
            self.model_counts[state, action, next_state] += 1.0
            self.model_rewards[state, action, next_state] += reward
            self.model_visits[state, action] += 1.0

    def _plan_q_values(self):
        """高速化カーネルを呼び出す"""
        if self.mb_forget:
            # 完全忘却モード: 毎回MB推定をゼロから再構築
            self.q_mb.fill(0.0)

        # 乱数配列を事前生成 (再現性のため、単一のRNGを使用)
        rand_vals = self.rng.random(N_PRIORITIZED_SWEEPS)

        run_prioritized_sweeping(
            self.q_mb,
            self.model_counts,
            self.model_rewards,
            NUM_STATES,
            NUM_ACTIONS,
            N_PRIORITIZED_SWEEPS,
            T_MB,
            DISCOUNT,
            rand_vals
        )


# ==========================================
# 4. シミュレーション実行 & Main
# ==========================================
def simulate(
    num_agents: int,
    num_runs: int,
    base_seed: int,
    progress_cb=None,
    mb_forget: bool = False,
    debug_episode: bool = False,
    debug_csv_path: Optional[str] = None,
    debug_txt_path: Optional[str] = None,
    collect_beta_stats: bool = False,
    fixed_beta: Optional[float] = None,
    collect_occupancy_data: bool = False,
    time_bin_size: int = 2000,
) -> Tuple[float, List[float], float, List[float], dict, Optional[BetaStatistics], Optional[dict]]:
    """
    シミュレーションを実行する。
    
    Returns:
        Tuple of:
        - overall_addiction_rate: 全体の依存率
        - run_addiction_rates: 各ラン毎の依存率
        - overall_reversal_adaptation_rate: 全体のreversal適応率
        - run_reversal_rates: 各ラン毎のreversal適応率
        - state_visit_stats: 状態訪問統計（フェーズ別）
        - beta_stats: β統計情報（収集した場合）
        - occupancy_data: 時系列状態占有率データ（収集した場合）
    """
    addictions = 0
    reversal_adaptations = 0  # reversal適応したエージェント数
    total_agents = num_agents * num_runs
    debug_records = []
    # ラン毎の依存者数を記録（後で run ごとの率に変換）
    run_addictions = [0 for _ in range(num_runs)]
    run_reversal_adaptations = [0 for _ in range(num_runs)]  # reversal適応数
    
    # 状態訪問統計（全エージェント集計用）
    total_addiction_state_visits = np.zeros(NUM_STATES, dtype=np.int64)
    total_reversal_state_visits = np.zeros(NUM_STATES, dtype=np.int64)
    
    # β統計収集用（全エージェント）
    beta_stats = None
    if collect_beta_stats:
        beta_stats = BetaStatistics(
            beta_history_all=[],
            state_history_all=[],
            phase_history_all=[],
            step_history=[],
            addiction_status=[],
            num_agents=0
        )
    
    # 時系列占有率データ収集用
    occupancy_data = None
    if collect_occupancy_data:
        # 全フェーズの総ステップ数を計算
        total_steps = sum(phase[1] for phase in PHASES)
        # 必要なbin数を計算（切り上げ）
        num_bins = (total_steps + time_bin_size - 1) // time_bin_size
        # time_binsを正しく生成（num_bins + 1個の境界値）
        time_bins = [i * time_bin_size for i in range(num_bins + 1)]
        
        occupancy_data = {
            'time_bin_size': time_bin_size,
            'num_bins': num_bins,
            'total_steps': total_steps,
            'time_bins': time_bins,
            'runs': []  # 各runのデータを保存
        }
    
    # テキストログファイルを開く (追記モードではなく新規作成)
    log_file = None
    if debug_episode and debug_txt_path:
        log_file = open(debug_txt_path, "w", encoding="utf-8")

    try:
        for seed_idx in range(num_runs):
            # この run の時系列データ（全エージェント）
            if collect_occupancy_data:
                run_occupancy = {
                    'agents': []  # 各エージェントの状態訪問履歴
                }
            print(f"\n[Run {seed_idx + 1}/{num_runs}] Starting...")
            for agent_idx in range(num_agents):
                # 100体ごとに進捗表示
                if agent_idx > 0 and agent_idx % 100 == 0:
                    current_addiction_rate = run_addictions[seed_idx] / agent_idx * 100
                    print(f"  [Run {seed_idx + 1}/{num_runs}] Processed {agent_idx}/{num_agents} agents | Current addiction rate: {current_addiction_rate:.2f}%")
                
                # 固定ペアリングシード: (run, agent) ごとに一意のシードを割り当て
                # これにより同じ (run, agent) 番号は全ての beta 値で同じ初期 RNG を使います
                seed_for_agent = base_seed + seed_idx * num_agents + agent_idx
                rng = np.random.default_rng(seed_for_agent)

                env = AddictionEnvironment(rng)
                agent = HybridAgent(rng, mb_forget=mb_forget, fixed_beta=fixed_beta)
                
                state = env.reset()
                counts = PhaseResult()
                
                # グローバルステップカウンタ
                global_step = 0
                
                # 各エージェントのβ履歴（β統計収集用）
                agent_beta_history = [] if collect_beta_stats else None
                agent_state_history = [] if collect_beta_stats else None
                agent_phase_history = [] if collect_beta_stats else None
                
                # 各エージェントの状態訪問履歴（occupancy data収集用）
                agent_state_sequence = [] if collect_occupancy_data else None
                
                # フェーズ実行
                for phase_idx, (_, length, _, _) in enumerate(PHASES):
                    # フェーズ開始時にSTATE_STARTにリセット
                    # if phase_idx > 0:
                    #     state = env.reset()
                    
                    report_points = {1, length // 2, length}
                    
                    for step_in_phase in range(length):
                        # Progress Log
                        if progress_cb and (step_in_phase + 1) in report_points:
                            progress_cb(seed_idx, agent_idx, num_agents, num_runs, phase_idx, step_in_phase + 1, length)
                        
                        # 行動選択前にQ値を取得したいが、select_action内でMBのPlanningが走るため
                        # select_action後に取得すると、そのステップでのPlanning結果が反映された状態になる
                        # ここでは「行動選択に使われたQ値」に近いものを表示するため、select_action直後の値を参照する
                        
                        # 行動選択(MBはPlaning)
                        action = agent.select_action(state)
                        
                        # β統計収集（全エージェント）
                        if collect_beta_stats:
                            selected_beta = BETA_VALUES[agent.current_beta_idx]
                            agent_beta_history.append(selected_beta)
                            agent_state_history.append(state)
                            agent_phase_history.append(phase_idx)
                        
                        # 状態訪問履歴を記録（occupancy data収集用）
                        if collect_occupancy_data:
                            agent_state_sequence.append(state)
                        
                        # step_historyはβ統計収集時のみ記録
                        if collect_beta_stats and seed_idx == 0 and agent_idx == 0:
                            beta_stats.step_history.append(global_step)
                        
                        # Debug出力用にβ値とQ値を取得
                        current_q_mf = None
                        current_q_mb = None
                        current_q_mix = None
                        current_beta_value = None
                        current_q_beta_values = None
                        if debug_episode and seed_idx == 0 and agent_idx == 0:
                            current_q_mf = agent.q_mf[state].copy()
                            current_q_mb = agent.q_mb[state].copy()
                            current_beta_value = BETA_VALUES[agent.current_beta_idx]
                            current_q_beta_values = agent.q_beta.copy()
                            current_q_mix = current_beta_value * current_q_mb + (1.0 - current_beta_value) * current_q_mf
                        
                        # 行動、学習
                        next_state, reward = env.step(action, phase_idx)
                        agent.observe(state, action, reward, next_state, phase_idx)

                        if debug_episode and seed_idx == 0 and agent_idx == 0:
                            # ログメッセージの構築
                            q_mf_str = ", ".join([f"{x:.6f}" for x in current_q_mf])
                            q_mb_str = ", ".join([f"{x:.6f}" for x in current_q_mb])
                            q_mix_str = ", ".join([f"{x:.6f}" for x in current_q_mix])
                            
                            log_lines = [
                                f"[DEBUG] Beta={current_beta_value:.2f} Phase={PHASES[phase_idx][0]} Step={step_in_phase+1}",
                                f"  State: {state} -> Action: {ACTION_NAMES.get(action, str(action))} -> Next: {next_state} (Reward: {reward})",
                                f"  Q_MF:  [{q_mf_str}]",
                                f"  Q_MB:  [{q_mb_str}]",
                                f"  Q_MIX: [{q_mix_str}]",
                            ]
                            
                            # βのQ値を出力
                            if current_q_beta_values is not None:
                                q_beta_str = ", ".join([f"{x:.6f}" for x in current_q_beta_values])
                                beta_names = ", ".join([f"β={b:.1f}" for b in BETA_VALUES])
                                log_lines.append(f"  Q_BETA: [{q_beta_str}] ({beta_names})")
                            
                            # 現在の状態と行動に紐づくMBモデル（遷移カウントと累積報酬）も併せて出力
                            curr_model_counts = agent.model_counts[state, action].copy()
                            curr_model_rewards = agent.model_rewards[state, action].copy()
                            model_counts_str = ", ".join([f"{x:.3f}" for x in curr_model_counts])
                            model_rewards_str = ", ".join([f"{x:.3f}" for x in curr_model_rewards])
                            
                            log_lines.extend([
                                f"  MODEL_COUNTS(s,a,*):  [{model_counts_str}]",
                                f"  MODEL_REWARDS(s,a,*): [{model_rewards_str}]",
                                "-" * 40
                            ])
                            
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
                        
                        # === 状態訪問統計の収集 ===
                        if phase_idx == 1:  # Addictionフェーズ
                            counts.addiction_phase_state_visits[state] += 1
                        elif phase_idx == 2:  # Reversalフェーズ
                            counts.reversal_phase_state_visits[state] += 1
                        
                        # 統計収集 (AddictionフェーズとReversalフェーズ)
                        # === Addictionフェーズ (phase_idx=1) ===
                        # Drug選択: 
                        #   1. Neutral最終状態(6)からDrug行動でDrug状態(7)へ遷移
                        #   2. アフターエフェクト区域(Drug状態含む)に滞在/行動して罰則を受けた時
                        # Healthy選択: Goal状態(0)からGoal行動でStart状態(3)へ遷移（報酬獲得）
                        if phase_idx == 1:
                            if (state == NEUTRAL_MAX and action == ACTION_DRUG 
                                and next_state == STATE_DRUG):
                                counts.drug_choices += 1
                            elif (state == STATE_DRUG or state in STATE_AFTER_SET) and \
                                 (next_state == STATE_DRUG or next_state in STATE_AFTER_SET):
                                # アフターエフェクト区域内に留まった場合（罰則-1.2を受ける）
                                counts.drug_choices += 1
                            elif (state == STATE_GOAL and action == ACTION_GOAL 
                                  and next_state == STATE_START):
                                counts.healthy_choices += 1
                        
                        # === Reversalフェーズ (phase_idx=2) ===
                        # 報酬が逆転: Goal=依存的(高報酬+アフターエフェクト), Drug=健康的(低報酬+直接戻る)
                        # Goal選択(依存的): Goal状態(0)→Goal行動→Drug状態(7)へ(アフターエフェクト入り)
                        # Drug選択(健康的): Neutral最終(6)→Drug行動→Start(3)へ(直接戻る)
                        elif phase_idx == 2:
                            if (state == STATE_GOAL and action == ACTION_GOAL 
                                and next_state == STATE_DRUG):
                                counts.reversal_goal_choices += 1  # Goal選択（reversalでは依存的）
                            elif (state == STATE_DRUG or state in STATE_AFTER_SET) and \
                                 (next_state == STATE_DRUG or next_state in STATE_AFTER_SET):
                                # Goal版アフターエフェクト区域内に留まった場合
                                counts.reversal_goal_choices += 1
                            elif (state == NEUTRAL_MAX and action == ACTION_DRUG 
                                  and next_state == STATE_START):
                                counts.reversal_drug_choices += 1  # Drug選択（reversalでは健康的）
                                
                        state = next_state
                        global_step += 1
                
                # 依存判定（addictionフェーズ）
                is_addicted = counts.drug_choices > counts.healthy_choices
                if is_addicted:
                    addictions += 1
                    run_addictions[seed_idx] += 1
                
                # reversal適応度判定（reversalフェーズで健康的選択が多いほど良い適応）
                # reversal_drug_choices = 健康的（新環境に適応）
                # reversal_goal_choices = 依存的（旧環境の習慣を継続）
                reversal_adapted = counts.reversal_drug_choices > counts.reversal_goal_choices
                if reversal_adapted:
                    reversal_adaptations += 1
                    run_reversal_adaptations[seed_idx] += 1
                
                # 状態訪問統計を集計
                total_addiction_state_visits += counts.addiction_phase_state_visits
                total_reversal_state_visits += counts.reversal_phase_state_visits
                
                # β統計を集約（依存状態も含める）
                if collect_beta_stats:
                    beta_stats.beta_history_all.append(agent_beta_history)
                    beta_stats.state_history_all.append(agent_state_history)
                    beta_stats.phase_history_all.append(agent_phase_history)
                    beta_stats.addiction_status.append(is_addicted)
                    beta_stats.num_agents += 1
                
                # 状態訪問履歴を保存（occupancy data収集用）
                if collect_occupancy_data:
                    run_occupancy['agents'].append(agent_state_sequence)

                if debug_episode and seed_idx == 0 and agent_idx == 0 and log_file:
                    log_file.write(f"Final Counts - Drug Choices: {counts.drug_choices}, Healthy Choices: {counts.healthy_choices}\n")
                    log_file.write(f"Reversal Counts - Goal Choices (addictive): {counts.reversal_goal_choices}, Drug Choices (healthy): {counts.reversal_drug_choices}\n")
                    log_file.write(f"Reversal Adaptation: {'Adapted' if reversal_adapted else 'Not Adapted'}\n")
            
            # Run終了時の統計表示
            run_addiction_rate = run_addictions[seed_idx] / num_agents * 100
            run_reversal_rate = run_reversal_adaptations[seed_idx] / num_agents * 100
            print(f"  [Run {seed_idx + 1}/{num_runs}] Completed! Addiction rate: {run_addiction_rate:.2f}%, Reversal adaptation rate: {run_reversal_rate:.2f}%")
            
            # このrunのoccupancy dataを集計
            if collect_occupancy_data:
                occupancy_data['runs'].append(run_occupancy)
    finally:
        if log_file:
            log_file.close()
            print(f"Debug log (seed=0, agent=0) written to {debug_txt_path}")

    if debug_records:
        if debug_csv_path:
            csv_path = debug_csv_path
        else:
            beta_str = f"{fixed_beta:.2f}" if fixed_beta is not None else "learning"
            csv_path = f"debug_episode_beta_{beta_str}.csv"
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

    # run ごとの率を計算して返す
    run_rates = [cnt / num_agents for cnt in run_addictions]
    run_reversal_rates = [cnt / num_agents for cnt in run_reversal_adaptations]
    overall_rate = addictions / total_agents
    overall_reversal_rate = reversal_adaptations / total_agents
    
    # 状態訪問統計を割合に変換
    addiction_total_steps = np.sum(total_addiction_state_visits)
    reversal_total_steps = np.sum(total_reversal_state_visits)
    
    addiction_state_ratios = (total_addiction_state_visits / addiction_total_steps * 100).tolist() if addiction_total_steps > 0 else [0.0] * NUM_STATES
    reversal_state_ratios = (total_reversal_state_visits / reversal_total_steps * 100).tolist() if reversal_total_steps > 0 else [0.0] * NUM_STATES
    
    # 統計結果を辞書で返す
    state_visit_stats = {
        "addiction_state_ratios": addiction_state_ratios,
        "reversal_state_ratios": reversal_state_ratios,
        "addiction_total_steps": int(addiction_total_steps),
        "reversal_total_steps": int(reversal_total_steps),
    }
    
    # occupancy dataを時間区間ごとに集計
    if collect_occupancy_data:
        # 全run、全エージェントのデータを統合して時間区間ごとの状態占有率を計算
        num_bins = occupancy_data['num_bins']
        total_steps = occupancy_data['total_steps']
        
        # 各状態について、各時間区間での占有率を計算（全run平均）
        states_occupancy = {}
        for state_id in range(NUM_STATES):
            state_occupancy_by_run = []
            
            for run_data in occupancy_data['runs']:
                # この run の全エージェントのデータを統合
                bin_counts = np.zeros(num_bins)
                bin_totals = np.zeros(num_bins)
                
                for agent_sequence in run_data['agents']:
                    for step, state in enumerate(agent_sequence):
                        if step >= total_steps:
                            break
                        bin_idx = min(step // time_bin_size, num_bins - 1)
                        bin_totals[bin_idx] += 1
                        if state == state_id:
                            bin_counts[bin_idx] += 1
                
                # この run での各区間の占有率を計算
                run_occupancy = np.zeros(num_bins)
                for bin_idx in range(num_bins):
                    if bin_totals[bin_idx] > 0:
                        run_occupancy[bin_idx] = bin_counts[bin_idx] / bin_totals[bin_idx]
                
                state_occupancy_by_run.append(run_occupancy.tolist())
            
            states_occupancy[str(state_id)] = state_occupancy_by_run
        
        # occupancy dataに集計結果を追加
        occupancy_data['states'] = states_occupancy
        # runs（生データ）は容量が大きいので削除
        del occupancy_data['runs']
    
    return overall_rate, run_rates, overall_reversal_rate, run_reversal_rates, state_visit_stats, beta_stats, occupancy_data




def plot_beta_analysis(beta_stats: BetaStatistics, output_prefix: str = "beta_analysis"):
    """β選択の分析グラフを複数作成（全エージェントの平均±標準偏差）"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib
    
    # 日本語フォント設定（Windows環境）
    matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
    
    # 全エージェントのデータを2次元配列に変換
    # beta_history_all: List[List[float]] -> (num_agents, num_steps)
    beta_matrix = np.array(beta_stats.beta_history_all)  # shape: (num_agents, num_steps)
    state_matrix = np.array(beta_stats.state_history_all)
    phase_matrix = np.array(beta_stats.phase_history_all)
    step_array = np.array(beta_stats.step_history)
    
    # 各ステップでの平均と標準偏差を計算
    beta_mean = np.mean(beta_matrix, axis=0)  # shape: (num_steps,)
    beta_std = np.std(beta_matrix, axis=0)
    state_mean = np.mean(state_matrix, axis=0)
    phase_mean = phase_matrix[0]  # フェーズは全エージェント共通
    
    # ===== Figure 1: β選択の時系列変化（平均±標準偏差） =====
    fig1 = plt.figure(figsize=(14, 8))
    gs1 = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # 上段: β値の時系列（平均±標準偏差）
    ax1_top = fig1.add_subplot(gs1[0])
    colors = ['#3498db' if p == 0 else '#e74c3c' for p in phase_mean]
    
    # 平均値をプロット
    ax1_top.plot(step_array, beta_mean, 'k-', linewidth=2, label=f'Mean β (n={beta_stats.num_agents} agents)')
    
    # 標準偏差をシェーディング
    ax1_top.fill_between(step_array, beta_mean - beta_std, beta_mean + beta_std, 
                          alpha=0.3, color='gray', label='±1 SD')
    
    # フェーズ境界を描画
    phase_boundary = np.where(np.diff(phase_mean) != 0)[0]
    for boundary in phase_boundary:
        ax1_top.axvline(step_array[boundary], color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1_top.text(step_array[boundary], 1.02, 'Phase Change', ha='center', fontsize=9)
    
    ax1_top.set_ylabel('Selected β Value', fontsize=12)
    ax1_top.set_ylim(-0.05, 1.1)
    ax1_top.set_title(f'β Selection Over Time - Mean±SD across {beta_stats.num_agents} agents', 
                      fontsize=14, fontweight='bold')
    ax1_top.legend(loc='upper right')
    ax1_top.grid(True, alpha=0.3)
    
    # 下段: 状態の時系列（平均）
    ax1_bottom = fig1.add_subplot(gs1[1], sharex=ax1_top)
    ax1_bottom.plot(step_array, state_mean, 'b-', linewidth=1, alpha=0.7)
    ax1_bottom.set_xlabel('Step', fontsize=12)
    ax1_bottom.set_ylabel('Mean State', fontsize=12)
    ax1_bottom.set_title('Average State Trajectory', fontsize=12)
    ax1_bottom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_timeseries.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_timeseries.png")
    plt.close()
    
    # ===== Figure 2: フェーズ別β選択分布（全エージェント集計） =====
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    phase_names = ['Pre-drug', 'Addiction']
    for phase_idx, (ax, phase_name) in enumerate(zip(axes2, phase_names)):
        # 全エージェントのデータを統合
        all_phase_betas = []
        for agent_phases, agent_betas in zip(phase_matrix, beta_matrix):
            mask = agent_phases == phase_idx
            all_phase_betas.extend(agent_betas[mask])
        
        if len(all_phase_betas) == 0:
            continue
        
        phase_betas = np.array(all_phase_betas)
        
        # ヒストグラム
        counts, bins, patches = ax.hist(phase_betas, bins=BETA_VALUES.tolist() + [1.1], 
                                        align='left', rwidth=0.8, alpha=0.7, 
                                        color='#3498db' if phase_idx == 0 else '#e74c3c')
        
        # 各βの選択頻度を表示
        for i, beta_val in enumerate(BETA_VALUES):
            count = np.sum(np.isclose(phase_betas, beta_val))
            percentage = 100.0 * count / len(phase_betas)
            ax.text(beta_val, count, f'{percentage:.1f}%', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('β Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{phase_name} Phase (n={len(phase_betas)} steps, {beta_stats.num_agents} agents)', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(BETA_VALUES)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_phase_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_phase_distribution.png")
    plt.close()
    
    # ===== Figure 3: 状態別β選択ヒートマップ（全エージェント統合） =====
    fig3, ax3 = plt.subplots(figsize=(16, 8))
    
    # 各状態でのβ選択頻度をカウント（全エージェント）
    state_beta_matrix = np.zeros((NUM_STATES, len(BETA_VALUES)))
    for agent_states, agent_betas in zip(state_matrix, beta_matrix):
        for state, beta in zip(agent_states, agent_betas):
            beta_idx = np.argmin(np.abs(BETA_VALUES - beta))
            state_beta_matrix[int(state), beta_idx] += 1
    
    # 各状態で正規化（割合に変換）
    row_sums = state_beta_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # ゼロ除算回避
    state_beta_matrix_norm = state_beta_matrix / row_sums * 100
    
    # ヒートマップ描画
    im = ax3.imshow(state_beta_matrix_norm.T, aspect='auto', cmap='YlOrRd', 
                    interpolation='nearest', vmin=0, vmax=100)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Selection Percentage (%)', fontsize=12)
    
    # 軸ラベル
    ax3.set_xlabel('State', fontsize=12)
    ax3.set_ylabel('β Value', fontsize=12)
    ax3.set_title(f'β Selection Heatmap by State ({beta_stats.num_agents} agents)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xticks(range(NUM_STATES))
    ax3.set_yticks(range(len(BETA_VALUES)))
    ax3.set_yticklabels([f'{b:.1f}' for b in BETA_VALUES])
    
    # 重要な状態を強調表示
    important_states = [STATE_START, STATE_GOAL_ENTRY, STATE_GOAL, NEUTRAL_MAX, STATE_DRUG]
    for s in important_states:
        ax3.axvline(s - 0.5, color='blue', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_state_heatmap.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_state_heatmap.png")
    plt.close()
    
    # ===== Figure 4: β選択の全体統計サマリー（全エージェント） =====
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    
    # 全エージェントのデータを統合
    beta_all = beta_matrix.flatten()
    state_all = state_matrix.flatten()
    phase_all = phase_matrix.flatten()
    
    # (1) 全体のβ分布（円グラフ）
    ax4_1 = axes4[0, 0]
    beta_counts = [np.sum(np.isclose(beta_all, b)) for b in BETA_VALUES]
    colors_pie = plt.cm.viridis(np.linspace(0, 1, len(BETA_VALUES)))
    wedges, texts, autotexts = ax4_1.pie(beta_counts, labels=[f'β={b:.1f}' for b in BETA_VALUES],
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax4_1.set_title(f'Overall β Distribution\n({beta_stats.num_agents} agents)', 
                    fontsize=13, fontweight='bold')
    
    # (2) フェーズごとの平均β
    ax4_2 = axes4[0, 1]
    phase_means = [np.mean(beta_all[phase_all == i]) for i in range(len(PHASES))]
    phase_stds = [np.std(beta_all[phase_all == i]) for i in range(len(PHASES))]
    bars = ax4_2.bar(phase_names, phase_means, yerr=phase_stds, 
                     color=['#3498db', '#e74c3c'], alpha=0.7, capsize=10)
    ax4_2.set_ylabel('Mean β Value', fontsize=12)
    ax4_2.set_ylim(0, 1)
    ax4_2.set_title('Average β by Phase', fontsize=13, fontweight='bold')
    ax4_2.grid(True, alpha=0.3, axis='y')
    
    # 値を表示
    for bar, mean, std in zip(bars, phase_means, phase_stds):
        ax4_2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02, 
                  f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # (3) 状態タイプ別の平均β
    ax4_3 = axes4[1, 0]
    state_types = {
        'Goal (0)': [STATE_GOAL],
        'Neutral (1-6)': list(STATE_NEUTRALS),
        'Drug (7)': [STATE_DRUG],
        'After (8-21)': list(STATE_AFTEREFFECTS)
    }
    type_means = []
    type_stds = []
    type_labels = []
    for label, states in state_types.items():
        mask = np.isin(state_all, states)
        if np.sum(mask) > 0:
            type_means.append(np.mean(beta_all[mask]))
            type_stds.append(np.std(beta_all[mask]))
            type_labels.append(label)
    
    bars3 = ax4_3.barh(type_labels, type_means, xerr=type_stds, 
                       color=['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'], 
                       alpha=0.7, capsize=10)
    ax4_3.set_xlabel('Mean β Value', fontsize=12)
    ax4_3.set_xlim(0, 1)
    ax4_3.set_title('Average β by State Type', fontsize=13, fontweight='bold')
    ax4_3.grid(True, alpha=0.3, axis='x')
    
    # (4) β値の推移（学習曲線、平均±標準偏差）
    ax4_4 = axes4[1, 1]
    n_bins = 20
    steps_per_bin = len(step_array) // n_bins
    if steps_per_bin > 0:
        bin_means = []
        bin_stds = []
        bin_steps = []
        for i in range(n_bins):
            start_idx = i * steps_per_bin
            end_idx = (i + 1) * steps_per_bin if i < n_bins - 1 else len(step_array)
            bin_means.append(np.mean(beta_mean[start_idx:end_idx]))
            bin_stds.append(np.mean(beta_std[start_idx:end_idx]))
            bin_steps.append(np.mean(step_array[start_idx:end_idx]))
        
        ax4_4.errorbar(bin_steps, bin_means, yerr=bin_stds, 
                      fmt='o-', linewidth=2, markersize=8, color='#e67e22',
                      capsize=5, capthick=2)
        ax4_4.set_xlabel('Step', fontsize=12)
        ax4_4.set_ylabel('Mean β Value', fontsize=12)
        ax4_4.set_ylim(0, 1)
        ax4_4.set_title('β Learning Curve (binned average±SD)', fontsize=13, fontweight='bold')
        ax4_4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_summary.png")
    plt.close()
    
    # ===== Figure 5: 依存群 vs 非依存群の比較 =====
    addiction_array = np.array(beta_stats.addiction_status)
    n_addicted = np.sum(addiction_array)
    n_non_addicted = len(addiction_array) - n_addicted
    
    if n_addicted > 0 and n_non_addicted > 0:
        fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
        
        # 依存群と非依存群のデータを分離
        beta_addicted = []
        beta_non_addicted = []
        for i, is_addicted in enumerate(addiction_array):
            if is_addicted:
                beta_addicted.extend(beta_matrix[i])
            else:
                beta_non_addicted.extend(beta_matrix[i])
        
        beta_addicted = np.array(beta_addicted)
        beta_non_addicted = np.array(beta_non_addicted)
        
        # (1) β選択頻度の比較（棒グラフ）
        ax5_1 = axes5[0, 0]
        x_pos = np.arange(len(BETA_VALUES))
        width = 0.35
        
        counts_addicted = [np.sum(np.isclose(beta_addicted, b)) for b in BETA_VALUES]
        counts_non_addicted = [np.sum(np.isclose(beta_non_addicted, b)) for b in BETA_VALUES]
        
        freq_addicted = [c / len(beta_addicted) * 100 for c in counts_addicted]
        freq_non_addicted = [c / len(beta_non_addicted) * 100 for c in counts_non_addicted]
        
        bars1 = ax5_1.bar(x_pos - width/2, freq_addicted, width, 
                         label=f'Addicted (n={n_addicted})', color='#e74c3c', alpha=0.7)
        bars2 = ax5_1.bar(x_pos + width/2, freq_non_addicted, width,
                         label=f'Non-addicted (n={n_non_addicted})', color='#2ecc71', alpha=0.7)
        
        ax5_1.set_xlabel('β Value', fontsize=12)
        ax5_1.set_ylabel('Selection Frequency (%)', fontsize=12)
        ax5_1.set_title('β Selection Frequency: Addicted vs Non-addicted', 
                       fontsize=13, fontweight='bold')
        ax5_1.set_xticks(x_pos)
        ax5_1.set_xticklabels([f'{b:.1f}' for b in BETA_VALUES])
        ax5_1.legend()
        ax5_1.grid(True, alpha=0.3, axis='y')
        
        # (2) 分布の比較（ヒストグラム重ね合わせ）
        ax5_2 = axes5[0, 1]
        ax5_2.hist(beta_addicted, bins=20, alpha=0.6, label=f'Addicted (n={n_addicted})',
                  color='#e74c3c', density=True)
        ax5_2.hist(beta_non_addicted, bins=20, alpha=0.6, label=f'Non-addicted (n={n_non_addicted})',
                  color='#2ecc71', density=True)
        ax5_2.set_xlabel('β Value', fontsize=12)
        ax5_2.set_ylabel('Density', fontsize=12)
        ax5_2.set_title('β Distribution Comparison', fontsize=13, fontweight='bold')
        ax5_2.legend()
        ax5_2.grid(True, alpha=0.3)
        
        # (3) 統計量の比較（箱ひげ図）
        ax5_3 = axes5[1, 0]
        bp = ax5_3.boxplot([beta_addicted, beta_non_addicted],
                           labels=['Addicted', 'Non-addicted'],
                           patch_artist=True,
                           widths=0.6)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#2ecc71')
        bp['boxes'][1].set_alpha(0.7)
        
        ax5_3.set_ylabel('β Value', fontsize=12)
        ax5_3.set_title('β Value Distribution (Boxplot)', fontsize=13, fontweight='bold')
        ax5_3.grid(True, alpha=0.3, axis='y')
        
        # 平均値を表示
        mean_add = np.mean(beta_addicted)
        mean_non = np.mean(beta_non_addicted)
        ax5_3.plot([1, 2], [mean_add, mean_non], 'D', markersize=10, 
                  color='orange', label='Mean', zorder=3)
        ax5_3.legend()
        
        # (4) フェーズ別平均の比較
        ax5_4 = axes5[1, 1]
        
        # 依存群のフェーズ別平均
        phase_addicted = []
        for i, is_addicted in enumerate(addiction_array):
            if is_addicted:
                phase_addicted.append(phase_matrix[i])
        phase_addicted = np.array(phase_addicted)
        
        beta_addicted_by_phase = []
        for i, is_addicted in enumerate(addiction_array):
            if is_addicted:
                beta_addicted_by_phase.append(beta_matrix[i])
        beta_addicted_by_phase = np.array(beta_addicted_by_phase)
        
        # 非依存群のフェーズ別平均
        phase_non_addicted = []
        beta_non_addicted_by_phase = []
        for i, is_addicted in enumerate(addiction_array):
            if not is_addicted:
                phase_non_addicted.append(phase_matrix[i])
                beta_non_addicted_by_phase.append(beta_matrix[i])
        phase_non_addicted = np.array(phase_non_addicted)
        beta_non_addicted_by_phase = np.array(beta_non_addicted_by_phase)
        
        phase_names = ['Pre-drug', 'Addiction']
        x_pos_phase = np.arange(len(phase_names))
        width_phase = 0.35
        
        means_add = []
        stds_add = []
        means_non = []
        stds_non = []
        
        for phase_idx in range(len(PHASES)):
            # 依存群
            mask_add = phase_addicted == phase_idx
            if np.sum(mask_add) > 0:
                phase_data = beta_addicted_by_phase[mask_add].flatten()
                means_add.append(np.mean(phase_data))
                stds_add.append(np.std(phase_data))
            else:
                means_add.append(0)
                stds_add.append(0)
            
            # 非依存群
            mask_non = phase_non_addicted == phase_idx
            if np.sum(mask_non) > 0:
                phase_data = beta_non_addicted_by_phase[mask_non].flatten()
                means_non.append(np.mean(phase_data))
                stds_non.append(np.std(phase_data))
            else:
                means_non.append(0)
                stds_non.append(0)
        
        bars3 = ax5_4.bar(x_pos_phase - width_phase/2, means_add, width_phase,
                         yerr=stds_add, label='Addicted', 
                         color='#e74c3c', alpha=0.7, capsize=5)
        bars4 = ax5_4.bar(x_pos_phase + width_phase/2, means_non, width_phase,
                         yerr=stds_non, label='Non-addicted',
                         color='#2ecc71', alpha=0.7, capsize=5)
        
        ax5_4.set_xlabel('Phase', fontsize=12)
        ax5_4.set_ylabel('Mean β Value', fontsize=12)
        ax5_4.set_title('Average β by Phase: Addicted vs Non-addicted', 
                       fontsize=13, fontweight='bold')
        ax5_4.set_xticks(x_pos_phase)
        ax5_4.set_xticklabels(phase_names)
        ax5_4.set_ylim(0, 1)
        ax5_4.legend()
        ax5_4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_addiction_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_prefix}_addiction_comparison.png")
        plt.close()
        
        # ===== Figure 6: 追加の詳細分析グラフ =====
        fig6, axes6 = plt.subplots(2, 3, figsize=(18, 10))
        
        # (1) β変動性の比較（各エージェントの標準偏差）
        ax6_1 = axes6[0, 0]
        std_addicted = [np.std(beta_matrix[i]) for i, is_add in enumerate(addiction_array) if is_add]
        std_non_addicted = [np.std(beta_matrix[i]) for i, is_add in enumerate(addiction_array) if not is_add]
        
        bp1 = ax6_1.boxplot([std_addicted, std_non_addicted],
                            tick_labels=['Addicted', 'Non-addicted'],
                            patch_artist=True, widths=0.6)
        bp1['boxes'][0].set_facecolor('#e74c3c')
        bp1['boxes'][0].set_alpha(0.7)
        bp1['boxes'][1].set_facecolor('#2ecc71')
        bp1['boxes'][1].set_alpha(0.7)
        
        ax6_1.set_ylabel('β Standard Deviation (per agent)', fontsize=11)
        ax6_1.set_title('β Variability Comparison', fontsize=12, fontweight='bold')
        ax6_1.grid(True, alpha=0.3, axis='y')
        
        # 平均値を表示
        ax6_1.plot([1, 2], [np.mean(std_addicted), np.mean(std_non_addicted)], 
                  'D', markersize=10, color='orange', label='Mean', zorder=3)
        ax6_1.legend()
        
        # (2) 極端値使用率の比較
        ax6_2 = axes6[0, 1]
        extreme_rates_add = []
        extreme_rates_non = []
        for i, is_add in enumerate(addiction_array):
            agent_betas = beta_matrix[i]
            extreme_rate = (np.sum(np.isclose(agent_betas, 0.0)) + 
                          np.sum(np.isclose(agent_betas, 1.0))) / len(agent_betas) * 100
            if is_add:
                extreme_rates_add.append(extreme_rate)
            else:
                extreme_rates_non.append(extreme_rate)
        
        ax6_2.hist(extreme_rates_add, bins=20, alpha=0.6, label='Addicted',
                  color='#e74c3c', density=True)
        ax6_2.hist(extreme_rates_non, bins=20, alpha=0.6, label='Non-addicted',
                  color='#2ecc71', density=True)
        ax6_2.set_xlabel('Extreme Value Usage (%)', fontsize=11)
        ax6_2.set_ylabel('Density', fontsize=11)
        ax6_2.set_title('Extreme Strategy Usage Distribution', fontsize=12, fontweight='bold')
        ax6_2.legend()
        ax6_2.grid(True, alpha=0.3)
        ax6_2.axvline(np.mean(extreme_rates_add), color='#e74c3c', linestyle='--', linewidth=2)
        ax6_2.axvline(np.mean(extreme_rates_non), color='#2ecc71', linestyle='--', linewidth=2)
        
        # (3) 各βの使用率差（棒グラフ）
        ax6_3 = axes6[0, 2]
        freq_diff = []
        for beta_val in BETA_VALUES:
            freq_add = np.sum(np.isclose(beta_addicted, beta_val)) / len(beta_addicted) * 100
            freq_non = np.sum(np.isclose(beta_non_addicted, beta_val)) / len(beta_non_addicted) * 100
            freq_diff.append(freq_add - freq_non)
        
        colors_diff = ['#e74c3c' if d > 0 else '#2ecc71' for d in freq_diff]
        bars_diff = ax6_3.bar(range(len(BETA_VALUES)), freq_diff, color=colors_diff, alpha=0.7)
        ax6_3.axhline(0, color='black', linewidth=1)
        ax6_3.set_xlabel('β Value', fontsize=11)
        ax6_3.set_ylabel('Usage Difference (%)', fontsize=11)
        ax6_3.set_title('β Usage: Addicted - Non-addicted', fontsize=12, fontweight='bold')
        ax6_3.set_xticks(range(len(BETA_VALUES)))
        ax6_3.set_xticklabels([f'{b:.1f}' for b in BETA_VALUES])
        ax6_3.grid(True, alpha=0.3, axis='y')
        
        # 値をバーに表示
        for i, (bar, diff) in enumerate(zip(bars_diff, freq_diff)):
            height = bar.get_height()
            ax6_3.text(bar.get_x() + bar.get_width()/2, height,
                      f'{diff:+.1f}%', ha='center', 
                      va='bottom' if height > 0 else 'top', fontsize=9)
        
        # (4) 時系列でのβ平均（依存群vs非依存群）
        ax6_4 = axes6[1, 0]
        
        # 各時刻での平均を計算
        beta_add_mean_t = np.mean(beta_addicted_by_phase, axis=0)
        beta_non_mean_t = np.mean(beta_non_addicted_by_phase, axis=0)
        beta_add_std_t = np.std(beta_addicted_by_phase, axis=0)
        beta_non_std_t = np.std(beta_non_addicted_by_phase, axis=0)
        
        ax6_4.plot(step_array, beta_add_mean_t, '-', color='#e74c3c', linewidth=2, 
                  label='Addicted', alpha=0.8)
        ax6_4.fill_between(step_array, beta_add_mean_t - beta_add_std_t, 
                          beta_add_mean_t + beta_add_std_t,
                          color='#e74c3c', alpha=0.2)
        
        ax6_4.plot(step_array, beta_non_mean_t, '-', color='#2ecc71', linewidth=2,
                  label='Non-addicted', alpha=0.8)
        ax6_4.fill_between(step_array, beta_non_mean_t - beta_non_std_t,
                          beta_non_mean_t + beta_non_std_t,
                          color='#2ecc71', alpha=0.2)
        
        # フェーズ境界
        phase_boundary = np.where(np.diff(phase_mean) != 0)[0]
        for boundary in phase_boundary:
            ax6_4.axvline(step_array[boundary], color='gray', linestyle='--', alpha=0.5)
        
        ax6_4.set_xlabel('Step', fontsize=11)
        ax6_4.set_ylabel('Mean β Value', fontsize=11)
        ax6_4.set_title('β Dynamics: Addicted vs Non-addicted', fontsize=12, fontweight='bold')
        ax6_4.set_ylim(-0.05, 1.05)
        ax6_4.legend()
        ax6_4.grid(True, alpha=0.3)
        
        # (5) 中間値使用率の比較
        ax6_5 = axes6[1, 1]
        middle_rates_add = []
        middle_rates_non = []
        for i, is_add in enumerate(addiction_array):
            agent_betas = beta_matrix[i]
            # β=0.2, 0.4, 0.6, 0.8の使用率
            middle_count = 0
            for beta_val in [0.2, 0.4, 0.6, 0.8]:
                middle_count += np.sum(np.isclose(agent_betas, beta_val))
            middle_rate = middle_count / len(agent_betas) * 100
            if is_add:
                middle_rates_add.append(middle_rate)
            else:
                middle_rates_non.append(middle_rate)
        
        bp2 = ax6_5.boxplot([middle_rates_add, middle_rates_non],
                            tick_labels=['Addicted', 'Non-addicted'],
                            patch_artist=True, widths=0.6)
        bp2['boxes'][0].set_facecolor('#e74c3c')
        bp2['boxes'][0].set_alpha(0.7)
        bp2['boxes'][1].set_facecolor('#2ecc71')
        bp2['boxes'][1].set_alpha(0.7)
        
        ax6_5.set_ylabel('Middle Values Usage (%)', fontsize=11)
        ax6_5.set_title('Balanced Strategy Usage (β=0.2-0.8)', fontsize=12, fontweight='bold')
        ax6_5.grid(True, alpha=0.3, axis='y')
        
        # 平均値を表示
        ax6_5.plot([1, 2], [np.mean(middle_rates_add), np.mean(middle_rates_non)],
                  'D', markersize=10, color='orange', label='Mean', zorder=3)
        ax6_5.legend()
        
        # (6) 相関分析：β平均値と依存確率
        ax6_6 = axes6[1, 2]
        
        # 各エージェントの平均β値を計算
        agent_mean_betas = [np.mean(beta_matrix[i]) for i in range(len(beta_matrix))]
        
        # β値を10区間に分割
        beta_bins = np.linspace(0, 1, 11)
        addiction_rates_by_beta = []
        bin_centers = []
        bin_counts = []
        
        for i in range(len(beta_bins) - 1):
            bin_low = beta_bins[i]
            bin_high = beta_bins[i + 1]
            mask = (np.array(agent_mean_betas) >= bin_low) & (np.array(agent_mean_betas) < bin_high)
            if i == len(beta_bins) - 2:  # 最後のビンは上限を含む
                mask = (np.array(agent_mean_betas) >= bin_low) & (np.array(agent_mean_betas) <= bin_high)
            
            if np.sum(mask) > 0:
                addiction_rate = np.sum(np.array(beta_stats.addiction_status)[mask]) / np.sum(mask) * 100
                addiction_rates_by_beta.append(addiction_rate)
                bin_centers.append((bin_low + bin_high) / 2)
                bin_counts.append(np.sum(mask))
        
        ax6_6.plot(bin_centers, addiction_rates_by_beta, 'o-', linewidth=2, markersize=8,
                  color='#9b59b6')
        ax6_6.set_xlabel('Mean β Value (binned)', fontsize=11)
        ax6_6.set_ylabel('Addiction Rate (%)', fontsize=11)
        ax6_6.set_title('Addiction Rate by Mean β Value', fontsize=12, fontweight='bold')
        ax6_6.grid(True, alpha=0.3)
        ax6_6.set_xlim(0, 1)
        
        # サンプル数を注釈
        for x, y, count in zip(bin_centers, addiction_rates_by_beta, bin_counts):
            ax6_6.annotate(f'n={count}', (x, y), textcoords='offset points',
                          xytext=(0, 5), ha='center', fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_detailed_analysis.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_prefix}_detailed_analysis.png")
        plt.close()
        
        # ===== Figure 7: 依存/非依存エージェントのβ選択推移（各β値の使用率の時系列、移動平均版） =====
        fig7, axes7 = plt.subplots(2, 1, figsize=(16, 10))
        
        # 時間窓の設定（移動平均のウィンドウサイズ）
        window_size = 50  # 50ステップごとの移動平均
        
        # ウィンドウの中心位置を計算
        n_windows = len(step_array) // window_size
        window_centers = []
        
        # 各β値の使用率を時間窓ごとに計算
        beta_usage_addicted = {beta: [] for beta in BETA_VALUES}
        beta_usage_non_addicted = {beta: [] for beta in BETA_VALUES}
        
        n_addicted_agents = np.sum(addiction_array)
        n_non_addicted_agents = len(addiction_array) - n_addicted_agents
        
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, len(step_array))
            window_centers.append(step_array[start_idx + (end_idx - start_idx) // 2])
            
            # 依存群: この時間窓でのβ値を収集
            window_beta_addicted = []
            for agent_idx, is_addicted in enumerate(addiction_array):
                if is_addicted:
                    window_beta_addicted.extend(beta_matrix[agent_idx, start_idx:end_idx])
            
            # 非依存群: この時間窓でのβ値を収集
            window_beta_non_addicted = []
            for agent_idx, is_addicted in enumerate(addiction_array):
                if not is_addicted:
                    window_beta_non_addicted.extend(beta_matrix[agent_idx, start_idx:end_idx])
            
            # 各β値の使用率を計算（依存群）
            if len(window_beta_addicted) > 0:
                for beta in BETA_VALUES:
                    usage_rate = np.sum(np.isclose(window_beta_addicted, beta)) / len(window_beta_addicted) * 100
                    beta_usage_addicted[beta].append(usage_rate)
            else:
                for beta in BETA_VALUES:
                    beta_usage_addicted[beta].append(0)
            
            # 各β値の使用率を計算（非依存群）
            if len(window_beta_non_addicted) > 0:
                for beta in BETA_VALUES:
                    usage_rate = np.sum(np.isclose(window_beta_non_addicted, beta)) / len(window_beta_non_addicted) * 100
                    beta_usage_non_addicted[beta].append(usage_rate)
            else:
                for beta in BETA_VALUES:
                    beta_usage_non_addicted[beta].append(0)
        
        # カラーマップ
        colors_beta = plt.cm.viridis(np.linspace(0, 1, len(BETA_VALUES)))
        
        # フェーズ境界
        phase_boundary = np.where(np.diff(phase_mean) != 0)[0]
        
        # Addicted群のプロット（上段）
        ax7_1 = axes7[0]
        
        for beta_idx, beta_val in enumerate(BETA_VALUES):
            ax7_1.plot(window_centers, beta_usage_addicted[beta_val], 
                      'o-', label=f'β={beta_val:.1f}', linewidth=2, markersize=4,
                      color=colors_beta[beta_idx], alpha=0.8)
        
        # フェーズ境界を描画
        for boundary in phase_boundary:
            ax7_1.axvline(step_array[boundary], color='gray', linestyle='--', 
                         linewidth=1.5, alpha=0.5)
        
        ax7_1.set_ylabel('Usage Rate (%)', fontsize=12)
        ax7_1.set_title(f'時間窓別β使用率: 依存群 (n={n_addicted_agents} agents)', 
                       fontsize=14, fontweight='bold')
        ax7_1.legend(loc='best', fontsize=10, ncol=3)
        ax7_1.grid(True, alpha=0.3)
        
        # Non-addicted群のプロット（下段）
        ax7_2 = axes7[1]
        
        for beta_idx, beta_val in enumerate(BETA_VALUES):
            ax7_2.plot(window_centers, beta_usage_non_addicted[beta_val], 
                      'o-', label=f'β={beta_val:.1f}', linewidth=2, markersize=4,
                      color=colors_beta[beta_idx], alpha=0.8)
        
        # フェーズ境界を描画
        for boundary in phase_boundary:
            ax7_2.axvline(step_array[boundary], color='gray', linestyle='--',
                         linewidth=1.5, alpha=0.5)
        
        ax7_2.set_xlabel('Step (window center)', fontsize=12)
        ax7_2.set_ylabel('Usage Rate (%)', fontsize=12)
        ax7_2.set_title(f'時間窓別β使用率: 非依存群 (n={n_non_addicted_agents} agents)', 
                       fontsize=14, fontweight='bold')
        ax7_2.legend(loc='best', fontsize=10, ncol=3)
        ax7_2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_usage_timeseries.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_prefix}_usage_timeseries.png")
        plt.close()
    
    print("\n" + "="*60)
    print("β SELECTION STATISTICS")
    print("="*60)
    print(f"Total steps analyzed: {len(beta_all)} ({beta_stats.num_agents} agents)")
    print(f"  Addicted: {n_addicted} agents")
    print(f"  Non-addicted: {n_non_addicted} agents")
    
    print(f"\nOverall β statistics:")
    print(f"  Mean: {np.mean(beta_all):.3f} ± {np.std(beta_all):.3f}")
    print(f"  Median: {np.median(beta_all):.3f}")
    print(f"  Min: {np.min(beta_all):.3f}, Max: {np.max(beta_all):.3f}")
    
    if n_addicted > 0 and n_non_addicted > 0:
        print(f"\n--- Addicted agents (n={n_addicted}) ---")
        print(f"  Mean β: {np.mean(beta_addicted):.3f} ± {np.std(beta_addicted):.3f}")
        print(f"  Median β: {np.median(beta_addicted):.3f}")
        # 極端値の使用率
        extreme_add = np.sum(np.isclose(beta_addicted, 0.0)) + np.sum(np.isclose(beta_addicted, 1.0))
        print(f"  Extreme values (β=0.0 or 1.0): {extreme_add / len(beta_addicted) * 100:.2f}%")
        print(f"\nβ selection frequency (Addicted):")
        for beta_val in BETA_VALUES:
            count = np.sum(np.isclose(beta_addicted, beta_val))
            percentage = 100.0 * count / len(beta_addicted)
            print(f"  β={beta_val:.1f}: {count:6d} times ({percentage:5.2f}%)")
        
        print(f"\n--- Non-addicted agents (n={n_non_addicted}) ---")
        print(f"  Mean β: {np.mean(beta_non_addicted):.3f} ± {np.std(beta_non_addicted):.3f}")
        print(f"  Median β: {np.median(beta_non_addicted):.3f}")
        # 極端値の使用率
        extreme_non = np.sum(np.isclose(beta_non_addicted, 0.0)) + np.sum(np.isclose(beta_non_addicted, 1.0))
        print(f"  Extreme values (β=0.0 or 1.0): {extreme_non / len(beta_non_addicted) * 100:.2f}%")
        print(f"\nβ selection frequency (Non-addicted):")
        for beta_val in BETA_VALUES:
            count = np.sum(np.isclose(beta_non_addicted, beta_val))
            percentage = 100.0 * count / len(beta_non_addicted)
            print(f"  β={beta_val:.1f}: {count:6d} times ({percentage:5.2f}%)")
        
        # 差の統計
        print(f"\n--- Comparison (Addicted vs Non-addicted) ---")
        print(f"  Mean difference: {np.mean(beta_addicted) - np.mean(beta_non_addicted):.3f}")
        print(f"  Extreme value difference: {(extreme_add / len(beta_addicted) - extreme_non / len(beta_non_addicted)) * 100:.2f}%")
    else:
        print(f"\nβ selection frequency:")
        for beta_val in BETA_VALUES:
            count = np.sum(np.isclose(beta_all, beta_val))
            percentage = 100.0 * count / len(beta_all)
            print(f"  β={beta_val:.1f}: {count:6d} times ({percentage:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Fast Hybrid RL Addiction Simulation with Beta Learning")
    parser.add_argument("--num-agents", type=int, default=900, help="Agents per seed")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)",
    )
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
        help="Path to write debug episode transitions as CSV (default: ./debug_episode_learn_beta.csv)",
    )
    parser.add_argument(
        "--plot-beta",
        action="store_true",
        help="Generate β selection analysis plots",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="beta_analysis",
        help="Prefix for output plot files (default: beta_analysis)",
    )
    parser.add_argument(
        "--fixed-beta",
        type=float,
        default=None,
        choices=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Fix beta to a specific value instead of learning (choices: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)",
    )
    parser.add_argument(
        "--save-occupancy-data",
        action="store_true",
        help="Collect and save time-series state occupancy data",
    )
    parser.add_argument(
        "--occupancy-output",
        type=str,
        default=None,
        help="Output file for occupancy data (JSON format, default: beta_X_episodes.json)",
    )
    parser.add_argument(
        "--time-bin-size",
        type=int,
        default=2000,
        help="Time bin size for occupancy data aggregation (default: 2000)",
    )
    args = parser.parse_args()

    print(f"Simulation Start: Agents={args.num_agents}, Runs={args.num_runs}, Seed={args.seed}")
    if args.fixed_beta is not None:
        print(f"Beta Mode: Fixed (β={args.fixed_beta})")
    else:
        print(f"Beta Mode: Learning (adaptive β selection)")
    print("-" * 60)

    def log_progress(seed, agent, n_agents, n_runs, p_idx, step, length):
        # 進捗表示 (間引いて表示)
        if seed == 0 and agent == 0 and step % 100 == 0:
            p_name = PHASES[p_idx][0]
            print(f"Phase: {p_name} Step: {step}/{length}")

    # デバッグログのファイル名を生成
    debug_txt_path = None
    if args.debug_episode:
        debug_txt_path = "debug_log_learn_beta.txt"

    overall_rate, run_rates, overall_reversal_rate, run_reversal_rates, state_visit_stats, beta_stats, occupancy_data = simulate(
        args.num_agents,
        args.num_runs,
        args.seed,
        log_progress,
        mb_forget=args.mb_forget,
        debug_episode=args.debug_episode,
        debug_csv_path=args.debug_csv,
        debug_txt_path=debug_txt_path,
        collect_beta_stats=args.plot_beta,
        fixed_beta=args.fixed_beta,
        collect_occupancy_data=args.save_occupancy_data,
        time_bin_size=args.time_bin_size,
    )
    
    mean_addiction_percent = np.mean(run_rates) * 100.0
    mean_reversal_percent = np.mean(run_reversal_rates) * 100.0
    print(f"Result: Addiction Rate={mean_addiction_percent:.2f}%, Reversal Adaptation Rate={mean_reversal_percent:.2f}% (mean over runs)")
    print("-" * 60)

    # 結果をファイルに出力（並列実行用）
    if args.output is not None:
        import json
        output_data = {
            "addiction_rate": mean_addiction_percent,
            "reversal_adaptation_rate": mean_reversal_percent,
            "run_rates": [r * 100.0 for r in run_rates],
            "run_reversal_rates": [r * 100.0 for r in run_reversal_rates],
            "addiction_state_ratios": state_visit_stats["addiction_state_ratios"],
            "reversal_state_ratios": state_visit_stats["reversal_state_ratios"],
            "addiction_total_steps": state_visit_stats["addiction_total_steps"],
            "reversal_total_steps": state_visit_stats["reversal_total_steps"],
            "num_agents": args.num_agents,
            "num_runs": args.num_runs,
            "seed": args.seed,
            "mb_forget": args.mb_forget,
            "beta_learning": args.fixed_beta is None,
            "fixed_beta": args.fixed_beta,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")
        
        # occupancy dataも別ファイルに保存
        if args.save_occupancy_data and occupancy_data is not None:
            if args.occupancy_output:
                occupancy_output_path = args.occupancy_output
            else:
                # デフォルトのファイル名を生成
                if args.fixed_beta is not None:
                    beta_str = f"{args.fixed_beta:.1f}" if args.fixed_beta != int(args.fixed_beta) else str(int(args.fixed_beta))
                    occupancy_output_path = f"beta_{beta_str}_episodes.json"
                else:
                    occupancy_output_path = "beta_learning_episodes.json"
            
            occupancy_output_data = {
                "beta": args.fixed_beta,
                "beta_mode": "fixed" if args.fixed_beta is not None else "learning",
                "n_runs": args.num_runs,
                "time_bins": occupancy_data['time_bins'],
                "time_bin_size": occupancy_data['time_bin_size'],
                "total_steps": occupancy_data['total_steps'],
                "states": occupancy_data['states']
            }
            
            with open(occupancy_output_path, 'w') as f:
                json.dump(occupancy_output_data, f, indent=2)
            print(f"Occupancy data saved to {occupancy_output_path}")
        
        return

    # 簡易統計表示
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Addiction Rate: {mean_addiction_percent:.2f}%")
    print(f"Standard Error (Addiction): {np.std(run_rates) * 100.0 / np.sqrt(len(run_rates)):.2f}%")
    print(f"Min Rate (Addiction): {np.min(run_rates) * 100.0:.2f}%")
    print(f"Max Rate (Addiction): {np.max(run_rates) * 100.0:.2f}%")
    print("-" * 40)
    print(f"Overall Reversal Adaptation Rate: {mean_reversal_percent:.2f}%")
    print(f"Standard Error (Reversal): {np.std(run_reversal_rates) * 100.0 / np.sqrt(len(run_reversal_rates)):.2f}%")
    print(f"Min Rate (Reversal): {np.min(run_reversal_rates) * 100.0:.2f}%")
    print(f"Max Rate (Reversal): {np.max(run_reversal_rates) * 100.0:.2f}%")
    
    # β選択の分析グラフを生成
    if args.plot_beta and beta_stats is not None:
        print("\n" + "="*60)
        print("Generating β selection analysis plots...")
        print("="*60)
        plot_beta_analysis(beta_stats, output_prefix=args.plot_prefix)

if __name__ == "__main__":
    main()