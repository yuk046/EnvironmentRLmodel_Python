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

# 遷移確率テーブル (f1: Pre-drug, f2: Addiction)
NEUTRAL_MOVE_SUCCESS = [0.99, 0.99] #隣り合う状態遷移
NEUTRAL_SKIP_SUCCESS = [0.0001, 0.0001] #離れた状態遷移
# アフターエフェクト区間の状態遷移
AFTER_AG_EXIT = [0.001, 0.001]
AFTER_AS_EXIT = [0.001, 0.001]
AW_FORWARD = [0.4995, 0.4995]
AW_BACKWARD = [0.4995, 0.4995]
# 状態15(14)での特別遷移
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
PHASE_REWARD_TABLE = np.zeros((len(PHASES), NUM_STATES)) #2x22
PHASE_DRUG_REWARDS = np.array([phase[2] for phase in PHASES], dtype=np.float64) #[0.0, 10.0]
for idx in range(len(PHASES)):
    PHASE_REWARD_TABLE[idx, list(STATE_AFTEREFFECTS)] = AFTER_PHASE_REWARDS[idx] #アフターエフェクト区間の状態に罰を定義


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
            # 自己遷移を除外: s == s_tilde の場合はスキップ
            if s == s_tilde:
                continue
                
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

        # 状態0でagした際
        if current_state == STATE_GOAL and action == ACTION_GOAL and next_state == STATE_START:
            r += R_G
        # 状態6でadした際
        elif (current_state == NEUTRAL_MAX and action == ACTION_DRUG and next_state == STATE_DRUG):
            r += PHASE_DRUG_REWARDS[phase_idx]
        # 状態７から７への移動
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

    def select_action(self, state: int) -> int:
        # MB Planning (JIT function call)
        self._plan_q_values()
        
        # ε-Greedy
        if self.rng.random() < EPSILON:
            return int(self.rng.integers(NUM_ACTIONS))
            
        # Hybrid Q-value
        q_mix = self.beta * self.q_mb[state] + (1.0 - self.beta) * self.q_mf[state]
        
        # 最も価値が高い行動を選ぶ (Argmax)
        max_val = np.max(q_mix)
        # 同着1位が複数ある場合に備えて、最大値に近い行動を全てリストアップ
        best_actions = np.flatnonzero(np.isclose(q_mix, max_val,rtol=1e-08, atol=1e-12))
        # 同着の中からランダムに1つ選んで返す
        return int(self.rng.choice(best_actions))

    # モデルの学習機構
    def observe(self, state: int, action: int, reward: float, next_state: int, phase_idx: int):
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
    # ラン毎の依存者数を記録（後で run ごとの率に変換）
    run_addictions = [0 for _ in range(num_runs)]
    
    # テキストログファイルを開く (追記モードではなく新規作成)
    log_file = None
    if debug_episode and debug_txt_path:
        log_file = open(debug_txt_path, "w", encoding="utf-8")

    try:
        for seed_idx in range(num_runs):
            for agent_idx in range(num_agents):
                # 固定ペアリングシード: (run, agent) ごとに一意のシードを割り当て
                # これにより同じ (run, agent) 番号は全ての beta 値で同じ初期 RNG を使います
                seed_for_agent = base_seed + seed_idx * num_agents + agent_idx
                rng = np.random.default_rng(seed_for_agent)

                env = AddictionEnvironment(rng)
                agent = HybridAgent(beta, rng, mb_forget=mb_forget)
                
                state = env.reset()
                counts = PhaseResult()
                
                # フェーズ実行
                for phase_idx, (_, length, _) in enumerate(PHASES):
                    # フェーズ開始時にSTATE_STARTにリセット
                    # if phase_idx > 0:
                    #     state = env.reset()
                    
                    report_points = {1, length // 2, length}
                    
                    for step_in_phase in range(length):
                        # Progress Log
                        if progress_cb and (step_in_phase + 1) in report_points:
                            progress_cb(beta, seed_idx, agent_idx, num_agents, num_runs, phase_idx, step_in_phase + 1, length)
                        
                        # 行動選択前にQ値を取得したいが、select_action内でMBのPlanningが走るため
                        # select_action後に取得すると、そのステップでのPlanning結果が反映された状態になる
                        # ここでは「行動選択に使われたQ値」に近いものを表示するため、select_action直後の値を参照する
                        
                        # 行動選択(MBはPlaning)
                        action = agent.select_action(state)
                        
                        # Debug出力用にQ値を取得 (現在の状態 state における全行動のQ値)
                        # debug_episodeが有効な場合のみコピーを行う
                        current_q_mf = None
                        current_q_mb = None
                        current_q_mix = None
                        if debug_episode and seed_idx == 0 and agent_idx == 0:
                            current_q_mf = agent.q_mf[state].copy()
                            current_q_mb = agent.q_mb[state].copy()
                            current_q_mix = beta * current_q_mb + (1.0 - beta) * current_q_mf
                        
                        # 行動、学習
                        next_state, reward = env.step(action, phase_idx)
                        agent.observe(state, action, reward, next_state, phase_idx)

                        if debug_episode and seed_idx == 0 and agent_idx == 0:
                            # ログメッセージの構築
                            q_mf_str = ", ".join([f"{x:.6f}" for x in current_q_mf])
                            q_mb_str = ", ".join([f"{x:.6f}" for x in current_q_mb])
                            q_mix_str = ", ".join([f"{x:.6f}" for x in current_q_mix])
                            # 現在の状態と行動に紐づくMBモデル（遷移カウントと累積報酬）も併せて出力
                            curr_model_counts = agent.model_counts[state, action].copy()
                            curr_model_rewards = agent.model_rewards[state, action].copy()
                            model_counts_str = ", ".join([f"{x:.3f}" for x in curr_model_counts])
                            model_rewards_str = ", ".join([f"{x:.3f}" for x in curr_model_rewards])
                            
                            log_lines = [
                                f"[DEBUG] Beta={beta:.1f} Phase={PHASES[phase_idx][0]} Step={step_in_phase+1}",
                                f"  State: {state} -> Action: {ACTION_NAMES.get(action, str(action))} -> Next: {next_state} (Reward: {reward})",
                                f"  Q_MF:  [{q_mf_str}]",
                                f"  Q_MB:  [{q_mb_str}]",
                                f"  Q_MIX: [{q_mix_str}]",
                                f"  MODEL_COUNTS(s,a,*):  [{model_counts_str}]",
                                f"  MODEL_REWARDS(s,a,*): [{model_rewards_str}]",
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
                                
                        state = next_state
                
                # 依存判定
                if counts.drug_choices > counts.healthy_choices:
                    addictions += 1
                    run_addictions[seed_idx] += 1

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

    # run ごとの率を計算して返す
    run_rates = [cnt / num_agents for cnt in run_addictions]
    overall_rate = addictions / total_agents
    return overall_rate, run_rates


def main():
    parser = argparse.ArgumentParser(description="Fast Hybrid RL Addiction Simulation")
    parser.add_argument("--num-agents", type=int, default=300, help="Agents per seed")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
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

    for beta in beta_values:
        # デバッグログのファイル名を生成
        debug_txt_path = None
        if args.debug_episode:
            debug_txt_path = f"debug_log_beta_{beta:.2f}.txt"

        overall_rate, run_rates = simulate(
            beta,
            args.num_agents,
            args.num_runs,
            args.seed,  # use the same base seed for paired comparisons across betas
            log_progress,
            mb_forget=args.mb_forget,
            debug_episode=args.debug_episode,
            debug_csv_path=args.debug_csv,
            debug_txt_path=debug_txt_path,
        )
        # keep base seed unchanged so pairing holds across betas
        # run_rates: list of length num_runs with rates in [0,1]
        mean_percent = np.mean(run_rates) * 100.0
        rates.append(mean_percent)
        # store run-level rates (as percent) for plotting error bars
        # convert to percent and keep in a separate list parallel to beta_values
        if 'all_run_rates' not in locals():
            all_run_rates = []
        all_run_rates.append([r * 100.0 for r in run_rates])
        print(f"Result: Beta={beta:.1f} => Addiction Rate={mean_percent:.2f}% (mean over runs)")
        print("-" * 60)

    # グラフ描画: run毎の率から平均 ± SEM を描画し、平均線を太くする
    means = np.array(rates)
    # all_run_rates: list[num_betas][num_runs]
    run_matrix = np.array(all_run_rates)  # shape: (num_betas, num_runs)
    sems = np.std(run_matrix, axis=1, ddof=1) / np.sqrt(run_matrix.shape[1])

    plt.figure(figsize=(8, 5))
    # 薄い点で各 run の値をプロット（軽く横にジッタ）
    for i, vals in enumerate(run_matrix):
        x = np.full(len(vals), beta_values[i]) + (np.random.random(len(vals)) - 0.5) * 0.01
        plt.scatter(x, vals, color='gray', alpha=0.25, s=10)

    # 平均 ± SEM を太い色強い線で描画
    plt.errorbar(beta_values, means, yerr=sems, fmt='-o', color='C0', ecolor='C0', elinewidth=1.5, capsize=4, linewidth=3, markersize=6, label='Mean ± SEM')
    plt.title(f"Transition to Addiction (N={args.num_agents * args.num_runs})")
    plt.xlabel("Degree of MB Control (Beta)")
    plt.ylabel("Addiction Rate (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # 画像保存または表示
    # plt.savefig("addiction_result.png")
    plt.show()

if __name__ == "__main__":
    main()