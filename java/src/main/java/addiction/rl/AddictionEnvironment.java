package addiction.rl;

import java.util.Random;

import static addiction.rl.Constants.*;

/**
 * 依存症シミュレーション環境
 */
public class AddictionEnvironment {
    private int state;
    private final Random rng;

    public AddictionEnvironment(Random rng) {
        this.rng = rng;
        this.state = STATE_START;
    }

    public int reset() {
        this.state = STATE_START;
        return this.state;
    }

    public int getState() {
        return this.state;
    }

    /**
     * 環境のステップ実行
     * @param action 選択されたアクション
     * @param phaseIdx フェーズインデックス
     * @return [nextState, reward]
     */
    public StepResult step(int action, int phaseIdx) {
        int curr = this.state;
        int nextState = curr;

        if (STATE_NEUTRAL_SET.contains(curr)) {
            nextState = transitionNeutral(curr, action, phaseIdx);
        } else if (curr == STATE_DRUG || STATE_AFTER_SET.contains(curr)) {
            nextState = transitionAftereffect(curr, action, phaseIdx);
        } else if (curr == STATE_GOAL) {
            // GoalからはStartへ戻るか、Goalに留まるか(行動による)
            nextState = (action == ACTION_GOAL) ? STATE_START : STATE_GOAL;
        }
        // Start地点などはそのまま

        double reward = calculateReward(curr, action, nextState, phaseIdx);
        this.state = nextState;
        return new StepResult(nextState, reward);
    }

    private int transitionNeutral(int state, int action, int phaseIdx) {
        // AS行動 (空間移動)
        if (AS_ACTION_TARGETS.containsKey(action)) {
            int target = AS_ACTION_TARGETS.get(action);
            if (target == state) return state;

            int dist = Math.abs(target - state);
            double successProb = (dist == 1) ? NEUTRAL_MOVE_SUCCESS[phaseIdx] : NEUTRAL_SKIP_SUCCESS[phaseIdx];

            if (rng.nextDouble() < successProb) {
                return target;
            }
            return state;
        }

        if (action == ACTION_GOAL && state == STATE_GOAL_ENTRY) {
            return STATE_GOAL;
        }
        if (action == ACTION_DRUG && state == NEUTRAL_MAX) {
            return STATE_DRUG;
        }

        return state;
    }

    private int transitionAftereffect(int state, int action, int phaseIdx) {
        double roll = rng.nextDouble();

        // 離脱 (Exit) の判定
        if (action == ACTION_GOAL) {
            return (roll < AFTER_AG_EXIT[phaseIdx]) ? STATE_START : state;
        }
        if (AS_ACTION_TARGETS.containsKey(action)) {
            return (roll < AFTER_AS_EXIT[phaseIdx]) ? STATE_START : state;
        }

        // 内部移動 (AW/AD)
        if (action == ACTION_AW) {
            // Special State (分岐点)
            if (state == STATE_AW_SPECIAL) {
                double pMove = AW_SPECIAL_MOVE[phaseIdx] / 2.0;
                if (roll < pMove) {
                    return Math.max(STATE_DRUG, state - 1);
                }
                roll -= pMove;
                if (roll < pMove) {
                    return Math.min(AFTER_MAX, state + 1);
                }
                roll -= pMove;
                if (roll < AW_SPECIAL_EXIT[phaseIdx]) {
                    return STATE_START;
                }
                return state;
            }

            // Normal Aftereffect State
            if (roll < AW_FORWARD[phaseIdx]) {
                return aftereffectForwardState(state);
            }
            roll -= AW_FORWARD[phaseIdx];
            if (roll < AW_BACKWARD[phaseIdx]) {
                return aftereffectBackwardState(state);
            }
            // 残り0.1%でlocation 4 (STATE_START) へ抜ける
            return STATE_START;
        }

        if (action == ACTION_DRUG) {
            if (roll < AD_FORWARD[phaseIdx]) {
                return aftereffectForwardState(state);
            }
            roll -= AD_FORWARD[phaseIdx];
            if (roll < AD_BACKWARD[phaseIdx]) {
                return aftereffectBackwardState(state);
            }
            // 残り1%でlocation 4 (STATE_START) へ抜ける
            return STATE_START;
        }

        return state;
    }

    private double calculateReward(int currentState, int action, int nextState, int phaseIdx) {
        double r = PHASE_REWARD_TABLE[phaseIdx][nextState];

        if (currentState == STATE_GOAL && action == ACTION_GOAL && nextState == STATE_START) {
            r += R_G;
        } else if (currentState == NEUTRAL_MAX && action == ACTION_DRUG && nextState == STATE_DRUG) {
            r += PHASE_DRUG_REWARDS[phaseIdx];
        } else if (currentState == STATE_DRUG && nextState == STATE_DRUG) {
            r += AFTER_PHASE_REWARDS[phaseIdx];
        }

        // Neutralエリアでのロングジャンプ失敗コスト
        if (STATE_NEUTRAL_SET.contains(currentState) && STATE_NEUTRAL_SET.contains(nextState)
            && Math.abs(nextState - currentState) > 1) {
            r += R_SKIP_LONG;
        }

        // Drug/Aftereffect から Neutral/Goal への遷移 (離脱時の罰則)
        if ((currentState == STATE_DRUG || STATE_AFTER_SET.contains(currentState))
            && (STATE_NEUTRAL_SET.contains(nextState) || nextState == STATE_GOAL || nextState == STATE_GOAL_ENTRY)) {
            r += R_P;
        }

        return r;
    }

    /**
     * ステップ結果を保持するクラス
     */
    public static class StepResult {
        public final int nextState;
        public final double reward;

        public StepResult(int nextState, double reward) {
            this.nextState = nextState;
            this.reward = reward;
        }
    }
}
