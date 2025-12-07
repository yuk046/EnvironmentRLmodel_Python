package addiction.rl;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * 定数・環境設定 (論文準拠)
 */
public final class Constants {
    private Constants() {}

    // ==========================================
    // 状態・アクション数
    // ==========================================
    public static final int NUM_STATES = 22;
    public static final int NUM_ACTIONS = 9;

    // 状態定義
    public static final int STATE_GOAL = 0;        // Goal state
    public static final int STATE_GOAL_ENTRY = 1;  // Neutral state leading to goal
    public static final int STATE_START = 3;       // Start state (S0)
    public static final int STATE_DRUG = 7;        // Drug state
    public static final int STATE_AW_SPECIAL = 14; // Special withdrawal state

    // 状態セットの定義
    public static final int[] STATE_NEUTRALS = {1, 2, 3, 4, 5, 6};
    public static final Set<Integer> STATE_NEUTRAL_SET = new HashSet<>();
    public static final int NEUTRAL_MAX = 6;

    public static final int[] STATE_AFTEREFFECTS;
    public static final Set<Integer> STATE_AFTER_SET = new HashSet<>();
    public static final int AFTER_MAX = 21;

    static {
        for (int s : STATE_NEUTRALS) {
            STATE_NEUTRAL_SET.add(s);
        }
        STATE_AFTEREFFECTS = new int[14];
        for (int i = 0; i < 14; i++) {
            STATE_AFTEREFFECTS[i] = 8 + i;
            STATE_AFTER_SET.add(8 + i);
        }
    }

    public static int aftereffectForwardState(int state) {
        if (state >= AFTER_MAX) {
            return STATE_DRUG;
        }
        return state + 1;
    }

    public static int aftereffectBackwardState(int state) {
        if (state <= STATE_DRUG) {
            return AFTER_MAX;
        }
        return state - 1;
    }

    // ==========================================
    // ハイパーパラメータ
    // ==========================================
    public static final double DISCOUNT = 0.9;
    public static final double ALPHA_MF = 0.05;
    public static final double MODEL_DECAY = 0.01;   // モデルカウントの減衰率 (毎ステップ)
    public static final double INITIAL_TRANSITION_COUNT = 5.0;  // 新規遷移観測時の初期カウント
    public static final double EPSILON = 0.1;        // 探索率
    public static final int N_PRIORITIZED_SWEEPS = 50;
    public static final double T_MB = 1.0;           // Softmax temperature for planning

    // ==========================================
    // 報酬設定
    // ==========================================
    public static final double R_G = 1.0;            // Goal reward
    public static final double R_P = -4.0;           // Punishment (shock)
    public static final double R_SKIP_LONG = -0.3;   // Cost of skipping states
    public static final double[] AFTER_PHASE_REWARDS = {-0.3, -1.2}; // Pre-drug / Addiction phase rewards in aftereffect

    // ==========================================
    // アクション定義
    // ==========================================
    public static final int ACTION_AS2 = 0;
    public static final int ACTION_AS3 = 1;
    public static final int ACTION_AS4 = 2;
    public static final int ACTION_AS5 = 3;
    public static final int ACTION_AS6 = 4;
    public static final int ACTION_AS7 = 5;
    public static final int ACTION_GOAL = 6;
    public static final int ACTION_DRUG = 7;
    public static final int ACTION_AW = 8;

    public static final Map<Integer, Integer> AS_ACTION_TARGETS = new HashMap<>();
    public static final Map<Integer, String> ACTION_NAMES = new HashMap<>();

    static {
        AS_ACTION_TARGETS.put(ACTION_AS2, 1);
        AS_ACTION_TARGETS.put(ACTION_AS3, 2);
        AS_ACTION_TARGETS.put(ACTION_AS4, 3);
        AS_ACTION_TARGETS.put(ACTION_AS5, 4);
        AS_ACTION_TARGETS.put(ACTION_AS6, 5);
        AS_ACTION_TARGETS.put(ACTION_AS7, 6);

        ACTION_NAMES.put(ACTION_AS2, "as2");
        ACTION_NAMES.put(ACTION_AS3, "as3");
        ACTION_NAMES.put(ACTION_AS4, "as4");
        ACTION_NAMES.put(ACTION_AS5, "as5");
        ACTION_NAMES.put(ACTION_AS6, "as6");
        ACTION_NAMES.put(ACTION_AS7, "as7");
        ACTION_NAMES.put(ACTION_GOAL, "ag");
        ACTION_NAMES.put(ACTION_DRUG, "ad");
        ACTION_NAMES.put(ACTION_AW, "aw");
    }

    // ==========================================
    // 遷移確率テーブル (f1: Pre-drug, f2: Addiction)
    // ==========================================
    public static final double[] NEUTRAL_MOVE_SUCCESS = {0.99, 0.99};
    public static final double[] NEUTRAL_SKIP_SUCCESS = {0.0001, 0.0001};
    public static final double[] AFTER_AG_EXIT = {0.001, 0.001};
    public static final double[] AFTER_AS_EXIT = {0.001, 0.001};
    public static final double[] AW_FORWARD = {0.4995, 0.4995};
    public static final double[] AW_BACKWARD = {0.4995, 0.4995};
    public static final double[] AW_SPECIAL_MOVE = {0.2, 0.2};
    public static final double[] AW_SPECIAL_EXIT = {0.6, 0.6};
    public static final double[] AD_FORWARD = {0.745, 0.745};
    public static final double[] AD_BACKWARD = {0.245, 0.245};

    // ==========================================
    // フェーズ定義
    // ==========================================
    public static final Phase[] PHASES = {
        new Phase("pre-drug", 50, 0.0),
        new Phase("addiction", 1000, 10.0)
    };

    // 報酬テーブルの事前構築
    public static final double[][] PHASE_REWARD_TABLE = new double[PHASES.length][NUM_STATES];
    public static final double[] PHASE_DRUG_REWARDS = new double[PHASES.length];

    static {
        for (int idx = 0; idx < PHASES.length; idx++) {
            PHASE_DRUG_REWARDS[idx] = PHASES[idx].drugReward;
            for (int s : STATE_AFTEREFFECTS) {
                PHASE_REWARD_TABLE[idx][s] = AFTER_PHASE_REWARDS[idx];
            }
        }
    }

    /**
     * フェーズ情報を保持するクラス
     */
    public static class Phase {
        public final String name;
        public final int steps;
        public final double drugReward;

        public Phase(String name, int steps, double drugReward) {
            this.name = name;
            this.steps = steps;
            this.drugReward = drugReward;
        }
    }
}
