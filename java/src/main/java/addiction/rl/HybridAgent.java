package addiction.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static addiction.rl.Constants.*;

/**
 * ハイブリッドエージェント (Model-Free + Model-Based)
 */
public class HybridAgent {
    private final double beta;
    private final Random rng;
    private final boolean mbForget;

    // Qテーブル (MF, MB)
    private final double[][] qMf;
    private final double[][] qMb;

    // メンタルモデル
    // modelCounts: [state][action][nextState] -> count
    private final double[][][] modelCounts;
    // modelRewards: [state][action][nextState] -> 遷移先依存の累積報酬 R(s,a,s')
    private final double[][][] modelRewards;
    private final double[][] modelVisits;

    public HybridAgent(double beta, Random rng, boolean mbForget) {
        this.beta = beta;
        this.rng = rng;
        this.mbForget = mbForget;

        // Qテーブル初期化
        this.qMf = new double[NUM_STATES][NUM_ACTIONS];
        this.qMb = new double[NUM_STATES][NUM_ACTIONS];

        // メンタルモデル初期化
        this.modelCounts = new double[NUM_STATES][NUM_ACTIONS][NUM_STATES];
        this.modelRewards = new double[NUM_STATES][NUM_ACTIONS][NUM_STATES];
        this.modelVisits = new double[NUM_STATES][NUM_ACTIONS];

        // 初期モデル: 全ての行動が自己遷移すると仮定 (論文準拠)
        for (int s = 0; s < NUM_STATES; s++) {
            for (int a = 0; a < NUM_ACTIONS; a++) {
                this.modelCounts[s][a][s] = 3.0;  // 自己遷移のカウント
                this.modelVisits[s][a] = 1.0;
            }
        }
    }

    public int selectAction(int state) {
        // MB Planning
        planQValues();

        // ε-Greedy
        if (rng.nextDouble() < EPSILON) {
            return rng.nextInt(NUM_ACTIONS);
        }

        // Hybrid Q-value
        double[] qMix = new double[NUM_ACTIONS];
        for (int a = 0; a < NUM_ACTIONS; a++) {
            qMix[a] = beta * qMb[state][a] + (1.0 - beta) * qMf[state][a];
        }

        // Argmax (tie-break random)
        double maxVal = qMix[0];
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (qMix[a] > maxVal) {
                maxVal = qMix[a];
            }
        }

        List<Integer> bestActions = new ArrayList<>();
        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (Math.abs(qMix[a] - maxVal) < 1e-12) {
                bestActions.add(a);
            }
        }

        return bestActions.get(rng.nextInt(bestActions.size()));
    }

    public void observe(int state, int action, double reward, int nextState, int phaseIdx) {
        // MF Update (Q-Learning)
        double maxNextQ = qMf[nextState][0];
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (qMf[nextState][a] > maxNextQ) {
                maxNextQ = qMf[nextState][a];
            }
        }
        double tdTarget = reward + DISCOUNT * maxNextQ;
        qMf[state][action] += ALPHA_MF * (tdTarget - qMf[state][action]);

        // MB Model Update (論文準拠)
        // 1. カウント減衰
        double decayFactor = 1.0 - MODEL_DECAY;
        for (int s = 0; s < NUM_STATES; s++) {
            for (int a = 0; a < NUM_ACTIONS; a++) {
                for (int ns = 0; ns < NUM_STATES; ns++) {
                    modelCounts[s][a][ns] *= decayFactor;
                    modelRewards[s][a][ns] *= decayFactor;
                }
                modelVisits[s][a] *= decayFactor;
            }
        }

        // 2. 新規遷移の検出と初期カウント設定
        if (state != nextState) {
            modelCounts[state][action][state] = 1.0;
            modelRewards[state][action][state] = 0.0;
        }

        if (modelCounts[state][action][nextState] < 0.5) {
            // 観測した遷移を 5.0 にセット
            modelCounts[state][action][nextState] = INITIAL_TRANSITION_COUNT;
            modelRewards[state][action][nextState] = reward * INITIAL_TRANSITION_COUNT;
        } else {
            modelCounts[state][action][nextState] += 1.0;
            modelRewards[state][action][nextState] += reward;
        }

        modelVisits[state][action] += 1.0;
    }

    private void planQValues() {
        if (mbForget) {
            // 完全忘却モード: 毎回MB推定をゼロから再構築
            for (int s = 0; s < NUM_STATES; s++) {
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    qMb[s][a] = 0.0;
                }
            }
        }

        // 乱数配列を事前生成
        double[] randVals = new double[N_PRIORITIZED_SWEEPS];
        for (int i = 0; i < N_PRIORITIZED_SWEEPS; i++) {
            randVals[i] = rng.nextDouble();
        }

        runPrioritizedSweeping(randVals);
    }

    /**
     * Model-Basedエージェントの思考プロセス (Prioritized Sweeping)
     */
    private void runPrioritizedSweeping(double[] randVals) {
        // 経験がまだなければ何もしない
        boolean hasExperience = false;
        outer:
        for (int s = 0; s < NUM_STATES; s++) {
            for (int a = 0; a < NUM_ACTIONS; a++) {
                for (int ns = 0; ns < NUM_STATES; ns++) {
                    if (modelCounts[s][a][ns] > 0) {
                        hasExperience = true;
                        break outer;
                    }
                }
            }
        }

        if (!hasExperience) {
            return;
        }

        // 思考の初期化 (Reset)
        double[] H = new double[NUM_STATES];
        double[] V = new double[NUM_STATES];
        double[] probs = new double[NUM_STATES];
        double[] qVals = new double[NUM_ACTIONS];

        // Planning Loop
        for (int sweepIdx = 0; sweepIdx < N_PRIORITIZED_SWEEPS; sweepIdx++) {
            // 1. 思考する状態の選択 (Softmax on Priority H)
            double maxH = H[0];
            for (int s = 1; s < NUM_STATES; s++) {
                if (H[s] > maxH) maxH = H[s];
            }

            double sumHExp = 0.0;
            for (int s = 0; s < NUM_STATES; s++) {
                probs[s] = Math.exp((H[s] - maxH) / T_MB);
                sumHExp += probs[s];
            }

            if (sumHExp > 1e-9) {
                for (int s = 0; s < NUM_STATES; s++) {
                    probs[s] /= sumHExp;
                }
            } else {
                for (int s = 0; s < NUM_STATES; s++) {
                    probs[s] = 1.0 / NUM_STATES;
                }
            }

            // 確率的選択 (CDF)
            double randVal = randVals[sweepIdx];
            double cumulative = 0.0;
            int sTilde = NUM_STATES - 1;
            for (int i = 0; i < NUM_STATES; i++) {
                cumulative += probs[i];
                if (randVal <= cumulative) {
                    sTilde = i;
                    break;
                }
            }

            // 2. 選択した状態のQ値をモデルから再計算
            for (int a = 0; a < NUM_ACTIONS; a++) {
                double totalCount = 0.0;
                for (int nextS = 0; nextS < NUM_STATES; nextS++) {
                    totalCount += modelCounts[sTilde][a][nextS];
                }

                if (totalCount <= 0) {
                    qVals[a] = 0.0;
                    continue;
                }

                double qVal = 0.0;
                for (int nextS = 0; nextS < NUM_STATES; nextS++) {
                    double count = modelCounts[sTilde][a][nextS];
                    if (count > 0) {
                        double probTrans = count / totalCount;
                        double rExpected = modelRewards[sTilde][a][nextS] / count;
                        qVal += probTrans * (rExpected + DISCOUNT * V[nextS]);
                    }
                }

                qVals[a] = qVal;
            }

            // 結果を保存
            System.arraycopy(qVals, 0, qMb[sTilde], 0, NUM_ACTIONS);

            // 3. 優先度 H の更新
            double maxQ = qVals[0];
            for (int a = 1; a < NUM_ACTIONS; a++) {
                if (qVals[a] > maxQ) maxQ = qVals[a];
            }
            double delta = Math.abs(V[sTilde] - maxQ);
            V[sTilde] = maxQ;

            // 全状態の優先度更新
            for (int s = 0; s < NUM_STATES; s++) {
                double maxP = 0.0;
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    double totalCountS = 0.0;
                    for (int ns = 0; ns < NUM_STATES; ns++) {
                        totalCountS += modelCounts[s][a][ns];
                    }

                    if (totalCountS <= 0) {
                        continue;
                    }

                    double cnt = modelCounts[s][a][sTilde];
                    if (cnt > 0) {
                        double p = cnt / totalCountS;
                        if (p > maxP) {
                            maxP = p;
                        }
                    }
                }

                double hS = delta * maxP;

                if (s == sTilde) {
                    H[s] = hS;
                } else {
                    if (hS > H[s]) {
                        H[s] = hS;
                    }
                }
            }
        }
    }

    // デバッグ用のアクセサ
    public double[] getQMf(int state) {
        return qMf[state].clone();
    }

    public double[] getQMb(int state) {
        return qMb[state].clone();
    }

    public double[] getQMix(int state) {
        double[] qMix = new double[NUM_ACTIONS];
        for (int a = 0; a < NUM_ACTIONS; a++) {
            qMix[a] = beta * qMb[state][a] + (1.0 - beta) * qMf[state][a];
        }
        return qMix;
    }
}
