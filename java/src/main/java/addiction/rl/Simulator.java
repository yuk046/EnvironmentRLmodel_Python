package addiction.rl;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static addiction.rl.Constants.*;

/**
 * シミュレーション実行クラス
 */
public class Simulator {

    /**
     * 進捗コールバック用インターフェース
     */
    public interface ProgressCallback {
        void onProgress(double beta, int seedIdx, int agentIdx, int numAgents, int numRuns, int phaseIdx, int step, int length);
    }

    /**
     * デバッグ記録用クラス
     */
    public static class DebugRecord {
        public final String phaseName;
        public final int phaseIdx;
        public final int step;
        public final int state;
        public final int action;
        public final String actionName;
        public final double reward;
        public final int nextState;

        public DebugRecord(String phaseName, int phaseIdx, int step, int state, int action,
                           String actionName, double reward, int nextState) {
            this.phaseName = phaseName;
            this.phaseIdx = phaseIdx;
            this.step = step;
            this.state = state;
            this.action = action;
            this.actionName = actionName;
            this.reward = reward;
            this.nextState = nextState;
        }
    }

    /**
     * シミュレーション実行
     */
    public static double simulate(
            double beta,
            int numAgents,
            int numRuns,
            long baseSeed,
            ProgressCallback progressCb,
            boolean mbForget,
            boolean debugEpisode,
            String debugCsvPath,
            String debugTxtPath
    ) {
        int addictions = 0;
        int totalAgents = numAgents * numRuns;
        List<DebugRecord> debugRecords = new ArrayList<>();

        PrintWriter logFile = null;
        if (debugEpisode && debugTxtPath != null) {
            try {
                logFile = new PrintWriter(new FileWriter(debugTxtPath));
            } catch (IOException e) {
                System.err.println("Failed to open debug log file: " + e.getMessage());
            }
        }

        try {
            for (int seedIdx = 0; seedIdx < numRuns; seedIdx++) {
                // Agent/Env用の乱数生成器 (単一のRNGで統一)
                long currentSeed = baseSeed + seedIdx;
                Random rng = new Random(currentSeed);

                for (int agentIdx = 0; agentIdx < numAgents; agentIdx++) {
                    AddictionEnvironment env = new AddictionEnvironment(rng);
                    HybridAgent agent = new HybridAgent(beta, rng, mbForget);

                    int state = env.reset();
                    PhaseResult counts = new PhaseResult();

                    // フェーズ実行
                    for (int phaseIdx = 0; phaseIdx < PHASES.length; phaseIdx++) {
                        Phase phase = PHASES[phaseIdx];
                        int length = phase.steps;

                        // フェーズ開始時にSTATE_STARTにリセット
                        if (phaseIdx > 0) {
                            state = env.reset();
                        }

                        Set<Integer> reportPoints = new HashSet<>(Arrays.asList(1, length / 2, length));

                        for (int stepInPhase = 0; stepInPhase < length; stepInPhase++) {
                            // Progress Log
                            if (progressCb != null && reportPoints.contains(stepInPhase + 1)) {
                                progressCb.onProgress(beta, seedIdx, agentIdx, numAgents, numRuns, phaseIdx, stepInPhase + 1, length);
                            }

                            int action = agent.selectAction(state);

                            // Debug出力用にQ値を取得
                            double[] currentQMf = null;
                            double[] currentQMb = null;
                            double[] currentQMix = null;
                            if (debugEpisode && seedIdx == 0 && agentIdx == 0) {
                                currentQMf = agent.getQMf(state);
                                currentQMb = agent.getQMb(state);
                                currentQMix = agent.getQMix(state);
                            }

                            AddictionEnvironment.StepResult result = env.step(action, phaseIdx);
                            int nextState = result.nextState;
                            double reward = result.reward;
                            agent.observe(state, action, reward, nextState, phaseIdx);

                            if (debugEpisode && seedIdx == 0 && agentIdx == 0) {
                                String qMfStr = formatQValues(currentQMf);
                                String qMbStr = formatQValues(currentQMb);
                                String qMixStr = formatQValues(currentQMix);

                                String[] logLines = {
                                    String.format("[DEBUG] Beta=%.1f Phase=%s Step=%d", beta, phase.name, stepInPhase + 1),
                                    String.format("  State: %d -> Action: %s -> Next: %d (Reward: %.2f)",
                                                  state, ACTION_NAMES.getOrDefault(action, String.valueOf(action)), nextState, reward),
                                    String.format("  Q_MF:  [%s]", qMfStr),
                                    String.format("  Q_MB:  [%s]", qMbStr),
                                    String.format("  Q_MIX: [%s]", qMixStr),
                                    "----------------------------------------"
                                };

                                // コンソール出力
                                for (String line : logLines) {
                                    System.out.println(line);
                                }

                                // ファイル出力
                                if (logFile != null) {
                                    for (String line : logLines) {
                                        logFile.println(line);
                                    }
                                }

                                debugRecords.add(new DebugRecord(
                                    phase.name,
                                    phaseIdx,
                                    stepInPhase + 1,
                                    state,
                                    action,
                                    ACTION_NAMES.getOrDefault(action, String.valueOf(action)),
                                    reward,
                                    nextState
                                ));
                            }

                            // 統計収集 (Addictionフェーズのみ)
                            if (phaseIdx == 1) {
                                if (state == NEUTRAL_MAX && action == ACTION_DRUG && nextState == STATE_DRUG) {
                                    counts.drugChoices++;
                                } else if (state == STATE_GOAL && action == ACTION_GOAL && nextState == STATE_START) {
                                    counts.healthyChoices++;
                                }
                            }

                            state = nextState;
                        }
                    }

                    // 依存判定
                    if (counts.drugChoices > counts.healthyChoices) {
                        addictions++;
                    }

                    if (debugEpisode && seedIdx == 0 && agentIdx == 0 && logFile != null) {
                        logFile.println(String.format("Final Counts - Drug Choices: %d, Healthy Choices: %d",
                                                      counts.drugChoices, counts.healthyChoices));
                    }
                }
            }
        } finally {
            if (logFile != null) {
                logFile.close();
                System.out.println("Debug log (seed=0, agent=0) written to " + debugTxtPath);
            }
        }

        // CSVファイル出力
        if (!debugRecords.isEmpty()) {
            String csvPath = debugCsvPath != null ? debugCsvPath : String.format("debug_episode_beta_%.2f.csv", beta);
            try (PrintWriter csvWriter = new PrintWriter(new FileWriter(csvPath))) {
                csvWriter.println("phase_name,phase_idx,step,state,action,action_name,reward,next_state");
                for (DebugRecord record : debugRecords) {
                    csvWriter.println(String.format("%s,%d,%d,%d,%d,%s,%.2f,%d",
                        record.phaseName,
                        record.phaseIdx,
                        record.step,
                        record.state,
                        record.action,
                        record.actionName,
                        record.reward,
                        record.nextState
                    ));
                }
                System.out.println(String.format("Debug episode (seed=0, agent=0) written to %s (%d steps)",
                                                 csvPath, debugRecords.size()));
            } catch (IOException e) {
                System.err.println("Failed to write debug CSV: " + e.getMessage());
            }
        }

        return (double) addictions / totalAgents;
    }

    private static String formatQValues(double[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.2f", values[i]));
        }
        return sb.toString();
    }
}
