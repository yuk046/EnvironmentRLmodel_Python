package addiction.rl;

import static addiction.rl.Constants.*;

/**
 * 依存症強化学習シミュレーション メインクラス
 */
public class Main {

    public static void main(String[] args) {
        // コマンドライン引数のパース
        int numAgents = 900;
        int numRuns = 1;
        long seed = 42;
        boolean mbForget = false;
        boolean debugEpisode = false;
        String debugCsv = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--num-agents":
                    if (i + 1 < args.length) {
                        numAgents = Integer.parseInt(args[++i]);
                    }
                    break;
                case "--num-runs":
                    if (i + 1 < args.length) {
                        numRuns = Integer.parseInt(args[++i]);
                    }
                    break;
                case "--seed":
                    if (i + 1 < args.length) {
                        seed = Long.parseLong(args[++i]);
                    }
                    break;
                case "--mb-forget":
                    mbForget = true;
                    break;
                case "--debug-episode":
                    debugEpisode = true;
                    break;
                case "--debug-csv":
                    if (i + 1 < args.length) {
                        debugCsv = args[++i];
                    }
                    break;
                default:
                    // 未知の引数は無視
                    break;
            }
        }

        // Beta値の範囲
        double[] betaValues = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
        double[] rates = new double[betaValues.length];

        System.out.printf("Simulation Start: Agents=%d, Runs=%d, Seed=%d%n", numAgents, numRuns, seed);
        System.out.println("------------------------------------------------------------");

        final int finalNumAgents = numAgents;
        final int finalNumRuns = numRuns;
        final boolean finalMbForget = mbForget;
        final boolean finalDebugEpisode = debugEpisode;
        final String finalDebugCsv = debugCsv;

        // 進捗コールバック
        Simulator.ProgressCallback progressCallback = (beta, seedIdx, agentIdx, nAgents, nRuns, pIdx, step, length) -> {
            if (seedIdx == 0 && agentIdx == 0 && step % 100 == 0) {
                String pName = PHASES[pIdx].name;
                System.out.printf("[Beta=%.1f] Phase: %s Step: %d/%d%n", beta, pName, step, length);
            }
        };

        long currentSeed = seed;
        for (int i = 0; i < betaValues.length; i++) {
            double beta = betaValues[i];

            // デバッグログのファイル名を生成
            String debugTxtPath = null;
            if (finalDebugEpisode) {
                debugTxtPath = String.format("debug_log_beta_%.2f.txt", beta);
            }

            double rate = Simulator.simulate(
                beta,
                finalNumAgents,
                finalNumRuns,
                currentSeed,
                progressCallback,
                finalMbForget,
                finalDebugEpisode,
                finalDebugCsv,
                debugTxtPath
            );

            currentSeed += finalNumRuns;  // 次のBetaではシードをずらす
            rates[i] = rate * 100;
            System.out.printf("Result: Beta=%.1f => Addiction Rate=%.2f%%%n", beta, rate * 100);
            System.out.println("------------------------------------------------------------");
        }

        // 結果サマリー
        System.out.println("\n=== Final Results ===");
        for (int i = 0; i < betaValues.length; i++) {
            System.out.printf("Beta=%.1f: %.2f%%%n", betaValues[i], rates[i]);
        }
    }
}
