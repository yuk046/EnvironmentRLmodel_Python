# Addiction RL Simulation (Java版)

PythonプロジェクトをJavaに変換した依存症強化学習シミュレーションです。

## プロジェクト構造

```
java/
├── pom.xml                          # Mavenビルドファイル
└── src/main/java/addiction/rl/
    ├── Constants.java               # 定数・設定クラス
    ├── AddictionEnvironment.java    # 環境クラス
    ├── HybridAgent.java             # ハイブリッドエージェント (MF + MB)
    ├── PhaseResult.java             # フェーズ結果クラス
    ├── Simulator.java               # シミュレーション実行クラス
    └── Main.java                    # メインクラス
```

## ビルド方法

### Mavenを使用する場合

```bash
cd java
mvn clean package
```

### 直接コンパイルする場合

```bash
cd java/src/main/java
javac -d ../../../target/classes addiction/rl/*.java
```

## 実行方法

### Mavenを使用する場合

```bash
cd java
mvn exec:java
```

または、引数を指定して実行:

```bash
mvn exec:java -Dexec.args="--num-agents 900 --num-runs 1 --seed 42"
```

### JARファイルから実行

```bash
cd java
java -jar target/addiction-rl-sim-1.0.0.jar
```

引数を指定:

```bash
java -jar target/addiction-rl-sim-1.0.0.jar --num-agents 900 --num-runs 1 --seed 42 --mb-forget
```

## コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--num-agents` | 900 | シード毎のエージェント数 |
| `--num-runs` | 1 | シードの数 |
| `--seed` | 42 | 基本乱数シード |
| `--mb-forget` | false | 毎回のPlanning呼び出しでMB Q値をリセット |
| `--debug-episode` | false | 最初のエージェントの詳細遷移を出力 |
| `--debug-csv` | null | デバッグCSV出力パス |

## Python版との対応関係

| Python | Java |
|--------|------|
| `Constants` (グローバル変数) | `Constants.java` |
| `AddictionEnvironment` | `AddictionEnvironment.java` |
| `HybridAgent` | `HybridAgent.java` |
| `PhaseResult` (@dataclass) | `PhaseResult.java` |
| `simulate()` 関数 | `Simulator.simulate()` |
| `main()` 関数 | `Main.main()` |
| Numba JIT関数 | `HybridAgent.runPrioritizedSweeping()` |

## 注意事項

- Python版のNumbaによるJITコンパイルは、Java版では純粋なJavaコードに置き換えています
- 乱数生成器の実装の違いにより、同じシードでも結果が完全には一致しない場合があります
- グラフ描画機能（matplotlib）はJava版には含まれていません
