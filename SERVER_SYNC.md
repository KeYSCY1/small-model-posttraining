# X-002已审先导运行与Git同步

私有仓库：https://github.com/KeYSCY1/small-model-posttraining

**已审版本：`r10-reviewed-20261013`。** 仓库根就是运行包，不再复制单个文件。以下步骤在Linux目标机由用户手动执行；先激活上次成功运行使用的Python环境。模型和数据继续使用原有缓存，原始结果留在runs/，不加入Git。

## 1 获取固定版本

首次使用新的代码目录，保留原目录与结果：

```bash
cd ~/autodl-tmp
git clone https://github.com/KeYSCY1/small-model-posttraining.git
cd small-model-posttraining
```

已有克隆时直接进入该目录。先执行 `git status --short`，有本地改动就停止并保留差异；作业运行时不切版本。

```bash
git fetch origin --tags
git switch --detach r10-reviewed-20261013
git describe --tags --exact-match
git rev-parse HEAD
git status --short
```

标签应为 `r10-reviewed-20261013`，工作树应无输出。私有仓库需要服务器自身的GitHub认证；浏览器登录不会自动传到服务器。认证失败只处理认证，不回到逐文件覆盖。

## 2 核对现有环境并检查

```bash
python3 -c "import torch,transformers,datasets,huggingface_hub as h;print(torch.__version__,transformers.__version__,datasets.__version__,h.__version__)"
```

应为上一轮已测组合：`2.5.1+cu121 4.57.6 3.6.0 0.36.2`。不一致或导入失败就停止并回传输出，不就地安装或升级。

```bash
bash run_offline_checks.sh offline_checks.log
git status --short
```

应为58项、结尾OK、退出0；工作树仍干净。失败就停止并回传日志与源码commit。

## 3 只运行一次

确认缓存已就绪后，以下使用Bash。先导最多48题，外层44分钟含加载和生成，用户总窗口不超过45分钟；不自动重跑、增时或扩题。

```bash
set -o pipefail
python3 -m x002.cli run --config config.json 2>&1 | tee console.log
x002_run_rc=${PIPESTATUS[0]}
echo "run_exit=${x002_run_rc}"
python3 -m x002.cli summarize --out-dir runs/x002_pilot_r08 2>&1 | tee summary_console.log
```

`runs/x002_pilot_r08` 是沿用的配置目录名，不代表运行了旧源码；身份以运行头commit/tag为准。目录已启动会拒绝再次生成，不删除标记或换目录绕过；若已存在该目录，停止并回传已有记录。

非零退出时只做上面的只读汇总和日志回收，不再次生成。真实题单在首次运行中按预先规则冻结，排除R06十二题、只取金标准受支持子集；实际入组数/组成、缺失分母、两臂成本均落盘。样本不足或没有合格错题如实报告，不以旧n=1或此次先导作机制/总体效果结论。

## 4 回传原始结果

```bash
git rev-parse HEAD > code_version.txt
git status --short >> code_version.txt
tar czf x002_pilot_r10_results.tar.gz code_version.txt offline_checks.log console.log summary_console.log runs/x002_pilot_r08
```

带回 `x002_pilot_r10_results.tar.gz` 即可，不需人工整理研究报告。包应包含原始records.jsonl、运行头、started/status、summary与日志；失败时能保存多少就回传多少，不为补齐文件重跑。模型、缓存与虚拟环境不回传。

## 5 AIME 2024 先导（R11，待 PI 审阅）

下一动作指向 **AIME 2024** 新配置；`r10-reviewed-20261013` 仍是已审 MATH-500 历史版本，**不包含** AIME 适配。PI 审阅 R11 并指定 commit/tag 后，按同一顺序执行（部署 → 环境 → 离线检查 → 运行）：

```bash
git fetch origin --tags && git switch --detach <PI 指定的 R11 commit 或 tag>
git status --short && git rev-parse HEAD          # 必须干净且与交接一致
bash run_offline_checks.sh offline_checks.log     # 期望 Ran 65 tests … OK
python3 -m x002.cli run --config config_aime24.json 2>&1 | tee console.log; echo "exit=${PIPESTATUS[0]}"
python3 -m x002.cli summarize --out-dir runs/x002_aime24_pilot_r11 | tee summary_console.log
```

- 数据：`math-ai/aime24` test，固定版本 `83a7f387baaa524a8bda0022eac0541582297103`。首次需把该小数据集下载进缓存（**准备动作，不计入 44 分钟**）；正式运行时读数据**计入** 44 分钟。
- 生成上限 8192（首轮与两臂同值）；最多 12 题、最多 36 次尝试；外层 44 分钟 / 用户总窗口 ≤45 分钟。
- 输出目录 `runs/x002_aime24_pilot_r11`（不得写入旧目录）；汇总与回收都用新目录。
- 目标机既有成功环境**优先保留，不就地升级**；版本检查命令与期望值同第 2 节。
- 结果打包：`tar czf x002_aime24_pilot_r11_results.tar.gz code_version.txt offline_checks.log console.log summary_console.log runs/x002_aime24_pilot_r11`
