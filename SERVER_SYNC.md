# X-002 AIME 2024 目标机运行与 Git 同步（当前唯一操作正文）

私有仓库：https://github.com/KeYSCY1/small-model-posttraining
**状态：R12工程已通过PI审阅，但AIME题单含已确认源题误写aime24:88，待R13剔除后发布。当前配置仍含该题，下列步骤暂不执行生成，尚无AIME可运行commit/tag。**
仓库根就是运行包，不复制单个文件。以下命令在 Linux 目标机由用户手动执行，**一次**完成。

## 1 获取 PI 指定的版本

```bash
cd ~/autodl-tmp
git clone https://github.com/KeYSCY1/small-model-posttraining.git   # 首次；已有克隆则进入该目录
cd small-model-posttraining
git status --short          # 必须无输出；有本地改动先停止并保留差异
git fetch origin --tags
git switch --detach <PI 指定的 commit 或 tag>
git rev-parse HEAD          # 必须与 PI 交接的版本一致
```

## 2 核对环境并跑离线检查（不要重装已有环境）

先激活上次成功运行使用的 Python 环境，再核对依赖：

```bash
python -c "import torch,transformers,datasets,huggingface_hub as h;print(torch.__version__,transformers.__version__,datasets.__version__,h.__version__)"
```

期望（唯一有实测依据的一组）：`2.5.1+cu121  4.57.6  3.6.0  0.36.2`。
不一致或导入失败 → **停止**并回传该输出与 `git rev-parse HEAD`；不就地安装或升级。

```bash
bash run_offline_checks.sh offline_checks.log     # 期望 Ran 68 tests … OK
tail -5 offline_checks.log
```

不是 `OK` 就停止并回传日志与 `git rev-parse HEAD`。

## 3 预缓存固定版本的小数据（准备动作，不计入 44 分钟）

```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("math-ai/aime24", split="test",
                  revision="83a7f387baaa524a8bda0022eac0541582297103")
print(len(ds), ds.column_names)      # 期望 30 与 ['id','problem','solution','url']
PY
```

**不要重新下载模型**（沿用原有缓存）。正式运行时读数据、加载与生成都**计入** 44 分钟。

## 4 运行前核对题单（无模型，可选但推荐）

```bash
python3 - <<'PY'
import json
frozen = json.load(open("frozen_problems_aime24.json", encoding="utf-8"))
print(frozen["selected_ids_sha16"], len(frozen["selected_ids"]),
      frozen["dataset_revision"])
PY
# 期望输出：e7fd0ff6c3875ed1 11 83a7f387baaa524a8bda0022eac0541582297103
```

正式运行会在**加载模型之前**自动核对题单与冻结清单；不一致会直接报错停止，不会换题后再发现。

## 5 只运行一次（外层 44 分钟 / 用户总窗口 ≤45 分钟）

```bash
python3 -m x002.cli run --config config_aime24.json 2>&1 | tee console.log; echo "exit=${PIPESTATUS[0]}"
python3 -m x002.cli summarize --out-dir runs/x002_aime24_pilot_r11 | tee summary_console.log
```

非零退出（含超时）也**只做只读汇总与回收**，不重跑、不放宽选题、不换模型、不增题量或时限。

## 6 版本文件、打包与回传

```bash
git rev-parse HEAD > code_version.txt && git status --short >> code_version.txt
tar czf x002_aime24_pilot_r11_results.tar.gz code_version.txt offline_checks.log console.log \
    summary_console.log runs/x002_aime24_pilot_r11
```

回传 `x002_aime24_pilot_r11_results.tar.gz`（含 records.jsonl、运行头、started/status、summary 与日志）；
不要包含 `.ipynb_checkpoints/`。失败时能保存多少就回传多少，不为补齐文件重跑。

## 7 历史索引（不作为当前步骤）

`r10-reviewed-20261013` 是已审的 **MATH-500** 版本（48 题、`config.json`、`runs/x002_pilot_r08`），
只作历史对照与回滚参考；AIME 结果不与 MATH 结果无说明地混合。发布标签由 PI 验收 AIME 版本后填入。
