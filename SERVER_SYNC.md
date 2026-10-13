# X-002 AIME 2024：固定版本部署、运行与回传

私有仓库：https://github.com/KeYSCY1/small-model-posttraining

**已审版本：`r13-reviewed-20261013`。** PI独立执行68项离线检查全部通过、无跳过，并复核固定原始数据走正式选题路径。以下在Linux单4090目标机由用户手动执行一次；使用上次成功的Python环境。仓库根就是运行包。

本次固定11题：原池30题，因源题转录错误排除aime24:88，质量合格池29题，保留原题单其余11题、不补题。首轮各一次，合格错题A/B各一次，最多33次尝试；三种生成均为8192输出上限，外层44分钟含加载与生成、用户总窗口≤45分钟。

## 1 获取固定版本

首次克隆（已有克隆时直接进入现有目录）：

```bash
cd ~/autodl-tmp
git clone https://github.com/KeYSCY1/small-model-posttraining.git
cd small-model-posttraining
```

先查看工作树；有本地改动时停止并保留差异，作业运行时不切版本。私库使用服务器自身的GitHub认证，浏览器登录不会自动传过去。

```bash
git status --short
git fetch origin --tags
git switch --detach r13-reviewed-20261013
git describe --tags --exact-match
git rev-parse HEAD
git status --short
```

标签必须为`r13-reviewed-20261013`，最后的工作树状态应无输出。代码按整个版本切换，不逐文件覆盖。

## 2 核对环境与离线检查

先激活上次成功运行使用的环境，全部步骤使用其中的`python3`：

```bash
python3 -c "import torch,transformers,datasets,huggingface_hub as h;print(torch.__version__,transformers.__version__,datasets.__version__,h.__version__)"
bash run_offline_checks.sh offline_checks.log
git status --short
```

已测依赖组合为`2.5.1+cu121 4.57.6 3.6.0 0.36.2`。不一致或导入失败时停止并回传输出，不就地安装或升级。

检查应发现68项、退出0，结尾为`OK`。服务器没有本工作区R11原始缓存时，其中2项缓存读取用例会跳过，结尾可为`OK (skipped=2)`；其余检查须通过，下面第3步再用服务器真实数据核对正式选题。工作树应干净。失败时只回收日志和源码版本，不启动模型。

## 3 缓存数据并核对正式题单（无模型）

只下载固定版本的小数据，沿用原有模型缓存。下面使用Bash；检查失败时停止，回传`data_preflight.log`，不要进入第4步。

```bash
set -o pipefail
python3 - <<'PY' 2>&1 | tee data_preflight.log
import json
from x002.worker import DatasetReader, freeze_selection, verify_frozen_selection

with open("config_aime24.json", encoding="utf-8") as f:
    cfg = json.load(f)
revision = cfg["data"]["revision"]
reader = DatasetReader(cfg, {"revision": revision})
problems, selection = freeze_selection(reader, cfg, dataset_revision=revision)
check = verify_frozen_selection(cfg, problems, selection, revision)
layers = selection["layers"]
assert len(reader.list_ids()) == 30
assert layers["data_quality_removed_ids"] == ["aime24:88"]
assert layers["pool_total_after_quality"] == 29
assert len(problems) == 11 and check["ok"]
assert selection["selected_ids_sha16"] == "e7fd0ff6c3875ed1"
print("题单核对通过：", len(problems), selection["selected_ids_sha16"], revision)
print("本次题号：", [p["problem_id"] for p in problems])
PY
x002_preflight_rc=${PIPESTATUS[0]}
echo "preflight_exit=${x002_preflight_rc}"
```

退出必须为0，核对结果应为`11 e7fd0ff6c3875ed1 83a7f387baaa524a8bda0022eac0541582297103`。这是生成前准备，不加载模型；正式运行中读数据、模型加载和生成仍计入44分钟。

## 4 只运行一次

在第2、3步成功后执行以下Bash代码。输出目录名沿用首版的`r11`，实际源码身份以本次已审标签和运行头commit为准。

```bash
set -o pipefail
if [ -e runs/x002_aime24_pilot_r11/started.json ]; then
    echo "已有一次运行记录，不再生成；只做第5步汇总和第6步回传。"
else
    python3 -m x002.cli run --config config_aime24.json 2>&1 | tee console.log
    x002_run_rc=${PIPESTATUS[0]}
    echo "run_exit=${x002_run_rc}"
fi
```

非零退出或到时也只做下面的汇总和回收，不删除启动标记，不换目录绕过，不重跑、增题或延时。11题是上限，到截止时可能尚未全部完成；如实保留已有记录。

## 5 只读汇总

```bash
python3 -m x002.cli summarize --out-dir runs/x002_aime24_pilot_r11 2>&1 | tee summary_console.log
```

汇总应保留入组、首轮、各臂、可判定配对和缺失分母，以及截断、UNKNOWN、错误和成本。样本不足或两臂均失败均如实报告；本次是数据适配与提示比较先导，不是总体效果或机制结论。

## 6 打包回传

```bash
git rev-parse HEAD > code_version.txt
git status --short >> code_version.txt
python3 - <<'PY'
from pathlib import Path
import tarfile

archive = "x002_aime24_pilot_r13_results.tar.gz"
names = ["code_version.txt", "offline_checks.log", "data_preflight.log",
         "console.log", "summary_console.log", "runs/x002_aime24_pilot_r11"]
with tarfile.open(archive, "w:gz") as tar:
    for name in names:
        if Path(name).exists():
            tar.add(name, arcname=name,
                    filter=lambda info: None if ".ipynb_checkpoints" in Path(info.name).parts else info)
        else:
            print("本次未产生：", name)
print("回传文件：", archive)
PY
```

带回`x002_aime24_pilot_r13_results.tar.gz`即可，包含已有原始records.jsonl、运行头、started/status、summary及日志。失败时打包已有文件，不为补齐产物重跑。模型、缓存、环境与原始结果不上传Git。

`r10-reviewed-20261013`是历史MATH-500版本；本次使用AIME配置和新输出目录，不无说明地混合两批结果。
