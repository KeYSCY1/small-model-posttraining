# X-002 运行包（一次性工程试运行）

**状态：未运行。** 本轮只做本地编写与无模型离线检查；未安装、未下载、未运行模型、未操作目标机。

## 1. 目录

```
run_package/
  config.json                  配置（模型/数据入口、上限、种子、反馈句、总截止）
  requirements.txt             依赖
  README.md                    本文件
  x002/
    judge.py                   答案抽取与数值等价判定（标量范围，范围外 UNKNOWN）
    prompts.py                 两臂消息、唯一编码路径、来源与完整结构核对
    generator.py               单一 transformers 后端（使用冻结 commit）+ 单次生成
    worker.py                  工作进程：冻结身份 → 首轮 → 判分 → A/B → 汇总
    cli.py                     父进程监督者：运行 / 只读汇总
    run.py                     记录与汇总分母
  tests/test_offline_checks.py 19 项离线回归检查（无模型、无网络）
  tests/fake_backend.py        测试替身后端（只替换外部模型，不复制业务循环）
```

## 2. Linux 准备步骤（**由用户执行**；本轮未执行）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
**依赖版本核对（交接基准）**：`requirements.txt` 已按**已测组合**加上界——torch 2.5.1+cu121 /
transformers 4.57.6 / datasets 3.6.0 / huggingface_hub 0.36.2 / sympy 1.13.1（2026-10-13 首次成功试运行的
运行头记录）。**torch 仍需按目标机 CUDA 版本单独安装**。上界只用于防止自动升到已知不兼容的大版本
（transformers 5.x 需要更新版 torch），**不代表区间内所有组合都已测**。目标机既有的成功环境应**优先保留**，
不必重装。

模型与数据可**预缓存**（正式计时运行之前完成，不计入 44 分钟总截止）。
官方 Python 下载 API（见 [HF 文档](https://huggingface.co/docs/huggingface_hub/guides/download)）：

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")   # 权重
from datasets import load_dataset
load_dataset("HuggingFaceH4/MATH-500", split="test")                      # 数据预缓存
```

**所有命令都先进入运行包目录**（例如 `cd /path/to/run_package`）再执行。
先跑离线检查：`PYTHONDONTWRITEBYTECODE=1 python3 tests/test_offline_checks.py`
（**34 项**全过才继续；离线通过 **不代表** GPU 路径可用。）

**当前状态**：本包已于 2026-10-13 在单张 RTX 4090 上**完成一次真实工程试运行**
（12 题、322.16 秒、14 条生成记录、无错误；仅产出 1 对可判定 A/B）。该次结果的判读见
`20261013/research_R06/R06_首次试运行结果判读.md`；它**不构成研究结论**。

**离线检查的实际范围**（与本节说明一致）：
* 使用**离线 fixture** 提供身份与环境，**不访问 HuggingFace Hub、不下载**；断言"依赖已安装"时也不会被导入
  （用哨兵包验证：纯离线路径对 torch/transformers/datasets/sympy/huggingface_hub 的导入记录为空）。
* 环境版本从**包元数据**读取（`importlib.metadata`），收集版本不会导入重依赖；符号判分默认关闭时**不导入 SymPy**。
* 短预算用例会在替身**进入阻塞点之后**写就绪凭据，再断言该阶段确已到达；启动阶段被截止另有一例单独覆盖。
* 仍**未**覆盖：真实 tokenizer/模型加载与生成、GPU 显存与吞吐、真实 Hub commit 解析、SymPy 判分分支。

## 2.1 部署方式（整包替换，勿逐文件挑）

历史教训：曾把测试文件与部分实现逐个覆盖到旧包上，结果 `cli.py` 仍是旧版、参数解析直接退出，
表现为"全部用例失败"。因此：

1. 用 `SHA256.txt` 核对压缩件解压后的**整包**（见交接说明第 4 步命令）；
2. 若目标机已有旧 `run_package`，**整目录替换**（或解压到新目录），不要逐文件复制；
3. 保留目标机既有可用 venv；只有整包替换才动代码。

## 3. 唯一推荐命令

```bash
cd run_package
python3 -m x002.cli run --config config.json          # 一次性运行到 config.paths.out_dir
python3 -m x002.cli summarize --out-dir runs/x002_pilot_r07   # 只读汇总（可随时对已有日志运行）
```

题号与身份**无需手填**。先导批次（`config.pilot`）使用**与模型输出无关的预先确定规则**：
排除 `exposed_problems_r06.json` 里已暴露的 12 个 ID → 只保留现行判分支持的**标量金标准**
（区间/集合/坐标/复数/无有效答案被筛除）→ 按 `sha256(协议版本|数据commit|种子|unique_id|归一化金标准)`
升序取前 `max_problems`（先导为 48）。逐层分母（原始池 / 排除已暴露 / 非标量筛除 / 候选 / 入组 / 差额）
与 subject·level 组成写入运行头的 `selection.layers`；候选不足时如实记 `shortfall`，**不放宽筛选**。
若配置不含 `pilot` 段，则退回旧规则（`unique_id` 升序取前 N）以便历史配置复现。
模型与数据 commit 由工作进程解析后写入运行头；解析失败或题目缺失直接报错。

## 4. 一次性语义（没有恢复、没有覆盖）

* 输出目录在**加载与生成之前**写入启动标记 `started.json`。
* 已存在标记的目录**拒绝再次生成**（返回码 6），只允许 `summarize`；上次失败或超时也不重跑，
  **不自动新建目录绕过预算**——原始日志留给 PI 判断下一批。
* 本包**不提供**续跑、覆盖、恢复或持久化预算账本（按 PI 裁决删除）。

## 5. 总截止与停止

| 机制 | 说明 |
| --- | --- |
| 总截止 | `run.smoke.total_minutes`（默认 45）从**进程启动**计时，**包含模型加载、编码与生成** |
| 宽限位置 | **终止宽限预留在总上限之内**：`SIGTERM` 在 `上限 − 宽限(1s)` 发出，`SIGKILL` 在上限时刻发出（不会"等满预算再额外等 5 秒"） |
| 监督方式 | 工作进程在独立进程组里运行；父进程在**启动时保存进程组号**，按该组号检查与清理整组，不因组长退出就认定全组结束；宽限后仍存活的成员被强杀并确认 |
| 终止结果 | `status.json` 记录 `worker_pid`、**原始 `worker_pgid`**、`reason`、`timed_out`、`worker_returncode`、`term_sent_at_seconds`、`sigkill_sent_at_seconds`、`group_alive_after_cleanup`；超时退出码 7 |
| 非零退出 | 工作进程非零退出时父进程**保留并传播该退出码**，`reason=worker_failed`（不误报为超时） |
| 已完成记录 | 逐条追加写入，终止不会丢已有记录；`records.jsonl` 仍可 `summarize` |
| 加载阻塞/异常 | 身份在加载**之前**已落盘，因此加载阶段超时或依赖缺失同样有可回收结果 |
| 不夸大 | 不使用线程软停止；**没有**后台线程继续计算；`max_new_tokens` 不是时间硬停止 |

## 6. 日志与回收

`runs/x002_smoke/` 目录整体打包即可：

* `runs/x002_pilot_r07/` —— 先导批次输出目录（R06 的 `runs/x002_smoke/` 保留只读，一次性语义会拒绝覆盖）
* `started.json` —— 启动标记（时间、配置路径、预算、父进程 pid）
* `status.json` —— 退出原因（`worker_exited` / `worker_failed` / `total_deadline_exceeded`）、是否超时、用时、
  `SIGTERM`/`SIGKILL` 发出时刻、`worker_pid` 与原始 `worker_pgid`、清理后是否仍有存活成员
* `records.jsonl` —— 第一行 `{"kind":"run", ...}` 运行头（模型/数据 commit、题号与题面/标签摘要、
  反馈、生成配置、判分配置、提示与判分版本、环境版本）；其后每条记录含阶段/臂、messages、
  `serialized_text`、`input_sha16`、`input_check`、生成原文、判分、token 与耗时、`run_status` 与错误原因
* `summary.json` / `summary.txt` —— 由 `records.jsonl` 派生，**不能替代原始记录**

## 7. 判分范围

* 支持：`\boxed{}`/`\fbox{}`（空盒→UNKNOWN）、`#### 42`、`final answer is X`（含英文小数）、
  整数/小数/简单分数/百分数/合法千分位、`$...$` 行内公式。
* 范围外一律 **UNKNOWN（不判错）**：带单位、区间、集合、元组/坐标、矩阵、分段函数、
  非法逗号串（`1,2`）、多答案、抽不出答案、参考答案非标量。因此 `The answer is 2.5 cm.` 对 `2` → UNKNOWN。
* 候选抽取**不做数字前缀回退**（不把 `2.5` 截成 `2`）；句子里的候选之后若还有别的数值
  （`The answer is 1,2.` / `1, 2.` / `2.5, or 7.`）一律 **UNKNOWN**，而合法千分位 `1,000.` 仍正确。
* 符号判错路径默认关闭（`judge.symbolic=false`）；需要 SymPy，**本轮未安装、未验证**。
* 标准答案只用于判分与日志，**不进入任何消息**。

## 8. 输入核对（生成前，失败即阻止推理）

* 消息**只由白名单字段构造**：题面、固定反馈、**仅 A 臂**的完整首轮旧答、共用 system。
  外部答案/参考解字段不参与构造；题面或旧答里自然出现与标准答案相同的内容**不算泄漏**。
* 生成与留证走**唯一编码路径**（`apply_chat_template(..., tokenize=True/False)`），
  核对通过后把**同一份 input_ids** 直接交给推理；编码结果兼容 `dict` 与 `UserDict`/`BatchEncoding`。
* 核对内容：角色序列一致；每条期望内容的**归一化完整正文**按顺序出现在实际解码文本里——
  缺尾部会失败，不靠"还有别的特有片段"放行。
* 核对或编码失败：不调用后端，记录 `input_check_failed` / `encode_error` 等明确错误。

## 9. 汇总口径

* 首轮：`correct` / `incorrect` / `undetermined` / `truncated` / `error`；只有**正常完成且判错**者进第二轮。
* 第二轮同样区分 `completed` / `truncated` / `error` / `unknown`。
* 默认比较只用 **`determinate_pairs`**（两臂正常完成且判分可确定），不代表全部入组题；
  未纳入者逐条列入 `excluded_from_pairs`。
* 另报 `verified_success_rate_all_attempts`：分子要求 **`run_status=completed` 且判分 `correct`**；
  截断、错误、UNKNOWN **不计成功**，但仍留在"全部尝试"的分母里。
* **先导/工程批次不作区间推断**：只报观察到的两臂修正率、配对数与成本，字段标为
  `interval_label="未估计（仅描述）"`；不再输出 bootstrap 区间（n=1 时的 `[−1,−1]` 是退化输出，已移除）。
* 截断但碰巧含答案**不算正常完成**；UNKNOWN 与运行错误**不会**被隐式当作答错。

## 10. 预算与未验证路径

* 试运行（提议，待 PI 审核）：≤12 道冻结题、首轮各 1 次、合格错题 A/B 各 1 次，单 4090，**总运行 ≤45 分钟**；
  安装与下载单独计，不占用这 45 分钟。
* **未验证**：真实 tokenizer 的 chat template 行为、真实模型生成、显存与吞吐、真实数据读取与 commit 解析、
  SymPy 分支。离线检查使用替身后端，**mock 通过 ≠ GPU 烟测通过**。
