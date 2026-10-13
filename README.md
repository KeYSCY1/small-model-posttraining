# X-002 运行包（一次性工程试运行）

**状态：R08 修补版，未运行。** 本轮只做本地编写与无模型离线检查；未安装、未下载、未运行模型、未操作目标机。
**本版本尚未获得目标机运行准入**（可由 PI 验收后的 commit/tag 取代）。

## 1. 目录

```
run_package/
  config.json                  配置（模型/数据入口、上限、种子、反馈句、总截止、先导选题规则）
  requirements.txt             依赖（含已测组合与"区间内未全测"声明）
  README.md                    本文件
  VERSION_CONTROL.md           Git 版本管理约定（唯一活动源码 + 私有仓库）
  REVIEW_STATUS.md             当前审阅/运行准入状态
  SERVER_SYNC.md               服务器取代码流程（含"当前只可同步、不启动模型"）
  run_offline_checks.sh        离线检查包装脚本（pipefail，失败即非零退出）
  run_worker.py                工作进程入口脚本（正式路径与离线检查共用）
  exposed_problems_r06.json    R06 已暴露的 12 个题号（选题排除用）
  x002/
    judge.py                   答案抽取与数值等价判定（标量范围，范围外 UNKNOWN）
    prompts.py                 两臂消息、唯一编码路径、来源与完整结构核对
    generator.py               单一 transformers 后端（使用冻结 commit）+ 单次生成
    worker.py                  工作进程：冻结身份 → 首轮 → 判分 → A/B → 汇总
    pilot.py                   先导选题（排除已暴露 + 标量金标准 + 稳定哈希排序）
    cli.py                     父进程监督者：运行 / 只读汇总
    run.py                     记录、汇总分母与描述统计
  tests/test_offline_checks.py 50 项离线回归检查（无模型、无网络）
  tests/fake_backend.py        测试替身后端（只替换外部模型，不复制业务循环）
```

## 2. 环境与部署（**由用户执行**；本轮未执行）

**顺序固定：先部署指定代码版本 → 再核对环境 → 再跑一次离线检查 → 才允许运行。**

```bash
cd ~/autodl-tmp
git clone https://github.com/KeYSCY1/small-model-posttraining.git   # 首次
cd small-model-posttraining && git status --short && git rev-parse HEAD
git fetch origin --tags && git switch --detach <PI 指定的 commit 或 tag>   # 后续更新
```

* 目标机既有成功环境**优先保留**；**不自动安装/升级**。
* 已测组合**只有一组**（2026-10-13 首次成功试运行运行头）：torch 2.5.1+cu121 / transformers 4.57.6 /
  datasets 3.6.0 / huggingface_hub 0.36.2 / sympy 1.13.1。其它版本**未知**，不得写成"版本不低于即兼容"。
* 版本或导入不符 → **停止并回传**（`git rev-parse HEAD`、`git status --short`、版本打印与日志），
  不就地重装、不放宽判据。
* 只有整包替换/整版本切换才动代码，不逐文件复制（历史教训见 `VERSION_CONTROL.md`）。
* 模型与数据可**预缓存**（正式计时之前完成，不计入 44 分钟总截止）：
  `snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")`；
  `load_dataset("HuggingFaceH4/MATH-500", split="test")`。

**所有命令都先进入运行包目录**再执行。先跑离线检查（退出码非零或结尾不是 `OK` 就停止）：

```bash
bash run_offline_checks.sh offline_checks.log    # 期望：Ran 50 tests … OK
```

（离线通过 **不代表** GPU 路径可用。）

**当前状态（R08）**：R07 交接版未通过 PI 复跑（37 项中 1 项报错）与四项边界缺口，已在本批定点修补；
历史事实：本包已于 2026-10-13 在单张 RTX 4090 上完成一次真实工程试运行（12 题、322.16 秒、
14 条生成记录、无错误；仅 1 对可判定 A/B），判读见 `20261013/research_R06/R06_首次试运行结果判读.md`，
**不构成研究结论**。

**版本与来源**：源码以 Git 管理（见 [VERSION_CONTROL.md](VERSION_CONTROL.md)、[REVIEW_STATUS.md](REVIEW_STATUS.md)）；
运行头记录实际源码版本（Git commit + 是否有未提交改动，无 Git 时记文件摘要）与解析后的数据 commit。

**离线检查的实际范围**（与本节说明一致）：
* 使用**离线 fixture** 提供身份与环境，**不访问 HuggingFace Hub、不下载**；断言"依赖已安装"时也不会被导入
  （用哨兵包验证：纯离线路径对 torch/transformers/datasets/sympy/huggingface_hub 的导入记录为空）。
* 环境版本从**包元数据**读取（`importlib.metadata`），收集版本不会导入重依赖；符号判分默认关闭时**不导入 SymPy**。
* 版本传递用例**同时**替换 torch 与 transformers 并完整恢复模块，因此在**未安装**这两个依赖的机器上也能跑
  （本批已在一个未安装 torch/transformers/datasets/huggingface_hub 的解释器 Python 3.13.2 上实测 50 项全过）。
* 短预算用例会在替身**进入阻塞点之后**写就绪凭据，再断言该阶段确已到达；启动阶段被截止另有一例单独覆盖。
* 仍**未**覆盖：真实 tokenizer/模型加载与生成、GPU 显存与吞吐、真实 Hub commit 解析、SymPy 判分分支。

## 2.1 版本切换（Git，勿逐文件挑）

历史教训：曾把测试文件与部分实现逐个覆盖到旧包上，结果 `cli.py` 仍是旧版、参数解析直接退出，
表现为"全部用例失败"。因此：

1. 服务器只 `git clone` / `git fetch` + `git switch --detach <PI 指定 commit 或 tag>`，
   切换后核对 `git rev-parse HEAD` 与交接版本一致；
2. **不要**逐个文件覆盖，也不要混用旧的压缩包与新的 Git 目录；
3. 切换前 `git status --short` 必须无输出；有本地修改先停止并保留差异，不强制覆盖；
4. 保留目标机既有可用 venv；只有换版本才动代码。

## 3. 唯一推荐命令

```bash
cd run_package
python3 -m x002.cli run --config config.json          # 一次性运行到 config.paths.out_dir
python3 -m x002.cli summarize --out-dir runs/x002_pilot_r08   # 只读汇总（可随时对已有日志运行）
```

题号与身份**无需手填**。先导批次（`config.pilot`）使用**与模型输出无关的预先确定规则**：
排除 `exposed_problems_r06.json` 里已暴露的 12 个 ID → 只保留现行判分支持的**标量金标准**
（区间/集合/坐标/复数/无有效答案被筛除）→ 按 `sha256(协议版本|数据commit|种子|unique_id|归一化金标准)`
升序取前 `max_problems`（先导为 48）。逐层分母（原始池 / 排除已暴露 / 非标量筛除 / 候选 / 入组 / 差额）
与 subject·level 组成写入运行头的 `selection.layers`；候选不足时如实记 `shortfall`，**不放宽筛选**。
若配置不含 `pilot` 段，则退回旧规则（`unique_id` 升序取前 N）以便历史配置复现。

**同一个解析后的数据 commit 支配三处**：真实读数据（`load_dataset(revision=…)`）、选题排序、
运行头 `dataset.revision` 与 `selection.dataset_revision_used_for_sorting`；
解析不到 commit 时先导选题**拒绝继续**，不用空串排序。模型 commit 同样解析后写入运行头。

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

输出目录整体打包即可（当前配置为 `runs/x002_pilot_r08/`）：

* `runs/x002_pilot_r08/` —— 本批（R08）先导批次输出目录；R06 的 `runs/x002_smoke/` 与
  R07 的 `runs/x002_pilot_r07/` 保留只读，一次性语义会拒绝覆盖
* `started.json` —— 启动标记（时间、配置路径、预算、父进程 pid）
* `status.json` —— 退出原因（`worker_exited` / `worker_failed` / `total_deadline_exceeded`）、是否超时、用时、
  `SIGTERM`/`SIGKILL` 发出时刻、`worker_pid` 与原始 `worker_pgid`、清理后是否仍有存活成员
* `records.jsonl` —— 第一行 `{"kind":"run", ...}` 运行头（模型/数据 commit、题号与题面/标签摘要、
  反馈、生成配置、判分配置、提示与判分版本、环境版本、源码版本 `code_version`）；其后每条记录含阶段/臂、
  messages、`serialized_text`、`input_sha16`、`input_check`、生成原文、判分、token 与耗时、`run_status` 与错误原因
* `summary.json` / `summary.txt` —— 由 `records.jsonl` 派生，**不能替代原始记录**

原始日志只读；本包不生成虚构失败记录，也不把"缺失"写成"答错"。

## 7. 判分范围

* 支持：`\boxed{}`/`\fbox{}`（空盒→UNKNOWN）、`#### 42`、`final answer is X`（含英文小数）、
  整数/小数/简单分数/百分数/合法千分位、`$...$` 行内公式。
* 范围外一律 **UNKNOWN（不判错）**：带单位、区间、集合、元组/坐标、矩阵、分段函数、
  非法逗号串（`1,2`）、多答案、抽不出答案、参考答案非标量。因此 `The answer is 2.5 cm.` 对 `2` → UNKNOWN。
* 候选抽取**不做数字前缀回退**（不把 `2.5` 截成 `2`）。
* **多答案/非法逗号串按完整候选区域判定**：候选之后到「空行」或「句末标点 + 下一句首字母大写」之间的
  整段文字里，只要出现并列形态（`,` `;` `、` `or` `and` 之后紧跟数值）或同一行内还有别的完整数值，
  一律 **UNKNOWN**。因此 `The answer is 1,2.` / `1, 2.` / `2.5, or 7.` / `$2.5$, or 7.` /
  `$1$, or $2$.` / `1,\n2.` / `\boxed{1}, or 2` 都是 UNKNOWN；数字候选与公式候选走**同一条**规则。
* 合法千分位 `1,000.`、单个小数/分数、`\boxed{497}`/`\boxed{129}` 不受影响；
  “候选之后的另一段解释里恰好有数字”（空行之后）**不算**多答案。
* 未覆盖（如实写，不在本批扩范围）：候选句结束后换行再写并列值（`The answer is 2.\nOr 3.`）目前按
  该句只有一个候选处理；不含 `answer is/:` 的裸并列数值仍按“抽不出答案”处理；
  LaTeX 千分位写法（`$1{,}000$`）一律 UNKNOWN，本批不新增该写法支持。
* 符号判错路径默认关闭（`judge.symbolic=false`）；**该分支未在真实数据上验证**——环境装有 SymPy 也不代表已测（关闭时不会导入 SymPy）。
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

* **完整入组分母来自运行头**（不是从已记录条数反推）：入组数与入组 ID、首轮已记录数与**缺失 ID**；
  `summary.txt` 与 `summary.json` 逐项一致。没有运行头的历史日志把入组数标为**未知**，不用已记录数冒充。
* **首轮已记录但未判错**的部分与“没有记录”分开：按 `correct` / `incorrect` / `undetermined` /
  `truncated` / `error` 分桶列出 ID；缺失只表示“入组但没有记录”，不写成答错。
* 各臂分别给出**计划 / 已记录 / 正常完成 / 已记录但 UNKNOWN / 缺失**及缺失 ID；
  计划数 = 首轮**已记录且判错**的题数（首轮未记录的题**不假定**会进入第二轮）。
* 首轮：`correct` / `incorrect` / `undetermined` / `truncated` / `error`；只有**正常完成且判错**者进第二轮。
* 第二轮同样区分 `completed` / `truncated` / `error` / `unknown`。
* 默认比较只用 **`determinate_pairs`**（两臂正常完成且判分可确定），不代表全部入组题；
  未纳入者逐条列入 `excluded_from_pairs`。
* 另报 `verified_success_rate_all_attempts`：分子要求 **`run_status=completed` 且判分 `correct`**；
  截断、错误、UNKNOWN **不计成功**，但仍留在“全部尝试”的分母里。
* **先导/工程批次不作区间推断**：只报观察到的两臂修正率、配对数与成本，字段标为
  `interval_label="未估计（仅描述）"`；不输出 bootstrap 区间（n=1 时的 `[−1,−1]` 是退化输出，已移除）。
  两臂差值直接取均值，不运行已不展示的 bootstrap 计算。
* 截断但碰巧含答案**不算正常完成**；UNKNOWN 与运行错误**不会**被隐式当作答错。

## 9.1 R08 定点修补要点（供核对）

* **离线测试隔离**：版本传递用例**同时**替换 torch 与 transformers，并在 finally 里完整恢复原模块
  （含"原本不存在"的情形）；在未安装这两个依赖的环境里也通过；真实子进程哨兵、加载/生成阻塞、
  截止与进程清理检查全部保留。
* **保守判分边界**：PI 给出的三个反例必须 UNKNOWN（`$2.5$, or 7.`、`$1$, or $2$.`、`1,\n2.`）；
  数字与公式候选共用同一套完整区域规则；回归含正例与反例表，不只匹配这三个字符串；
  未扩大单位/集合/区间/坐标/复数的判分范围。
* **实际 commit 接线**：解析后的数据 commit 同时传入数据加载、选题排序与运行头；缺失即拒绝继续。
  核验走真实读题器 + 运行头构造，不只测 pilot 函数。运行头记录协议版本、种子、完整排除 ID 清单、
  选题规则所用实际 commit、逐层分母与 subject/level 组成、`selected_ids_sha16`、`code_version`。
* **完整分母**：摘要从运行头导出入组/首轮/各臂分母，覆盖“首轮中断”“两臂之一缺失”“两臂都有但不可判定”
  “无运行头历史日志”四类；JSON 与文本一致。
* **交接顺序**：**先部署指定版本，再核对环境与离线检查**；版本不符或导入失败停止回传，不自动安装/升级。

## 10. 预算与未验证路径

* 下一次先导试运行：先导题量**最多 48 道**（排除 R06 已暴露的 12 题、只取标量金标准，按稳定哈希取前 N，
  实际入组看 `selection.layers` 与 `shortfall`）；首轮每题 1 次，首轮正常完成且判错的题 A/B 各 1 次；
  单张 4090；**外层总截止 44 分钟、用户总窗口 ≤45 分钟**，含模型加载。
* 安装与下载单独计，不占用这 45 分钟；正式计时之前完成缓存。
* 代码版本按 Git 交接（[VERSION_CONTROL.md](VERSION_CONTROL.md)、[SERVER_SYNC.md](SERVER_SYNC.md)）：
  PI 验收后指定 commit/tag，运行头记录实际源码版本与是否有未提交改动，不再靠人工压缩包对齐文件。
* **未验证**：真实 tokenizer 的 chat template 行为、真实模型生成、显存与吞吐、真实数据读取与 commit 解析、
  SymPy 分支。离线检查使用替身后端与合成题池，**mock 通过 ≠ GPU 烟测通过**。
