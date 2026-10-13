# X-002 运行包（AIME 2024 一次性先导；仓库根即运行包）

**当前动作：R12工程审阅已通过；AIME先导待剔除一处源题误写后发布，暂不执行生成。** PI独立68项全部通过、无跳过；当前配置仍含已确认转录错误的aime24:88，R13仅排除该题并保留其余11题，未推送/未打标签。
本轮只做本地编写与无模型离线检查；未安装、未下载模型、未运行 tokenizer/模型、未操作目标机。
**仓库根目录就是这个运行包**：`git clone` 后直接在仓库根执行命令，**不要再 `cd run_package`**。

## 1. 当前入口（AIME 2024）

```bash
python3 -m x002.cli run --config config_aime24.json        # 一次性运行
python3 -m x002.cli summarize --out-dir runs/x002_aime24_pilot_r11   # 只读汇总
```

* 数据：`math-ai/aime24` test，**固定版本** `83a7f387baaa524a8bda0022eac0541582297103`；
  字段 `id`/`problem`/`solution`/`url`；题号命名空间 `aime24:`；金标准取 `solution` 并只脱去外层 `\boxed{}`。
* 题量：原池 **30 题** → **数据质量剔除 1**（`aime24:88`：源数据题面把两种情况都写成 outside/externally tangent，与金标准 127 对应的内切/外切条件不符，PI 已核对获 MAA 许可的原题页面；不修题面、不把该题失败计为自然错误、不补新题）→ 质量合格池 **29** → 稳定哈希取前 **11**；冻结清单 `frozen_problems_aime24.json`（摘要 `e7fd0ff6c3875ed1`），**加载模型之前**核对题单，不一致立即停止。
* 生成：`max_new_tokens=8192`（首轮与 A/B 两臂同值）；首轮每题 1 次，仅正常完成且判错的题进 A/B 且每臂 1 次。
* 协议 `x002-aime24-pilot-r11.1`；输出目录 `runs/x002_aime24_pilot_r11`；外层 44 分钟 / 用户总窗口 ≤45 分钟。
* 目录与文件：

```
<仓库根>/
  config_aime24.json            当前 AIME 配置（唯一推荐）
  frozen_problems_aime24.json   固定题单与摘要（生成前核对）
  exposed_problems_aime24.json  本项目 AIME 已暴露清单（当前为空）
  config.json                   历史：MATH-500 配置（仅作历史保留，不再作为当前步骤）
  run_offline_checks.sh         离线检查包装（68 项；失败即非零退出）
  x002/  tests/  README.md  REVIEW_STATUS.md  SERVER_SYNC.md  VERSION_CONTROL.md
```

**环境**：目标机既有成功环境**优先保留，不就地升级**；已测组合只有一组
（torch 2.5.1+cu121 / transformers 4.57.6 / datasets 3.6.0 / huggingface_hub 0.36.2）。
版本或导入不符 → 停止并回传，不自动安装。

**先跑离线检查**（退出码非零或结尾不是 `OK` 就停止）：

```bash
bash run_offline_checks.sh offline_checks.log    # 期望：Ran 68 tests … OK
```

（离线通过 **不代表** GPU 路径可用。）

**历史索引（不作为当前步骤）**：MATH-500 的 `config.json`、48 题先导与
`r10-reviewed-20261013` 标签属于已审历史设置，结果只读；AIME 适配不改变它们的含义。

## 2. 数据版本与题单（事前冻结）

* 固定 commit 的原始文件摘要已核对（SHA-256 `2602156e…94a38`，= 仓库记录）；逐题核查见
  `20261013/research_R12/AIME2024_数据核验与题单.json` 与 `AIME2024_三题题面核查.md`。
* 图依赖判定：只出现 `[asy]`/示例图引用**不构成**排除理由；71/81 保留；88 因**源数据题面误写**排除（与图示无关）。
* 选题规则与生产读题路径**同一函数**；运行头记录逐层分母、题单摘要与冻结核对结果。

## 3. 唯一推荐命令与回收

```bash
# 在仓库根目录执行
python3 -m x002.cli run --config config_aime24.json 2>&1 | tee console.log; echo "exit=${PIPESTATUS[0]}"
python3 -m x002.cli summarize --out-dir runs/x002_aime24_pilot_r11 | tee summary_console.log
git rev-parse HEAD > code_version.txt && git status --short >> code_version.txt
tar czf x002_aime24_pilot_r11_results.tar.gz code_version.txt offline_checks.log console.log summary_console.log runs/x002_aime24_pilot_r11
```

非零退出也**只做只读汇总与回收**，不重跑、不放宽选题、不换模型。

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
* **多答案/并列值按完整候选区域判定**：区域从候选之后取**最近的**边界——「空行」或
  「句末标点 + 空白 + 下一句首字母大写/数学式」；但**行首并列连接词**（`\nOr 3`）不算边界，
  它是备选答案的延续。区域内只要出现并列形态（`,` `;` `、` `or` `and` 之后紧跟数值，含跨换行）
  或同一行内还有别的完整数值，一律 **UNKNOWN**。
* 因此以下都是 UNKNOWN：`The answer is 1,2.` / `1, 2.` / `2.5, or 7.` / `$2.5$, or 7.` /
  `$1$, or $2$.` / `1,\n2.` / `\boxed{1}, or 2` / **`2,\nor 123.`** / **`2,\n123.`** /
  **`\boxed{2},\nor 123.`** / **`2.\nOr 3.`** / `1, 000.`。裸数字、公式、`\boxed{}`、`####`
  四类候选走**同一条**规则。
* 保留：合法千分位 `1,000.`（只有**完整合法千分位**才受支持，三位数备选值不会被当成千分位片段）、
  单个小数/分数、`\boxed{497}`/`\boxed{129}`；句末边界取最近，因此
  `The answer is 2. We used 3 terms.` 后面再补 `\n\nMore discussion.` 仍是同一个单答案判断；
  `The answer is 2.\nWe checked 3 alternatives.` 这类**普通后续说明**也不算第二答案。
* 未覆盖（如实写，不在本批扩范围）：不含 `answer is/:` 的裸并列数值仍按“抽不出答案”处理；
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

## 9.1 R08/R09 定点修补要点（供核对）

* **离线测试隔离**：版本传递用例**同时**替换 torch 与 transformers，并在 finally 里完整恢复原模块
  （含"原本不存在"的情形）；在未安装这两个依赖的环境里也通过；真实子进程哨兵、加载/生成阻塞、
  截止与进程清理检查全部保留。
* **判分答案区域（R09 收口）**：区域取**最近**边界；跨行三位数备选值（`2,\nor 123.`、`2,\n123.`、
  `\boxed{2},\nor 123.`）与句末换行延续（`2.\nOr 3.`）一律 UNKNOWN；只有完整合法千分位受支持；
  普通后续说明不机械算第二答案；裸数字/公式/`\boxed{}`/`####` 四类候选共用同一条规则；
  未扩大单位/集合/区间/坐标/复数的判分范围。
* **实际 commit 接线**：解析后的数据 commit 同时传入数据加载、选题排序与运行头；缺失即拒绝继续。
  核验走真实读题器 + 运行头构造，不只测 pilot 函数。运行头记录协议版本、种子、完整排除 ID 清单、
  选题规则所用实际 commit、逐层分母与 subject/level 组成、`selected_ids_sha16`、`code_version`。
* **完整分母**：摘要从运行头导出入组/首轮/各臂分母，覆盖“首轮中断”“两臂之一缺失”“两臂都有但不可判定”
  “无运行头历史日志”四类；JSON 与文本一致。
* **Git 产物与源码身份（R09/R10）**：`.gitignore` 恢复 R07 原始排除集并补齐实际产物
  （`.venv/`、`.env.*`、`*.pt`/`*.pem`、`cache/`、`datasets/`、`outputs/`、`results/`、`*.tar.gz`、`*.zip`、
  `*.log.raw`、`code_version.txt` 等）；`capture_code_version()` 只按受控清单 `MANIFEST_PATHS`
  的**规范顺序**（按路径字节升序）计算源码摘要，记录 `source_mode`(git/archive)、`dirty`、
  `manifest_files/missing`、`manifest_sha16`，不扫描日志、环境、缓存与权重；
  同一份完整源码在 **Git 与无 Git 归档两种模式下得到相同 `files_sha16`**（R10 收口），
  新增日志/环境/缓存不改变摘要，受控源码改动必然改变摘要并如实反映 `dirty`。
* **测试只碰临时 fixture（R10）**：源码身份用例在 `TemporaryDirectory` 下自建 fixture 或本地临时 Git
  克隆（只复制受控清单），**不在活动包目录里造日志/环境/缓存、也不改用户源码**；活动包只读。
  已用“预置 `.env`/缓存/日志与源码哨兵的一次性克隆”核验：运行实际离线包装后预置文件与受控源码字节均未变、
  工作树干净、只更新约定的检查日志。
* **交接顺序**：**先部署指定版本，再核对环境与离线检查**；版本不符或导入失败停止回传，不自动安装/升级。

## 10. 预算与未验证路径

* **历史（MATH-500，已审 R10，非当前步骤）**：先导题量最多 48 道（排除 R06 已暴露的 12 题、只取标量金标准，按稳定哈希取前 N，
  实际入组看 `selection.layers` 与 `shortfall`）；首轮每题 1 次，首轮正常完成且判错的题 A/B 各 1 次；
  单张 4090；**外层总截止 44 分钟、用户总窗口 ≤45 分钟**，含模型加载。
* 安装与下载单独计，不占用这 45 分钟；正式计时之前完成缓存。
* 代码版本按 Git 交接（[VERSION_CONTROL.md](VERSION_CONTROL.md)、[SERVER_SYNC.md](SERVER_SYNC.md)）：
  PI 验收后指定 commit/tag，运行头记录实际源码版本与是否有未提交改动，不再靠人工压缩包对齐文件。
* **未验证**：真实 tokenizer 的 chat template 行为、真实模型生成、显存与吞吐、真实数据读取与 commit 解析、
  SymPy 分支。离线检查使用替身后端与合成题池，**mock 通过 ≠ GPU 烟测通过**。
