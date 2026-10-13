# 实验代码版本管理

唯一活动源码保留在现有 `run_package` 目录（不迁移、不复制第二份活动包）。该目录本身是 Git 仓库；
项目其余论文、阅读材料、日期归档和真实实验结果不纳入此代码仓库。

远端私有仓库：https://github.com/KeYSCY1/small-model-posttraining

本地 origin：https://github.com/KeYSCY1/small-model-posttraining.git

首个源码提交为 `db2b520a90a5a19735319b732aa0b49cca4aaffb`，已推送 main。
R07 交接源码状态快照：`eb909252512d3558c66727ba4dc61d21f03ee3c1`（本轮改动前的基线 commit）。
服务器操作见 [SERVER_SYNC.md](SERVER_SYNC.md)。

**本批（R08）改动以 Git diff 交付 PI 审阅；R07 源码与 R07/R06 原始结果目录均未被覆盖。**
本地离线检查通过只说明无模型路径可跑，**不构成运行准入**；可运行 commit/tag 由 PI 验收后确定。

## 日常流程

1. Harness 在本地这份源码中改动，并运行无模型离线检查。PI 通过 diff 审阅，确认后提交为一个明确版本。
2. 本地把该提交推送到私有仓库（需 PI/用户执行；Harness 不推送、不改远端设置）。
3. 服务器 `git clone`（首次）或 `git fetch` 后 `git switch --detach` 到 PI 给定的 commit/tag；
   切换前 `git status --short` 必须无输出，有本地改动先停止并保留差异，**不强制覆盖服务器修改**。
4. 运行头记录 `code_version`（Git commit + `dirty` 是否有未提交改动；无 Git 的归档包只记文件摘要，
   不伪造 commit）。
5. 服务器原始结果保留在 `runs/`，不加入 Git；结果回传仍使用原始日志包，可由运行头追溯代码版本。

## 版本标记

- `r07-submitted-20261013`：R07 提交待修历史快照，不能据此启动模型运行。
- 已审可运行版本：待 R08 修补经 PI 验收后确定（届时在此写明具体 commit 或 tag）。
- 服务器只运行明确交接的已审 commit 或 tag；在交接给出的版本之前**只可同步，不启动模型**。

后续更新不需要逐个文件复制，也不需要反复生成完整目录快照。不要在作业运行中切换版本。
区间内依赖版本未全部测过，不把“版本更高”当成兼容依据。

## 同步范围

Git 只跟踪源码、离线测试、配置、依赖说明、已暴露题号清单和操作文档。模型权重、数据缓存、虚拟环境、
密钥、日志、原始实验记录及压缩包由 `.gitignore` 排除；论文与阅读材料不在本仓库内。
