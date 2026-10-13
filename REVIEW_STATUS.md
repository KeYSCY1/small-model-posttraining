# 当前代码审阅状态

2026-10-13：**R12 收口已提交，待 PI 审阅；未发布、未推送、未打标签。**

已审基线：`r10-reviewed-20261013`（MATH-500 历史设置，只读参考）。

R12 修正 R11 审阅指出的三处：
1. **题单**：71/81/88 经完整题面核对后判定文字条件完整、图示仅为示例 → **不再排除**；题单从固定版本**完整 30 题池**按稳定哈希规则重算为 12 题，摘要 `951787dd1e77c128`（与 PI 以真实 `DatasetReader → freeze_selection` 复现的一致）；新增 `frozen_problems_aime24.json`，并在**加载模型之前**由 `verify_frozen_selection()` 核对题单/摘要/commit/协议，不一致即停止。
2. **源码摘要**：`MANIFEST_PATHS` 纳入 `config_aime24.json`、`exposed_problems_aime24.json`、`frozen_problems_aime24.json`，改动这些受控文件会使 `files_sha16` 变化。
3. **操作正文**：README / SERVER_SYNC 改为**只给 AIME 当前步骤**（含固定数据预缓存命令、`code_version.txt` 生成、AIME 目录打包），MATH 只留历史索引。

离线检查 68 项；真实 GPU 生成、截断率与可纠正性仍待目标机那一次运行。
