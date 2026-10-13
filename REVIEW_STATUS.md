# 当前代码审阅状态

2026-10-13：**R11 新增 AIME 2024 接入，待 PI 审阅；未发布、未推送、未打标签。**

已审基线：`r10-reviewed-20261013`（实现审阅对象 `3b70102`），本轮起点。
本轮新增（**尚未审阅**）：`config_aime24.json`、`exposed_problems_aime24.json`、
`x002/judge.py` 的 `gold_value()`（只脱外层 `\boxed{}`）、`x002/worker.py` 的
`row_problem_id()` / `row_gold()` / `_index_rows()`（可配置题号字段与命名空间、拒绝缺失/重复身份）、
以及 7 项新增离线用例。

未测：真实 `datasets` 加载链路（本机未安装 `datasets`；用替身载入原始 rows 核验）、GPU 吞吐、
AIME 自然错误数量、截断率与可纠正性——这些都要靠目标机的**一次**真实试运行。

旧 MATH 配置与历史结果只读；X-002 仍非正式论文候选。可运行版本需 PI 审阅后指定。
