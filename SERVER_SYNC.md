# 服务器同步实验代码

私有仓库：https://github.com/KeYSCY1/small-model-posttraining

Git 管理代码和配置；模型与数据继续使用服务器已有缓存，虚拟环境沿用上次成功运行的环境。
原始结果留在 `runs/`，继续按实验交接回传结果包。

## 首次获取

在服务器选用**新的代码目录**，不覆盖此前的结果目录。服务器需要能访问 GitHub，并完成私有仓库的
Git 身份认证（浏览器登录状态不会自动传到服务器）。

```bash
cd ~/autodl-tmp
git clone https://github.com/KeYSCY1/small-model-posttraining.git
cd small-model-posttraining
git status --short      # 应无输出
git rev-parse HEAD
```

代码已经在仓库根目录，不必再进入 `run_package` 子目录。

**当前只可同步，不启动模型。** `r07-submitted-20261013` 是待修快照；R08 修补尚待 PI 验收，
可运行 commit 或标签由 PI 在验收后给出。

## 后续更新

先结束当前作业，再进入服务器这份代码目录。执行 `git status --short`，应没有输出；
有本地修改时先停止更新并保留差异，不强制覆盖。

```bash
git fetch origin --tags
```

随后按 PI 交接给出的**具体 commit 或标签**执行 `git switch --detach <commit 或 tag>`。
此处不提供可直接运行的占位命令，避免把未审版本当成可运行版本。切换后：

```bash
git rev-parse HEAD         # 必须与交接版本一致
git status --short         # 必须无输出（运行头会记录 dirty 状态）
```

然后在**原有环境**中运行该版本要求的离线检查（`bash run_offline_checks.sh offline_checks.log`，
期望结尾 `Ran 50 tests … OK`）。检查通过并且运行准入明确后，才执行实验命令。

后续无需逐个上传修改文件或混用压缩包。`runs/` 及日志被 `.gitignore` 排除，
切换代码版本不会把它们纳入版本历史；实验结果仍需单独保存和回传。
