# 服务器同步实验代码

私有仓库：https://github.com/KeYSCY1/small-model-posttraining

Git管理代码和配置；模型与数据继续使用服务器已有缓存，虚拟环境沿用上次成功运行的环境。原始结果留在runs/，继续按实验交接回传结果包。

## 首次获取

在服务器选用新的代码目录，不覆盖此前的结果目录。服务器需要能访问GitHub，并完成私有仓库的Git身份认证。

```bash
cd ~/autodl-tmp
git clone https://github.com/KeYSCY1/small-model-posttraining.git
cd small-model-posttraining
git status --short
git rev-parse HEAD
```

代码已经在仓库根目录，不必再进入run_package子目录。私有仓库若提示认证，需要在服务器完成GitHub认证；浏览器登录状态不会自动传到服务器。

**当前只可同步，不启动模型。** `r07-submitted-20261013` 是待修快照；R08及PI审阅通过后会另给精确的可运行commit或标签。

## 后续更新

先结束当前作业，再进入服务器这份代码目录。执行 `git status --short`，应没有输出；有本地修改时先停止更新并保留差异，不强制覆盖。

```bash
git fetch origin --tags
```

随后按PI交接给出的**具体commit或标签**执行 `git switch --detach`。此处不提供可直接运行的占位命令，避免把未审版本当成可运行版本。切换后核对 `git rev-parse HEAD` 与交接版本一致，再在原有环境中运行该版要求的离线检查。检查通过及运行准入明确后，才执行实验命令。

后续无需逐个上传修改文件或混用压缩包。runs/及日志被.gitignore排除，切换正常代码版本不会将它们纳入版本历史；实验结果仍需单独保存和回传。
