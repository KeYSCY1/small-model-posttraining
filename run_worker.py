#!/usr/bin/env python3
"""工作进程入口脚本（由 x002.cli 的父进程监督调用）。

这里把**运行包所在目录**加入 ``sys.path`` 后再调用 ``x002.worker.worker_main``，
因此在正式路径与离线检查路径下都是同一个入口，且工作进程内可 import
仓库根的测试替身（``tests.fake_backend``）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from x002.worker import worker_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(worker_main())
