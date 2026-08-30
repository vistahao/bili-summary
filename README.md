# bili-summary

阶段 1 的离线命令行骨架。它目前只验证输入、配置和归档位置，不下载视频、不读取媒体轨、不调用 Codex 或转写 API，也不写入正式成果目录。

## 为什么使用这些工具

- Python 诞生于 1991 年，强调可读性和“自带电池”的标准库。本阶段使用 Ubuntu 22.04 自带的 Python 3.10、`argparse`、`configparser` 和 `unittest`，所以没有第三方依赖，也没有需要锁定的第三方版本。
- `venv` 自 Python 3.3 起进入标准库，用来隔离项目解释器环境。当前 `.venv` 不带 pip，因为阶段 1 没有需要安装的包；出现第一项真实第三方依赖时再补充 pip 和锁定文件。
- `pyproject.toml` 源于 Python 社区自 2016 年开始统一构建配置的工作。当前只记录项目元数据和未来命令入口，不在阶段 1 执行打包安装。
- FFmpeg 和 `ffprobe` 留到阶段 4。本阶段不安装播放器，也不读取 MP4 的媒体内容。

## 运行

```bash
./bili-summary --help
./bili-summary doctor
./bili-summary run "https://www.bilibili.com/video/BV1fKtN6DErG" --subject "计算机"
./bili-summary run-file "/mnt/d/Downloads/04_公考事业编/视频课程/行测/20250828 片段刷题4.mp4" --subject "公考事业编" --course "行测"
```

上述 `run` 和 `run-file` 只输出离线预览。`run-file` 默认读取文件元数据但不计算全文哈希；显式增加 `--hash` 才会顺序读取文件并计算 SHA-256。

## 测试

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

测试只使用临时小文件，不读取用户的真实视频。

## 文件与备份边界

- 必须备份：源代码、测试、`plan.md`、非秘密配置模板、以后产生的依赖锁定文件。
- 单独备份：`/home/dev/bili-summary-data/` 中的学习成果，以及用户自己的原视频库。
- 密码管理器保存：未来的 API Key、App Key、Access Token；它们不能进入仓库或聊天。
- 不备份：`.venv/`、缓存、日志、临时音频和可重新生成的构建产物。

当前关键配置模板是 `config.example.ini`。用户自己的 `config.ini` 被 Git 忽略；即使丢失，也不会损坏已有 Markdown，但需要重新指定数据根目录和处理偏好。
