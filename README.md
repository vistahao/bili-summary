# bili-summary

阶段 3 的命令行实现。默认命令仍是离线预览；显式使用 `--execute` 后，程序只获取哔哩哔哩元数据和已有平台字幕，不下载视频或音频，然后通过只读临时 `codex exec` 生成学习资料。长视频可使用分片缓存和失败恢复。

## 为什么使用这些工具

- Python 诞生于 1991 年，强调可读性和“自带电池”的标准库。本项目目前使用 Ubuntu 22.04 自带的 Python 3.10、HTTP、JSON、子进程和测试模块，所以仍没有第三方 Python 依赖或锁定文件。
- `venv` 自 Python 3.3 起进入标准库，用来隔离项目解释器环境。当前 `.venv` 不带 pip，因为项目仍没有第三方 Python 依赖；出现第一项真实依赖时再补充 pip 和锁定文件。
- `pyproject.toml` 源于 Python 社区自 2016 年开始统一构建配置的工作。当前记录项目元数据和未来命令入口，不执行全局安装。
- `codex exec` 是 Codex CLI 的非交互模式。程序使用只读沙箱、临时会话、JSON Schema 和 JSONL 用量事件，不安装 OpenAI SDK，也不需要单独配置 OpenAI API Key。
- FFmpeg 和 `ffprobe` 留到阶段 4。本阶段不安装播放器，也不读取 MP4 的媒体内容。

## 运行

```bash
./bili-summary --help
./bili-summary doctor
./bili-summary run "https://www.bilibili.com/video/BV1fKtN6DErG" --subject "计算机"
./bili-summary --config config.ini run "https://www.bilibili.com/video/BV1fKtN6DErG" --subject "计算机" --execute
./bili-summary --config config.ini run "BV1pb8o6yE8f" --subject "计算机" --course "生成式软件工程" --long --compare-deep --execute
./bili-summary run-file "/mnt/d/Downloads/04_公考事业编/视频课程/行测/20250828 片段刷题4.mp4" --subject "公考事业编" --course "行测"
```

`run` 不加 `--execute` 时只输出离线预览。`run-file` 在阶段 4 前也只预览；它默认读取文件元数据但不计算全文哈希，显式增加 `--hash` 才会顺序读取文件并计算 SHA-256。

同一成果目录已经完整时，程序不会重复调用 Codex。长流程每完成一片就保存结构化缓存；中断后再次运行同一命令，只执行未完成步骤。只有显式使用 `--force` 才忽略缓存并重新产生全部调用。写文件采用“同目录临时文件 + 原子替换”。

`config.ini` 的 `[long_processing]` 包含四个关键阈值：普通片目标/上限为 15/20 分钟，Deep 审校片目标/上限为 50/55 分钟。小片恢复更细，大片重复的模型系统开销更少；调整前应先用真实任务比较 token 与遗漏情况。

阶段 2 的样例成果位于 `/home/dev/bili-summary-data/计算机/失败的AI Slop数字永生尝试/`。该次任务转换平台字幕 88 段，真实调用 Codex 1 次；`source.json` 记录模型、CLI 版本、token、耗时和处理状态。重复运行已经验证不会产生第二次调用。

阶段 3 的样例成果位于 `/home/dev/bili-summary-data/计算机/生成式软件工程/欢迎来到未来 [01-Raw-26生成式软件工程-NJU]/`。100 分钟字幕分为 6 个主片、1 次总结合并和 2 个 Deep 审校片，共 9 次调用；重复运行已验证为 0 次新调用。

## B 站登录态

平台当前可能只向登录用户返回 AI 字幕。程序不会读取浏览器数据库；如公开接口没有字幕，按以下方式把 Cookie 保存到项目内被 Git 忽略的秘密目录：

1. Windows 浏览器登录哔哩哔哩，打开目标视频，按 `F12` 进入“网络/Network”，刷新页面。
2. 选择域名为 `api.bilibili.com` 的 `x/player/...` 请求，在“请求标头/Request Headers”中复制完整 `Cookie` 值。不要把它发到聊天。
3. 在 WSL 终端执行下面命令。终端等待时粘贴 Cookie 并按回车；`read -s` 不回显，Cookie 也不会进入 shell 历史。

```bash
mkdir -p -m 700 /home/dev/projects/bili-summary/.secrets
umask 077
read -rsp "粘贴 B 站 Cookie 后按回车：" BILI_SUMMARY_COOKIE; printf '%s' "$BILI_SUMMARY_COOKIE" > /home/dev/projects/bili-summary/.secrets/bilibili_cookie.txt; unset BILI_SUMMARY_COOKIE; printf '\n已保存\n'
./bili-summary --config config.ini doctor --json
```

Cookie 等同登录凭据，失效后重新生成；不要提交、备份到普通网盘或发到聊天。退出 B 站账号通常会使旧 Cookie 失效。

## 测试

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

测试只使用临时小文件，不读取用户的真实视频。

## 文件与备份边界

- 必须备份：源代码、测试、`plan.md`、非秘密配置模板、以后产生的依赖锁定文件。
- 单独备份：`/home/dev/bili-summary-data/` 中的学习成果，以及用户自己的原视频库。
- 密码管理器保存：B 站 Cookie、未来的 API Key、App Key、Access Token；它们不能进入仓库或聊天。
- 不备份：`.venv/`、`/home/dev/bili-summary-data/.bili-summary-cache/`、日志、临时音频和可重新生成的构建产物。结构化切片缓存当前很小，用于失败恢复；成果验收后可以重建，不属于长期学习资料。

当前关键配置模板是 `config.example.ini`，它随 Git 备份；本机的 `config.ini` 被 Git 忽略，可从模板重建。真正需要单独长期备份的是 `/home/dev/bili-summary-data/`；`.venv`、`.secrets` 和临时文件不应随项目迁移，凭据从密码管理器重新配置。
