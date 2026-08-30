# bili-summary

阶段 3.1 已验收提交，完成首次真实模型对比和长期默认路由选择。默认命令仍是离线预览；显式使用 `--execute` 后，程序先取得已有平台字幕，再于首次文本调用前一次性确认本次方案。长期默认由 DeepSeek Flash high 生成完整整理稿、Codex high 生成学习总结、DeepSeek Pro high 生成 Basic/Deep 审校报告。

## 为什么使用这些工具

- Python 诞生于 1991 年，强调可读性和“自带电池”的标准库。本项目目前使用 Ubuntu 22.04 自带的 Python 3.10、HTTP、JSON、子进程和测试模块，所以仍没有第三方 Python 依赖或锁定文件。
- `venv` 自 Python 3.3 起进入标准库，用来隔离项目解释器环境。当前 `.venv` 不带 pip，因为项目仍没有第三方 Python 依赖；出现第一项真实依赖时再补充 pip 和锁定文件。
- `pyproject.toml` 源于 Python 社区自 2016 年开始统一构建配置的工作。当前记录项目元数据和未来命令入口，不执行全局安装。
- `codex exec` 是 Codex CLI 的非交互模式。程序使用只读沙箱、临时会话、JSON Schema 和 JSONL 用量事件，不安装 OpenAI SDK，也不需要单独配置 OpenAI API Key。
- DeepSeek 使用 Python 标准库直接发送非流式 HTTPS 请求，不安装 OpenAI SDK。模拟错误测试和 Flash/Pro 真实 JSON 调用均已通过。
- FFmpeg 和 `ffprobe` 留到阶段 4。本阶段不安装播放器，也不读取 MP4 的媒体内容。

## 运行

```bash
./bili-summary --help
./bili-summary doctor
./bili-summary run "https://www.bilibili.com/video/BV1fKtN6DErG" --subject "计算机"
./bili-summary --config config.ini run "https://www.bilibili.com/video/BV1fKtN6DErG" --subject "计算机" --execute
./bili-summary --config config.ini run "BV1pb8o6yE8f" --subject "计算机" --course "生成式软件工程" --audit-level "deep" --execute
./bili-summary --config config.ini run "BV1pb8o6yE8f" --subject "计算机" --profile "quality" --yes --execute
./bili-summary --config config.ini compare-text "/path/to/字幕.srt" --profiles deepseek_flash_low deepseek_pro_high --yes
./bili-summary run-file "/mnt/d/Downloads/04_公考事业编/视频课程/行测/20250828 片段刷题4.mp4" --subject "公考事业编" --course "行测"
```

`run` 不加 `--execute` 时只输出离线预览。`run-file` 在阶段 4 前也只预览；它默认读取文件元数据但不计算全文哈希，显式增加 `--hash` 才会顺序读取文件并计算 SHA-256。

交互终端会在取得字幕、估算切片后显示一次完整方案；直接按 Enter 使用配置默认值，`1` 修改本次任务，`2`/`3` 使用质量/速度预设，`q` 取消且不发送字幕。非交互运行必须显式增加 `--yes`，并可用 `--profile` 或重复的 `--route "summary=配置名"` 覆盖。

同一成果目录已经完整时，程序不会重复调用文本模型。每个任务和切片完成后保存结构化缓存；缓存指纹包含任务、后端、模型、推理档位、提示词和 Schema。中断后只执行缺失步骤，切换配置不会误用旧结果；显式使用 `--force` 才按本次方案重做。

文本长期默认值位于 `[text_routes]`，可复用配置位于 `[text_profile.<名称>]`，整体预设位于 `[text_preset.<名称>]`。`config.example.ini` 展示完整结构。DeepSeek Key 只能通过环境变量或权限为 `600` 的 Git 忽略文件提供；程序只报告“是否已配置”，不会把密钥写入 `source.json`、缓存或日志。

审校提示词优先检查知识与逻辑错误，包括概念混淆、前后矛盾、因果倒置、关键前提缺失、数字或量纲异常。普通口语、语法、文风、大小写和标点不进入报告；字幕错误只有在高置信且会改变知识含义时才上报。这样可以降低“讲者说得不够书面”一类无用猜测。

`config.ini` 的 `[long_processing]` 包含四个关键阈值：普通片目标/上限为 15/20 分钟，Deep 审校片目标/上限为 50/55 分钟。小片恢复更细，大片重复的模型系统开销更少；调整前应先用真实任务比较 token 与遗漏情况。

阶段 2 的样例成果位于 `/home/dev/bili-summary-data/计算机/失败的AI Slop数字永生尝试/`。该次任务转换平台字幕 88 段，真实调用 Codex 1 次；`source.json` 记录模型、CLI 版本、token、耗时和处理状态。重复运行已经验证不会产生第二次调用。

阶段 3 的样例成果位于 `/home/dev/bili-summary-data/计算机/生成式软件工程/欢迎来到未来 [01-Raw-26生成式软件工程-NJU]/`。100 分钟字幕分为 6 个主片、1 次总结合并和 2 个 Deep 审校片，共 9 次调用；重复运行已验证为 0 次新调用。

阶段 3.1 的同字幕对比位于 `/home/dev/bili-summary-data/模型对比/失败的AI Slop数字永生尝试-4451ca881535/`。8 项 Flash/Pro 调用全部通过结构和时间戳校验；`对比说明.md` 保存调用合计与明细，`质量观察.md` 保存抽查判断和最终路由决定。旧审校文件保留为原提示词的历史证据；收紧后的知识/逻辑审校规则从后续任务生效。再次运行相同版本会复用 `.comparison-cache/`，不会重复付费。

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
- 单独备份：`/home/dev/bili-summary-data/` 中的学习成果和模型对比记录，以及用户自己的原视频库。
- 密码管理器保存：B 站 Cookie、未来的 API Key、App Key、Access Token；它们不能进入仓库或聊天。
- 不备份：`.venv/`、`.bili-summary-cache/`、`.comparison-cache/`、日志、临时音频和可重新生成的构建产物。结构化缓存用于失败恢复和避免重复调用；结果文件验收后可以重建，不属于长期学习资料。

当前关键配置模板是 `config.example.ini`，它随 Git 备份；本机的 `config.ini` 被 Git 忽略，可从模板重建。真正需要单独长期备份的是 `/home/dev/bili-summary-data/`；`.venv`、`.secrets` 和临时文件不应随项目迁移，凭据从密码管理器重新配置。
