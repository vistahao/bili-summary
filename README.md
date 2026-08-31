# bili-summary

阶段 4 已验收：本地 MP4、Qwen3 默认转写、Paraformer 回退和显式 whisper.cpp 兜底均已接入统一文本流程；刷题实跑暴露的内容边界问题已通过 `lecture/practice` 模式和“审校→整理→总结”依赖修正。当前进入阶段 5 的缓存、备份、迁移与极长视频可靠性工作。默认命令仍是离线预览；显式使用 `--execute` 后，程序在费用门槛和文本方案确认通过后才调用模型。

## 为什么使用这些工具

- Python 诞生于 1991 年，强调可读性和“自带电池”的标准库。本项目目前使用 Ubuntu 22.04 自带的 Python 3.10、HTTP、JSON、子进程和测试模块，所以仍没有第三方 Python 依赖或锁定文件。
- `venv` 自 Python 3.3 起进入标准库，用来隔离项目解释器环境。当前 `.venv` 不带 pip，因为项目仍没有第三方 Python 依赖；出现第一项真实依赖时再补充 pip 和锁定文件。
- `pyproject.toml` 源于 Python 社区自 2016 年开始统一构建配置的工作。当前记录项目元数据和未来命令入口，不执行全局安装。
- `codex exec` 是 Codex CLI 的非交互模式。程序使用只读沙箱、临时会话、JSON Schema 和 JSONL 用量事件，不安装 OpenAI SDK，也不需要单独配置 OpenAI API Key。
- DeepSeek 使用 Python 标准库直接发送非流式 HTTPS 请求，不安装 OpenAI SDK。模拟错误测试和 Flash/Pro 真实 JSON 调用均已通过。
- FFmpeg 源于 2000 年启动的开源多媒体项目；`ffprobe` 只读检查轨道，`ffmpeg` 只在没有字幕时准备音频。当前 Ubuntu `ffmpeg` 软件包已提供两者，没有安装播放器或 Python 视频库。

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
./bili-summary --config config.ini run-file "/mnt/d/Downloads/04_公考事业编/视频课程/行测/20250828 片段刷题4.mp4" --probe --json
./bili-summary --config config.ini run-file "/mnt/d/Downloads/04_公考事业编/视频课程/行测/20250828 片段刷题4.mp4" --prepare-audio-sample --sample-start 1440 --sample-minutes 10 --json
./bili-summary --config config.ini run-file "/mnt/d/Downloads/04_公考事业编/视频课程/行测/20250828 片段刷题4.mp4" --subject "公考事业编" --course "行测" --long --content-mode practice --audit-level basic --execute --yes --json
./bili-summary --config config.ini check-aliyun-asr --json
./bili-summary --config config.ini compare-aliyun-asr "/path/to/5-10分钟样本.wav" --yes --json
./bili-summary --config config.ini cache-status --json
./bili-summary --config config.ini cache-clean --json
# 到期后先预览，再显式执行；非交互执行还必须增加 --yes
./bili-summary --config config.ini cache-clean --execute --yes --json
```

`run` 不加 `--execute` 时只输出离线预览。`run-file` 默认也只读取文件元数据，不计算全文哈希；`--probe` 增加只读轨道检查。`--prepare-audio-sample` 会自动计算 SHA-256、确认确实没有可用字幕，再生成单声道 16 kHz、16-bit WAV；样本时长只允许 5～10 分钟，默认 10 分钟。该命令不调用语音或文本模型。`run-file --execute` 才进入正式流程，文本来源顺序是同名外置 SRT、可提取的内嵌字幕、Qwen3、Paraformer、显式配置的本地 whisper.cpp；在线转写前显示完整时长的最坏费用估算并应用本机提交门槛。

转写样本保存在数据根目录的 `.bili-summary-cache/local-<哈希前缀>/media/`，与长期成果分离。元数据记录来源哈希、音频规格、最后成功使用时间和“最后使用后 5 天可清理”；重复命令会复用同一配置的样本并顺延时间。目前只标记清理资格，不自动删除文件。真实样例使用 24:00～34:00 的连续讲课段；视频开头 10 分钟以背景音为主，不适合作为识别质量样本。

阶段 4 的在线范围已收敛为阿里云 `qwen3-asr-flash-filetrans` 与 `paraformer-v2`；当前价格、数据边界和调用前清单位于 [`docs/stage4-transcription-evaluation.md`](docs/stage4-transcription-evaluation.md)。两者共用标准库异步 HTTP 客户端，不安装 DashScope SDK；该记录不包含密钥或音频正文。

`check-aliyun-asr` 只为两个模型申请短时私有上传策略，用来检查 API Key 鉴权和模型绑定；它不上传音频、不提交转写，也不产生识别费用。阿里云 Key 只能通过 `DASHSCOPE_API_KEY` 环境变量或 `[aliyun_asr] api_key_file` 指向的权限 `600` 文件提供。该预检成功说明账号能取得模型专用上传权限，最终推理权限仍由首次 10 分钟实际转写确认。

`compare-aliyun-asr` 会校验 5～10 分钟、单声道 16 kHz、16-bit WAV，把同一文件分别上传为两个模型绑定的私有临时对象，完成异步转写后保存厂商原始 JSON 和统一 SRT。临时对象预计 48 小时后失效，不备份；对比成果位于数据根目录并应随学习成果备份。

交互终端会在取得字幕、估算切片后显示一次完整方案；直接按 Enter 使用配置默认值，`1` 修改本次任务，`2`/`3` 使用质量/速度预设，`q` 取消且不发送字幕。非交互运行必须显式增加 `--yes`，并可用 `--profile` 或重复的 `--route "summary=配置名"` 覆盖。

内容模式默认是 `lecture`，适合知识讲座；刷题课显式使用 `--content-mode practice`。刷题模式保留题意、答案、教师推理、选项辨析、可迁移方法和易错点，过滤课前歌曲、点名、收音确认、投票等待、无教学作用的正确率播报与闲聊。连续删除区间只在完整整理稿中保留一行时间范围，不复述或解释歌词。

长流程的依赖顺序是“原始字幕 → Basic 审校 → 完整整理 → 分片总结 → 最终总结”。整理只采用有上下文支持的高置信字幕修正；知识或逻辑风险不会被擅自改写，只作为讲者观点的表述约束并进入审校报告。Deep 审校仍是独立的可选全文风险检查。切换内容模式不会重新转写音频；在文本路由和字幕未变时，Basic 审校缓存仍可复用，整理与总结会按新模式重做。

`cache-status` 只读统计缓存总量、受管临时音频和5天到期时间。`cache-clean` 默认仍是只读预览；只有增加 `--execute` 并在交互终端输入 `clean`，或非交互时同时增加 `--yes`，才删除已到期且通过路径、文件名、元数据和符号链接检查的临时 WAV 及对应元数据。它不会删除字幕、厂商原始响应、文本恢复缓存或用户成果。

同一成果目录已经完整时，程序不会重复调用文本模型。每个任务和切片完成后保存结构化缓存；缓存指纹包含任务、后端、模型、推理档位、提示词和 Schema。中断后只执行缺失步骤，切换配置不会误用旧结果；显式使用 `--force` 才按本次方案重做。

文本长期默认值位于 `[text_routes]`，可复用配置位于 `[text_profile.<名称>]`，整体预设位于 `[text_preset.<名称>]`。`config.example.ini` 展示完整结构。DeepSeek Key 只能通过环境变量或权限为 `600` 的 Git 忽略文件提供；程序只报告“是否已配置”，不会把密钥写入 `source.json`、缓存或日志。

审校提示词优先检查知识与逻辑错误，包括概念混淆、前后矛盾、因果倒置、关键前提缺失、数字或量纲异常。普通口语、语法、文风、大小写和标点不进入报告；字幕错误只有在高置信且会改变知识含义时才上报。这样可以降低“讲者说得不够书面”一类无用猜测。

`config.ini` 的 `[long_processing]` 包含四个关键阈值：普通片目标/上限为 15/20 分钟，Deep 审校片目标/上限为 50/55 分钟。小片恢复更细，大片重复的模型系统开销更少；调整前应先用真实任务比较 token 与遗漏情况。

阶段 2 的样例成果位于 `/home/dev/bili-summary-data/计算机/失败的AI Slop数字永生尝试/`。该次任务转换平台字幕 88 段，真实调用 Codex 1 次；`source.json` 记录模型、CLI 版本、token、耗时和处理状态。重复运行已经验证不会产生第二次调用。

阶段 3 的样例成果位于 `/home/dev/bili-summary-data/计算机/生成式软件工程/欢迎来到未来 [01-Raw-26生成式软件工程-NJU]/`。100 分钟字幕分为 6 个主片、1 次总结合并和 2 个 Deep 审校片，共 9 次调用；重复运行已验证为 0 次新调用。

阶段 3.1 的同字幕对比位于 `/home/dev/bili-summary-data/模型对比/失败的AI Slop数字永生尝试-4451ca881535/`。8 项 Flash/Pro 调用全部通过结构和时间戳校验；`对比说明.md` 保存调用合计与明细，`质量观察.md` 保存抽查判断和最终路由决定。旧审校文件保留为原提示词的历史证据；收紧后的知识/逻辑审校规则从后续任务生效。再次运行相同版本会复用 `.comparison-cache/`，不会重复付费。

阶段 4C 的完整本地样例成果位于 `/home/dev/bili-summary-data/公考事业编/行测/20250828 片段刷题4/`。86 分 11 秒的无字幕 MP4 经 Qwen3 返回 832 个句段，厂商报告计费用量 3,475 秒、处理耗时 40.311 秒；随后分为 6 个主片，完成 19 个文本任务。一次末段审校传输中断后，原命令只重试缺失步骤；该实跑又促使 DeepSeek 适配器对不完整 HTTP 响应增加一次有界自动重试。全部完成后再次运行用时约 1.47 秒，未提取音频，也未调用转写或文本模型。源视频 SHA-256 复核未变。

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
- 不备份：`.venv/`、`.bili-summary-cache/`、`.comparison-cache/`、日志、临时音频和可重新生成的构建产物。结构化缓存用于失败恢复和避免重复调用；转写样本在最后成功使用 5 天后才具备清理资格，但当前版本不会自动删除。

当前关键配置模板是 `config.example.ini`，它随 Git 备份；本机的 `config.ini` 被 Git 忽略，可从模板重建。真正需要单独长期备份的是 `/home/dev/bili-summary-data/`；`.venv`、`.secrets` 和临时文件不应随项目迁移，凭据从密码管理器重新配置。
