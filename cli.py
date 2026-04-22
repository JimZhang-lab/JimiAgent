'''
Author: JimZhang
Date: 2026-04-18 23:52:09
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:20:00
FilePath: /JimiAgent/cli.py
Description: CLI 入口。

'''
import asyncio
import os
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()


@click.group()
def cli():
    """JimiAgent 命令行工具。

    常用子命令: start / chat / status / doctor / migrate-env。
    """
    pass


@cli.command()
@click.option("--host", default=None, help="服务地址")
@click.option("--port", default=None, type=int, help="服务端口")
@click.option(
    "--verbose",
    is_flag=True,
    help="强制 DEBUG 日志等级（覆盖 yaml 的 logging.level）",
)
def start(host, port, verbose):
    """启动 Gateway 服务"""
    import uvicorn
    from server.config.settings import get_settings

    settings = get_settings()
    _host = host or settings.gateway.host
    _port = port or settings.gateway.port

    # --verbose 覆盖日志级别
    if verbose:
        settings.logging.level = "DEBUG"

    console.print(Panel.fit(
        f"[bold green]JimiAgent Gateway[/bold green]\n"
        f"HTTP : http://{_host}:{_port}\n"
        f"WS   : ws://{_host}:{_port}/ws/chat",
        title="启动中",
        border_style="green",
    ))

    uvicorn.run(
        "server.api.gateway:app",
        host=_host,
        port=_port,
        reload=False,
        log_level="info" if not verbose else "debug",
    )


@cli.command()
@click.option("--session", "-s", default=None, help="恢复指定会话 ID")
def chat(session):
    """终端交互式对话（React + Ink TUI）"""
    # 交给 Node TUI 接管 TTY
    import shutil
    from pathlib import Path

    project_root = Path(__file__).resolve().parent
    cli_js = project_root / "tui" / "dist" / "cli.js"
    node = shutil.which("node")

    if node is None:
        console.print(
            "[red]未找到 Node.js[/red]。TUI 需要 Node 20+。\n"
            "[dim]请安装 Node 后再试。[/dim]"
        )
        raise SystemExit(1)

    if not cli_js.exists():
        console.print(
            "[red]未找到 tui/dist/cli.js[/red]。请先构建：\n"
            "  [cyan]yarn --cwd tui install && yarn --cwd tui build[/cyan]"
        )
        raise SystemExit(1)

    env = dict(os.environ)
    if session:
        env["JIMI_TUI_SESSION"] = session
    # 让 worker 能导入 server/
    env["PYTHONPATH"] = (
        str(project_root)
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    # 固定使用当前解释器，避免命中错误的 Python
    if not env.get("JIMI_TUI_WORKER_CMD"):
        env["JIMI_TUI_WORKER_CMD"] = f"{sys.executable} -m server.core.tui_worker"

    # 透传用户启动 `jimi chat` 时的 cwd，避免 worker 把 project root 误当当前目录。
    env["JIMI_USER_CWD"] = os.getcwd()

    # 用 execvp 让 Node 接管当前进程
    os.execvpe(node, [node, str(cli_js)], env)


def _print_status(agent):
    """打印状态信息"""
    table = Table(title="系统状态", border_style="cyan")
    table.add_column("项目", style="bold")
    table.add_column("值")
    table.add_row("模型", f"{agent.settings.model.provider}/{agent.settings.model.model_id}")
    table.add_row("Skills", str(len(agent.workspace.list_skills())))
    table.add_row("会话数", str(len(agent.session_mgr.list_sessions())))
    table.add_row("Gateway", f"{agent.settings.gateway.host}:{agent.settings.gateway.port}")
    console.print(table)
    console.print()


# 会话与帮助展示已迁到 TUI。


@cli.command()
def status():
    """查看系统状态"""
    from server.core.agent import JimiAgent
    agent = JimiAgent()
    agent.initialize()
    _print_status(agent)


# sessions

@cli.group()
def sessions():
    """会话管理（list / delete）"""
    pass


@sessions.command("list")
def sessions_list():
    """列出所有会话（按更新时间倒序）"""
    from server.core.agent import JimiAgent
    import datetime

    agent = JimiAgent()
    agent.initialize()
    items = agent.session_mgr.list_sessions()

    table = Table(border_style="cyan", title=f"会话列表（共 {len(items)} 个）")
    table.add_column("ID", style="bold")
    table.add_column("标题")
    table.add_column("消息数", justify="right")
    table.add_column("最后更新")
    for s in items:
        updated = datetime.datetime.fromtimestamp(s.updated_at).strftime(
            "%Y-%m-%d %H:%M"
        )
        table.add_row(s.id, s.title, str(s.message_count), updated)
    console.print(table)


@sessions.command("delete")
@click.argument("session_id")
def sessions_delete(session_id):
    """删除指定会话（元信息 + checkpointer 线程）"""
    from server.core.agent import JimiAgent

    agent = JimiAgent()
    agent.initialize()
    # 先删元信息，再清理 checkpointer
    ok = agent.session_mgr.delete_session(session_id)
    if not ok:
        console.print(f"[yellow]未找到会话: {session_id}[/yellow]")
        raise SystemExit(1)
    try:
        asyncio.run(agent.clear_session_history(session_id))
    except Exception as e:
        console.print(f"[yellow]元信息已删但 checkpointer 清理失败: {e}[/yellow]")
    console.print(f"[green]已删除会话 {session_id}[/green]")


# scheduler

@cli.group()
def scheduler():
    """查看 / 触发定时任务（直连配置 + 本地 Scheduler，不经 Gateway）"""
    pass


@scheduler.command("list")
def scheduler_list():
    """列出所有 cron 任务及下次执行时间"""
    from server.core.agent import JimiAgent
    from server.core.scheduler import Scheduler

    agent = JimiAgent()
    agent.initialize()
    sched = Scheduler(agent, agent.settings.scheduler)
    jobs = sched.list_jobs()

    if not jobs:
        console.print("[yellow]无任务（scheduler.jobs 为空）[/yellow]")
        return
    table = Table(border_style="cyan", title="定时任务")
    table.add_column("ID", style="bold")
    table.add_column("schedule")
    table.add_column("enabled")
    table.add_column("下次执行")
    for j in jobs:
        table.add_row(
            j["id"],
            j["schedule"],
            "yes" if j["enabled"] else "no",
            j["next_run_at"] or "-",
        )
    console.print(table)


@scheduler.command("run")
@click.argument("job_id")
def scheduler_run(job_id):
    """立即触发一次（调用真实 LLM）"""
    from server.core.agent import JimiAgent
    from server.core.scheduler import Scheduler

    async def _run():
        agent = JimiAgent()
        await agent.ainitialize()
        sched = Scheduler(agent, agent.settings.scheduler)
        try:
            result = await sched.trigger_now(job_id)
            console.print(f"[green]job {job_id} 已触发[/green]")
            console.print(f"  last_status: {result['last_status']}")
            if result["last_error"]:
                console.print(f"  last_error:  {result['last_error']}")
            console.print(f"  run_count:   {result['run_count']}")
            console.print(f"  session_id:  {result['session_id']}")
        except KeyError:
            console.print(f"[red]未找到 job: {job_id}[/red]")
            raise SystemExit(1)
        finally:
            await agent.aclose()

    asyncio.run(_run())


# pairing

@cli.group()
def pairing():
    """DM Pairing / Allowlist 管理"""
    pass


@pairing.command("list")
def pairing_list():
    """列出 allowlist 和待审批的配对请求"""
    from server.plugin.channels.pairing import PairingStore
    from server.config.settings import get_settings

    s = get_settings()
    store = PairingStore(s.memory_abs_path.parent / "allowlist.json")

    allowed = store.list_allowed()
    if allowed:
        table = Table(border_style="cyan", title="Allowlist")
        table.add_column("Channel", style="bold")
        table.add_column("Sender IDs")
        for ch, sids in allowed.items():
            table.add_row(ch, ", ".join(sids) if sids else "(empty)")
        console.print(table)
    else:
        console.print("[yellow]Allowlist 为空[/yellow]")

    pending = store.list_pending()
    if pending:
        table2 = Table(border_style="yellow", title="待审批配对")
        table2.add_column("Channel", style="bold")
        table2.add_column("Sender ID")
        table2.add_column("Code", style="green")
        table2.add_column("剩余(s)", justify="right")
        for p in pending:
            table2.add_row(
                p["channel"], p["sender_id"],
                p["code"], str(p["remaining_seconds"]),
            )
        console.print(table2)
    else:
        console.print("[dim]无待审批配对请求[/dim]")


@pairing.command("approve")
@click.argument("channel")
@click.argument("code")
def pairing_approve(channel, code):
    """审批配对码：pairing approve <channel> <code>"""
    from server.plugin.channels.pairing import PairingStore
    from server.config.settings import get_settings

    s = get_settings()
    store = PairingStore(s.memory_abs_path.parent / "allowlist.json")
    sid = store.approve_code(channel, code)
    if sid:
        console.print(f"[green]已批准 {channel}/{sid}[/green]")
    else:
        console.print("[red]配对码无效或已过期[/red]")
        raise SystemExit(1)


@pairing.command("revoke")
@click.argument("channel")
@click.argument("sender_id")
def pairing_revoke(channel, sender_id):
    """吊销某 sender 的访问权限"""
    from server.plugin.channels.pairing import PairingStore
    from server.config.settings import get_settings

    s = get_settings()
    store = PairingStore(s.memory_abs_path.parent / "allowlist.json")
    if store.revoke(channel, sender_id):
        console.print(f"[green]已吊销 {channel}/{sender_id}[/green]")
    else:
        console.print(f"[yellow]未找到 {channel}/{sender_id}[/yellow]")


@pairing.command("add")
@click.argument("channel")
@click.argument("sender_id")
def pairing_add(channel, sender_id):
    """直接将 sender 加入白名单（免配对）"""
    from server.plugin.channels.pairing import PairingStore
    from server.config.settings import get_settings

    s = get_settings()
    store = PairingStore(s.memory_abs_path.parent / "allowlist.json")
    store.add_to_allowlist(channel, sender_id)
    console.print(f"[green]已加入 {channel}/{sender_id}[/green]")


# evolver

@cli.group()
def evolver():
    """Skill 自我修正 / 进化引擎（EvoMap 风格 GEP 协议）"""
    pass


@evolver.command("run")
@click.option(
    "--strategy", default="balanced",
    type=click.Choice(["balanced", "innovate", "harden", "repair-only"]),
    help="进化策略预设",
)
@click.option(
    "--review", is_flag=True, default=False,
    help="human-in-the-loop：对每条 event 提示 y/n/a/q 确认",
)
def evolver_run(strategy, review):
    """根据策略生成并输出进化提示"""
    from server.config.settings import get_settings
    from server.core.agent import JimiAgent

    async def _run():
        s = get_settings()
        agent = JimiAgent(s)
        try:
            await agent.ainitialize()
            events = agent.evolver.run_cycle(strategy=strategy, apply=False)
            if not events:
                console.print("[yellow]当前没有可进化 signal（或全部已处理）[/yellow]")
                return

            console.print(f"\n[bold]本轮生成 {len(events)} 条进化事件[/bold]\n")

            apply_all = False
            for i, ev in enumerate(events, 1):
                console.print(Panel(
                    Markdown(ev.prompt),
                    title=f"[{i}/{len(events)}] {ev.gene_id} · strategy={ev.strategy}",
                    border_style="cyan",
                ))
                if not review:
                    continue

                if apply_all:
                    choice = "y"
                else:
                    choice = click.prompt(
                        "应用此 event？[y]es / [n]o / [a]ll / [q]uit",
                        default="n",
                    ).strip().lower()

                if choice == "q":
                    console.print("[dim]已退出 review[/dim]")
                    break
                if choice == "a":
                    apply_all = True
                    choice = "y"
                if choice == "y":
                    result = await agent.evolver.apply_event(ev, agent)
                    console.print(Panel(
                        f"已在新 session [cyan]{result['session_id']}[/cyan] 派生修复任务\n\n"
                        f"Agent 首轮回复:\n{result['reply'][:500]}",
                        border_style="green",
                    ))
        finally:
            await agent.aclose()

    asyncio.run(_run())


@evolver.command("status")
@click.option("--limit", default=10, type=int, help="显示最近 N 条 events")
def evolver_status(limit):
    """显示最近 events + personality 状态"""
    from server.config.settings import get_settings
    from server.core.evolver import EvolutionEngine

    s = get_settings()
    engine = EvolutionEngine(
        gep_dir=s.memory_abs_path.parent / "gep",
        tool_log_path=s.memory_abs_path.parent / "tool_log.jsonl",
    )

    ps = engine.load_personality()
    console.print(Panel(
        f"mutation_count: [bold]{ps.mutation_count}[/bold]\n"
        f"last_evolution_ts: [dim]{ps.last_evolution_ts or '(无)'}[/dim]\n"
        f"focus_areas: {ps.focus_areas}",
        title="PersonalityState",
        border_style="cyan",
    ))

    events = engine.load_recent_events(limit=limit)
    if not events:
        console.print("[yellow]暂无历史 events[/yellow]")
        return

    table = Table(border_style="cyan", title=f"最近 {len(events)} 条 events")
    table.add_column("ts", style="dim")
    table.add_column("strategy")
    table.add_column("gene_id", style="bold")
    table.add_column("source")
    table.add_column("applied")
    for ev in events:
        table.add_row(
            ev.ts[:19],
            ev.strategy,
            ev.gene_id,
            (ev.signal or {}).get("source", "")[:24],
            "✓" if ev.applied else "",
        )
    console.print(table)


@evolver.command("signals")
def evolver_signals():
    """查看当前扫到的 signals（不写 event）"""
    from server.config.settings import get_settings
    from server.core.evolver import EvolutionEngine

    s = get_settings()
    engine = EvolutionEngine(
        gep_dir=s.memory_abs_path.parent / "gep",
        tool_log_path=s.memory_abs_path.parent / "tool_log.jsonl",
    )
    signals = engine.scan_signals()
    if not signals:
        console.print("[yellow]当前无信号[/yellow]")
        return
    table = Table(border_style="cyan")
    table.add_column("kind", style="bold")
    table.add_column("source")
    table.add_column("occurrences")
    table.add_column("detail", overflow="fold")
    for sig in signals:
        table.add_row(
            sig.kind, sig.source, str(sig.occurrences), sig.detail[:80],
        )
    console.print(table)


@evolver.command("genes")
def evolver_genes():
    """列出所有 Gene"""
    from server.config.settings import get_settings
    from server.core.evolver import EvolutionEngine

    s = get_settings()
    engine = EvolutionEngine(
        gep_dir=s.memory_abs_path.parent / "gep",
        tool_log_path=s.memory_abs_path.parent / "tool_log.jsonl",
    )
    genes = engine.load_genes()
    table = Table(border_style="cyan")
    table.add_column("id", style="bold")
    table.add_column("kind")
    table.add_column("intent")
    table.add_column("tags")
    for g in genes:
        table.add_row(g.id, g.kind, g.intent, ",".join(g.tags))
    console.print(table)


# message

@cli.group()
def message():
    """向 Agent 发送消息（不经过 Gateway，直接本地调用）"""
    pass


@message.command("send")
@click.argument("text")
@click.option("--session", "session_id", default=None, help="指定 session_id，默认自动")
def message_send(text, session_id):
    """发送一条消息给 Agent 并打印回复。

    用法: python cli.py message send "你好"
    """
    from server.config.settings import get_settings
    from server.core.agent import JimiAgent

    async def _run():
        s = get_settings()
        agent = JimiAgent(s)
        try:
            await agent.ainitialize()
            sid = session_id or agent.session_mgr.ensure_default_session().id
            resp, effective_sid = await agent.chat(text, sid)
            console.print(Panel(Markdown(resp), title=f"session={effective_sid}", border_style="green"))
        finally:
            await agent.aclose()

    asyncio.run(_run())


@message.command("stream")
@click.argument("text")
@click.option("--session", "session_id", default=None, help="指定 session_id，默认自动")
def message_stream(text, session_id):
    """流式发送一条消息给 Agent（逐 token 打印）。

    用法: python cli.py message stream "帮我写一首诗"
    """
    from server.config.settings import get_settings
    from server.core.agent import JimiAgent

    async def _run():
        s = get_settings()
        agent = JimiAgent(s)
        try:
            await agent.ainitialize()
            sid = session_id or agent.session_mgr.ensure_default_session().id
            async for event in agent.chat_stream(text, sid):
                etype = event.get("type")
                if etype == "text":
                    console.print(event.get("content", ""), end="")
                elif etype == "tool":
                    console.print(f"\n[dim]🔧 {event.get('name', '')}[/dim]", end="")
                elif etype == "error":
                    console.print(f"\n[red]{event.get('content', '')}[/red]")
            console.print()  # 末尾换行
        finally:
            await agent.aclose()

    asyncio.run(_run())


@cli.command("migrate-env")
@click.option("--env-file", default=".env", help=".env 文件路径")
@click.option("--yaml-file", default="config/agent_config.yaml", help="yaml 配置路径")
@click.option("--dry-run", is_flag=True, help="只打印将要写入的改动，不实际写文件")
def migrate_env(env_file, yaml_file, dry_run):
    """[DEPRECATED] 一次性迁移 .env 中的 API Key 到 yaml。

    项目已不再读取 `.env` 文件 —— 配置请直接写在 yaml，或用 `JIMI_*`
    进程环境变量覆盖。本命令仅为历史用户从 .env 迁移用，将在未来版本移除。
    """
    import re
    from pathlib import Path

    console.print(
        "[yellow]⚠ migrate-env 已废弃：项目不再读取 .env；"
        "本命令仅用于历史迁移，将在未来版本移除。[/yellow]"
    )

    env_path = Path(env_file)
    yaml_path = Path(yaml_file)

    if not env_path.exists():
        console.print(f"[yellow]未找到 {env_file}，跳过迁移。[/yellow]")
        return
    if not yaml_path.exists():
        console.print(f"[red]找不到 yaml 配置: {yaml_file}[/red]")
        return

    # 读取 .env
    envs = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        envs[k.strip()] = v.strip().strip('"').strip("'")

    # .env -> yaml 映射
    mapping = [
        ("API_KEY", r'(^\s*api_key:\s*)"[^"]*"', "model.api_key"),
        ("BASE_URL", r'(^\s*base_url:\s*)"[^"]*"', "model.base_url"),
        ("EMBEDDING_API_KEY", None, "embedding.api_key"),
        ("EMBEDDING_BASE_URL", None, "embedding.base_url"),
        ("EMBEDDING_MODEL", None, "embedding.model"),
    ]

    content = yaml_path.read_text(encoding="utf-8")
    changes: list[str] = []

    # 同名字段按出现顺序替换
    def _replace_first_two_blank(key_name: str, values: list[str]) -> None:
        """按出现顺序把 `key_name: ""` 替换为 values 列表中的值"""
        nonlocal content
        pattern = re.compile(rf'(^\s*{re.escape(key_name)}:\s*)"[^"]*"', re.MULTILINE)
        new_parts = []
        idx = 0
        pos = 0
        for m in pattern.finditer(content):
            if idx >= len(values):
                break
            val = values[idx]
            if not val:
                idx += 1
                continue
            new_parts.append(content[pos:m.start()])
            new_parts.append(f'{m.group(1)}"{val}"')
            pos = m.end()
            idx += 1
            changes.append(f"  {key_name}[#{idx}] → {_mask(val)}")
        new_parts.append(content[pos:])
        if new_parts:
            content = "".join(new_parts)

    api_key = envs.get("API_KEY", "")
    base_url = envs.get("BASE_URL", "")
    emb_key = envs.get("EMBEDDING_API_KEY", "")
    emb_base = envs.get("EMBEDDING_BASE_URL", "")
    emb_model = envs.get("EMBEDDING_MODEL", "")

    # api_key 顺序：model -> embedding
    _replace_first_two_blank("api_key", [api_key, emb_key])
    # base_url 同理
    _replace_first_two_blank("base_url", [base_url, emb_base])

    # embedding.model 单独处理
    if emb_model:
        content = re.sub(
            r'(^\s*model:\s*)"text-embedding-ada-002"',
            rf'\1"{emb_model}"',
            content,
            count=1,
            flags=re.MULTILINE,
        )
        changes.append(f"  embedding.model → {emb_model}")

    if not changes:
        console.print("[yellow]未找到需要迁移的键（或值为空）。[/yellow]")
        return

    console.print("[bold]将迁移以下键:[/bold]")
    for c in changes:
        console.print(c)

    if dry_run:
        console.print("\n[dim](dry-run，未写入文件)[/dim]")
        return

    # 备份旧文件
    backup = env_path.parent / f"{env_path.name}.bak"
    env_path.rename(backup)
    yaml_path.write_text(content, encoding="utf-8")

    console.print(f"\n[green]已写入 {yaml_path}[/green]")
    console.print(f"[dim]原 .env 已备份到 {backup}[/dim]")
    console.print("[yellow]检查 yaml 无误后可删除 .env.bak[/yellow]")


def _mask(v: str) -> str:
    """隐藏敏感值的中间部分"""
    if not v or len(v) < 8:
        return "***"
    return f"{v[:4]}...{v[-4:]}"


def _pkg_version(pkg_name: str) -> str:
    """读取已安装包的版本号，未安装返回空字符串"""
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return ""


@cli.group()
def config():
    """查看 / 修改 yaml 配置（持久化写回，保留注释）"""
    pass


@config.command("path")
def config_path():
    """打印当前使用的 yaml 路径（尊重 JIMI_CONFIG）"""
    from server.config.config_io import get_config_path

    console.print(str(get_config_path()))


@config.command("list")
def config_list():
    """列出所有可修改的配置项及当前值"""
    from server.config.config_io import list_keys

    table = Table(border_style="cyan", title="可修改配置项")
    table.add_column("key", style="bold")
    table.add_column("当前值")
    for key, val in list_keys():
        table.add_row(key, val)
    console.print(table)


@config.command("get")
@click.argument("key", required=False)
def config_get(key):
    """读取配置。省略 key 打印完整 yaml；传 key（如 `agent.model.model_id`）打印单项"""
    from server.config.config_io import get_value, render_doc

    try:
        if not key:
            console.print(render_doc())
        else:
            val = get_value(key)
            console.print(f"[bold]{key}[/bold] = {val!r}")
    except KeyError:
        console.print(f"[red]未找到 key: {key}[/red]")
        raise SystemExit(1)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """修改单个配置项并持久化。

    KEY   点分路径，例如 agent.model.model_id
    VALUE 字符串，按白名单自动转类型（bool/int/float/str）
    """
    from server.config.config_io import set_value

    try:
        new_val = set_value(key, value)
        console.print(f"[green]已写入[/green] {key} = {new_val!r}")
        console.print(
            "[dim]提示: 已运行的 Gateway 不会热重载，可重启或调用 "
            "`POST /api/skills/rebuild` + 重启 Agent 进程。[/dim]"
        )
    except PermissionError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)
    except (ValueError, KeyError) as e:
        console.print(f"[red]设置失败: {e}[/red]")
        raise SystemExit(1)


@config.command("edit")
def config_edit():
    """用 $EDITOR 打开 yaml（默认 vi / nano / vscode）"""
    import subprocess
    from server.config.config_io import get_config_path

    path = get_config_path()
    if not path.exists():
        console.print(f"[red]配置文件不存在: {path}[/red]")
        raise SystemExit(1)

    editor = os.getenv("EDITOR") or os.getenv("VISUAL") or "vi"
    try:
        subprocess.call([editor, str(path)])
    except FileNotFoundError:
        console.print(f"[red]未找到编辑器 {editor!r}，请设置 $EDITOR[/red]")
        raise SystemExit(1)


def _handle_daemon(install: bool, uninstall: bool) -> None:
    """处理 --install-daemon / --uninstall-daemon。不自动 load，只打印命令。"""
    import platform
    from pathlib import Path

    system = platform.system()
    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable
    cli_path = project_root / "cli.py"
    log_dir = project_root / "data" / "logs"

    if system == "Darwin":
        # macOS launchd
        plist_path = Path.home() / "Library" / "LaunchAgents" / "com.jimi.agent.plist"
        label = "com.jimi.agent"
        if uninstall:
            console.print(f"[bold]手动卸载 launchd 单元:[/bold]")
            console.print(f"  launchctl bootout gui/$(id -u) {plist_path}")
            console.print(f"  rm {plist_path}")
            return

        plist_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
            '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
            '<plist version="1.0">\n'
            '<dict>\n'
            f'    <key>Label</key>\n    <string>{label}</string>\n'
            '    <key>ProgramArguments</key>\n    <array>\n'
            f'        <string>{python_exe}</string>\n'
            f'        <string>{cli_path}</string>\n'
            '        <string>start</string>\n'
            '    </array>\n'
            f'    <key>WorkingDirectory</key>\n    <string>{project_root}</string>\n'
            '    <key>RunAtLoad</key>\n    <true/>\n'
            '    <key>KeepAlive</key>\n    <true/>\n'
            f'    <key>StandardOutPath</key>\n    <string>{log_dir / "daemon-stdout.log"}</string>\n'
            f'    <key>StandardErrorPath</key>\n    <string>{log_dir / "daemon-stderr.log"}</string>\n'
            '</dict>\n</plist>\n'
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(plist_xml, "utf-8")
        console.print(f"[green]已生成 launchd plist:[/green] {plist_path}")
        console.print()
        console.print("[bold]接下来手动执行（需要你确认）:[/bold]")
        console.print(f"  launchctl bootstrap gui/$(id -u) {plist_path}")
        console.print(f"  launchctl kickstart -k gui/$(id -u)/{label}")
        console.print()
        console.print(f"查看状态: [cyan]launchctl print gui/$(id -u)/{label}[/cyan]")
        console.print(f"日志:     [cyan]tail -F {log_dir}/daemon-*.log[/cyan]")
        console.print(f"卸载:     [cyan]python cli.py onboard --uninstall-daemon[/cyan]")

    elif system == "Linux":
        # systemd user
        svc_path = Path.home() / ".config" / "systemd" / "user" / "jimi-agent.service"
        if uninstall:
            console.print(f"[bold]手动卸载 systemd user 单元:[/bold]")
            console.print(f"  systemctl --user disable --now jimi-agent.service")
            console.print(f"  rm {svc_path}")
            console.print(f"  systemctl --user daemon-reload")
            return

        svc_text = (
            "[Unit]\n"
            "Description=JimiAgent Gateway\n"
            "After=network-online.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"WorkingDirectory={project_root}\n"
            f"ExecStart={python_exe} {cli_path} start\n"
            "Restart=on-failure\n"
            "RestartSec=5\n"
            f"StandardOutput=append:{log_dir}/daemon-stdout.log\n"
            f"StandardError=append:{log_dir}/daemon-stderr.log\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        svc_path.parent.mkdir(parents=True, exist_ok=True)
        svc_path.write_text(svc_text, "utf-8")
        console.print(f"[green]已生成 systemd user 单元:[/green] {svc_path}")
        console.print()
        console.print("[bold]接下来手动执行（需要你确认）:[/bold]")
        console.print("  systemctl --user daemon-reload")
        console.print("  systemctl --user enable --now jimi-agent.service")
        console.print()
        console.print("查看状态: [cyan]systemctl --user status jimi-agent.service[/cyan]")
        console.print(f"日志:     [cyan]tail -F {log_dir}/daemon-*.log[/cyan]")
        console.print(f"卸载:     [cyan]python cli.py onboard --uninstall-daemon[/cyan]")

    else:
        console.print(
            f"[yellow]暂不支持在 {system} 上自动安装 daemon。[/yellow]\n"
            "请手动配置操作系统级的服务管理器（Windows: NSSM / Task Scheduler）。"
        )


@cli.command()
@click.option(
    "--install-daemon", "install_daemon", is_flag=True, default=False,
    help="写入 launchd（macOS）/ systemd user（Linux）单元文件，使 Gateway 常驻",
)
@click.option(
    "--uninstall-daemon", "uninstall_daemon", is_flag=True, default=False,
    help="打印删除 daemon 的命令与路径（不自动删除）",
)
def onboard(install_daemon, uninstall_daemon):
    """交互式基础配置引导向导"""
    if install_daemon or uninstall_daemon:
        _handle_daemon(install_daemon, uninstall_daemon)
        return

    from pathlib import Path
    from server.config.config_io import get_config_path, set_value

    config_path = get_config_path()
    if not config_path.exists():
        # 从 example 复制
        example = Path("config/agent_config.example.yaml")
        if example.exists():
            import shutil
            config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(example, config_path)
            console.print(f"[green]已从 example 创建 {config_path}[/green]")
        else:
            console.print("[red]未找到 agent_config.example.yaml，请先手动创建配置文件[/red]")
            raise SystemExit(1)

    console.print(Panel("[bold]JimiAgent 配置向导[/bold]", border_style="cyan"))

    # API Key
    api_key = click.prompt("LLM API Key", default="", show_default=False)
    if api_key.strip():
        set_value("agent.model.api_key", api_key.strip())
        console.print("[green]  ✔ API Key 已写入[/green]")

    # Base URL
    base_url = click.prompt("LLM Base URL（留空使用 OpenAI 默认）", default="", show_default=False)
    if base_url.strip():
        set_value("agent.model.base_url", base_url.strip())
        console.print("[green]  ✔ Base URL 已写入[/green]")

    # Model ID
    model_id = click.prompt("模型 ID", default="gpt-4o-mini")
    set_value("agent.model.model_id", model_id.strip())
    console.print("[green]  ✔ 模型已写入[/green]")

    # Agent Name
    agent_name = click.prompt("Agent 名称（用于 @mention 激活）", default="jimi")
    set_value("agent.name", agent_name.strip())
    console.print("[green]  ✔ Agent 名称已写入[/green]")

    # Workspace
    workspace = click.prompt("Workspace 路径", default="./workspace")
    set_value("agent.workspace", workspace.strip())
    # 确保目录存在
    ws_path = Path(workspace.strip())
    if not ws_path.is_absolute():
        ws_path = Path.cwd() / ws_path
    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "skills").mkdir(exist_ok=True)
    (ws_path / "prompts").mkdir(exist_ok=True)
    console.print("[green]  ✔ Workspace 已创建[/green]")

    # Embedding（可选）
    if click.confirm("是否配置 Embedding（用于 Skills 语义召回）？", default=False):
        emb_key = click.prompt("Embedding API Key", default=api_key.strip() or "", show_default=False)
        if emb_key.strip():
            set_value("agent.embedding.api_key", emb_key.strip())
        emb_base = click.prompt("Embedding Base URL（留空同上）", default="", show_default=False)
        if emb_base.strip():
            set_value("agent.embedding.base_url", emb_base.strip())
        console.print("[green]  ✔ Embedding 配置已写入[/green]")

    console.print()
    console.print(Panel(
        "[bold green]配置完成！[/bold green]\n\n"
        "启动 Gateway:  [cyan]python cli.py start[/cyan]\n"
        "本地 TUI 聊天: [cyan]python cli.py chat[/cyan]\n"
        "健康检查:      [cyan]python cli.py doctor[/cyan]",
        border_style="green",
    ))


@cli.command()
def doctor():
    """健康检查：Python 版本、关键依赖、API Key、Workspace、Skills"""
    console.print("[bold]JimiAgent 健康检查[/bold]\n")
    checks = []

    # Python 版本
    py_ver = sys.version.split()[0]
    ok = tuple(int(x) for x in py_ver.split(".")[:2]) >= (3, 10)
    checks.append(("Python 版本", py_ver, ok))

    # 关键依赖
    for display_name, pkg_name in [
        ("LangChain", "langchain"),
        ("LangGraph", "langgraph"),
        ("LangGraph SQLite Checkpointer", "langgraph-checkpoint-sqlite"),
        ("LlamaIndex Core", "llama-index-core"),
        ("LlamaIndex Embeddings OpenAI", "llama-index-embeddings-openai"),
        ("LlamaIndex Retrievers BM25 (N 组 Hybrid)", "llama-index-retrievers-bm25"),
        ("pyautogui (M 组 Desktop)", "pyautogui"),
        ("mss (M 组 Screen Capture)", "mss"),
        ("playwright (M 组 Browser)", "playwright"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("aiosqlite", "aiosqlite"),
        ("croniter", "croniter"),
        ("ruamel.yaml", "ruamel.yaml"),
    ]:
        ver = _pkg_version(pkg_name)
        checks.append((display_name, ver or "未安装", bool(ver)))

    # API Key
    from server.config.settings import get_settings
    settings = get_settings()
    has_key = bool(settings.api_key)
    checks.append(("API Key", "已配置" if has_key else "未配置", has_key))

    # Workspace
    ws_exists = settings.workspace_abs_path.exists()
    checks.append(("Workspace", str(settings.workspace_abs_path), ws_exists))

    # Skills
    skills_dir = settings.workspace_abs_path / "skills"
    skill_count = 0
    if skills_dir.exists():
        skill_count = len([d for d in skills_dir.iterdir() if d.is_dir()])
    checks.append(("Skills", f"{skill_count} 个", skill_count > 0))

    # 日志
    log_cfg = settings.logging
    log_summary = f"{log_cfg.level} · {log_cfg.output}"
    log_ok = log_cfg.level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR")
    if log_cfg.output in ("file", "both"):
        log_summary += f" · {settings.log_file_abs_path}"
        try:
            settings.log_file_abs_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_ok = False
    checks.append(("日志", log_summary, log_ok))

    # DM 安全策略
    warnings: list[tuple[str, str]] = []
    channels_raw = settings.channels.raw if hasattr(settings.channels, "raw") else {}
    for ch_name, ch_cfg in (channels_raw or {}).items():
        if not isinstance(ch_cfg, dict) or not ch_cfg.get("enabled"):
            continue
        policy = ch_cfg.get("dmPolicy", "open")
        secret = ch_cfg.get("secret", "") or ""
        allow_from = ch_cfg.get("allowFrom") or []

        # open 且无保护
        if policy == "open" and not secret and (not allow_from or "*" in allow_from):
            warnings.append((
                f"DM:{ch_name}",
                "危险：dmPolicy=open 无 secret 且 allowFrom 含 *，任何人可投递",
            ))
        # open 且无 secret
        elif policy == "open" and not secret:
            warnings.append((
                f"DM:{ch_name}",
                "风险：dmPolicy=open 且 secret 为空，建议设置 Header 密钥",
            ))
        # closed 且无 allowFrom
        elif policy == "closed" and not allow_from:
            warnings.append((
                f"DM:{ch_name}",
                "警告：dmPolicy=closed 但未配置 allowFrom，将拒绝所有 sender",
            ))

    if warnings:
        for key, msg in warnings:
            checks.append((key, msg, False))
    else:
        # 只有配置了 channel 才显示 pass
        if channels_raw:
            checks.append(("DM 安全策略", "全部合规", True))

    # 打印结果
    table = Table(border_style="cyan")
    table.add_column("检查项", style="bold")
    table.add_column("状态")
    table.add_column("结果")

    all_ok = True
    for name, value, ok in checks:
        icon = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        if not ok:
            all_ok = False
        table.add_row(name, value, icon)

    console.print(table)

    if all_ok:
        console.print("\n[bold green]所有检查通过[/bold green]")
    else:
        console.print("\n[bold yellow]部分检查未通过，请查看上方详情[/bold yellow]")


# plugins

@cli.group()
def plugins():
    """OpenClaw 插件管理"""
    pass


def _build_registry_for_cli():
    from server.config.settings import get_settings
    from server.core.plugin_registry import PluginRegistry
    reg = PluginRegistry(get_settings())
    reg.scan()
    return reg


@plugins.command("list")
@click.option("--json", "json_out", is_flag=True, help="输出 JSON")
@click.option("--enabled", is_flag=True, help="仅显示 enabled 插件")
def plugins_list(json_out: bool, enabled: bool):
    """列出所有已发现插件"""
    import json as _json
    reg = _build_registry_for_cli()
    items = reg.enabled_plugins() if enabled else reg.list()

    if json_out:
        console.print_json(_json.dumps([s.to_dict() for s in items], ensure_ascii=False))
        return

    if not items:
        console.print(
            "[yellow]未发现插件。[/yellow] 放到 "
            "`<workspace>/.openclaw/plugins/<id>/` 或 `~/.openclaw/plugins/<id>/` 下再试。"
        )
        return

    table = Table(title="OpenClaw 插件")
    table.add_column("id", style="cyan")
    table.add_column("状态")
    table.add_column("类型")
    table.add_column("版本")
    table.add_column("skills", justify="right")
    table.add_column("来源", overflow="fold")
    for s in items:
        table.add_row(
            s.id, s.status, s.manifest.manifest_kind,
            s.manifest.version or "-",
            str(len(s.manifest.skills)),
            str(s.manifest.source),
        )
    console.print(table)


@plugins.command("inspect")
@click.argument("plugin_id")
@click.option("--json", "json_out", is_flag=True, help="输出 JSON")
def plugins_inspect(plugin_id: str, json_out: bool):
    """深度查看单个插件"""
    import json as _json
    reg = _build_registry_for_cli()
    state = reg.get(plugin_id)
    if state is None:
        console.print(f"[red]未找到插件: {plugin_id}[/red]")
        sys.exit(2)
    if json_out:
        console.print_json(_json.dumps(state.to_dict(), ensure_ascii=False))
        return
    m = state.manifest
    console.print(Panel.fit(
        f"[bold cyan]{state.id}[/bold cyan]  v{m.version or '-'}\n"
        f"[dim]{m.description or ''}[/dim]\n\n"
        f"manifest : {m.manifest_file} ({m.manifest_kind})\n"
        f"source   : {m.source}\n"
        f"kind     : {m.kind or '-'}\n"
        f"status   : {state.status}"
        + (f"  ({state.reason_disabled})" if state.reason_disabled else ""),
        title="插件详情",
    ))
    if m.skills:
        console.print("\n[bold]skills[/bold]: " + ", ".join(m.skills))
    if m.providers:
        console.print("[bold]providers[/bold]: " + ", ".join(m.providers))
    if m.channels:
        console.print("[bold]channels[/bold]: " + ", ".join(m.channels))
    if m.command_aliases:
        console.print(
            "[bold]commandAliases[/bold]: "
            + ", ".join(str(c.get("name")) for c in m.command_aliases)
        )
    if state.warnings:
        console.print("\n[bold yellow]诊断告警:[/bold yellow]")
        for w in state.warnings:
            console.print(f"  [yellow]⚠[/yellow] {w}")
    if m.error:
        console.print(f"\n[bold red]错误:[/bold red] {m.error}")


@plugins.command("install")
@click.argument("spec")
@click.option("-l", "--link", is_flag=True, help="以软链方式安装（仅本地路径）")
@click.option("--force", is_flag=True, help="覆盖已存在的同 id 插件")
def plugins_install(spec: str, link: bool, force: bool):
    """安装插件：本地路径 / Git URL / npm 包"""
    from server.config.settings import get_settings
    from server.core.plugin_installer import PluginInstaller
    inst = PluginInstaller(get_settings())
    res = inst.install(spec, link=link, force=force)
    if res.ok:
        console.print(f"[green]✅[/green] {res.message}\n目标: {res.target_dir}")
    else:
        console.print(f"[red]❌[/red] {res.message}")
        sys.exit(1)


@plugins.command("uninstall")
@click.argument("plugin_id")
@click.option("--keep-files", is_flag=True, help="保留目录，仅从 registry 卸载")
def plugins_uninstall(plugin_id: str, keep_files: bool):
    """卸载插件"""
    from server.config.settings import get_settings
    from server.core.plugin_installer import PluginInstaller
    inst = PluginInstaller(get_settings())
    res = inst.uninstall(plugin_id, keep_files=keep_files)
    if res.ok:
        console.print(f"[green]✅[/green] {res.message}")
    else:
        console.print(f"[red]❌[/red] {res.message}")
        sys.exit(1)


@plugins.command("doctor")
def plugins_doctor():
    """对所有插件做环境/依赖诊断"""
    reg = _build_registry_for_cli()
    items = reg.list()
    if not items:
        console.print("[dim]未发现插件[/dim]")
        return
    any_issue = False
    for s in items:
        if s.manifest.error:
            any_issue = True
            console.print(f"[red]❌ {s.id}[/red]: {s.manifest.error}")
            continue
        if s.warnings:
            any_issue = True
            console.print(f"[yellow]⚠ {s.id}[/yellow]")
            for w in s.warnings:
                console.print(f"  - {w}")
    if not any_issue:
        console.print("[green]全部插件无告警 ✅[/green]")


@plugins.command("enable")
@click.argument("plugin_id")
def plugins_enable(plugin_id: str):
    """在当前进程启用插件（不持久化；写 yaml 请用 /config）"""
    reg = _build_registry_for_cli()
    if reg.enable(plugin_id):
        console.print(f"[green]✅[/green] 已启用 {plugin_id}")
    else:
        console.print(f"[red]❌[/red] 未找到插件 {plugin_id}")
        sys.exit(1)


@plugins.command("disable")
@click.argument("plugin_id")
def plugins_disable(plugin_id: str):
    """在当前进程禁用插件"""
    reg = _build_registry_for_cli()
    if reg.disable(plugin_id):
        console.print(f"[green]✅[/green] 已禁用 {plugin_id}")
    else:
        console.print(f"[red]❌[/red] 未找到插件 {plugin_id}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
