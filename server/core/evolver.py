'''
Author: JimZhang
Date: 2026-04-19 18:30:00
LastEditors: JimZhang
LastEditTime: 2026-04-19 18:30:00
FilePath: /JimiAgent/server/core/evolver.py
Description: Skill 进化引擎。
'''
import datetime
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Strategy 预设

STRATEGIES = ("balanced", "innovate", "harden", "repair-only")


# 数据结构

@dataclass
class Signal:
    """从日志里提炼出的可进化信号。"""
    kind: str  # skill_failure / missing_skill / error_pattern / stagnation / …
    source: str  # 来源（skill 名 / tool 名 / log 行号）
    detail: str
    occurrences: int = 1

    def fingerprint(self) -> str:
        """24h 内同指纹不再重复触发"""
        today = datetime.date.today().isoformat()
        raw = f"{today}|{self.kind}|{self.source}|{self.detail[:120]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Gene:
    """基因：pattern → 建议修复方向"""
    id: str
    kind: str  # 匹配哪类 signal
    pattern: str  # 正则 / 关键字
    intent: str  # repair / innovate / harden
    prompt_template: str  # 结构化进化提示模板，支持 {source}/{detail}/{skill}
    tags: list = field(default_factory=list)


@dataclass
class Capsule:
    """胶囊：若干 gene 的复用包。"""
    id: str
    name: str
    description: str
    gene_ids: list = field(default_factory=list)


@dataclass
class EvolutionEvent:
    """一次 evolver run 的审计记录。"""
    ts: str  # ISO8601
    strategy: str
    signal: dict
    gene_id: str
    prompt: str
    applied: bool = False
    note: str = ""


@dataclass
class PersonalityState:
    """可进化的 agent 人格状态（简化版）"""
    mutation_count: int = 0
    last_evolution_ts: str = ""
    focus_areas: list = field(default_factory=list)


# 引擎主体

class EvolutionEngine:
    """Evolver 引擎。"""

    def __init__(
        self,
        gep_dir: Path,
        tool_log_path: Path,
        memory_store=None,
    ):
        self.gep_dir = Path(gep_dir)
        self.gep_dir.mkdir(parents=True, exist_ok=True)

        self.genes_path = self.gep_dir / "genes.json"
        self.capsules_path = self.gep_dir / "capsules.json"
        self.events_path = self.gep_dir / "events.jsonl"
        self.personality_path = self.gep_dir / "personality.json"
        self.tool_log_path = Path(tool_log_path)
        # 可选注入 MemoryStore
        self.memory_store = memory_store

        self._ensure_seed_assets()

    # 资产初始化

    def _ensure_seed_assets(self) -> None:
        """首次运行时写入默认资产。"""
        if not self.genes_path.exists():
            self._write_json(self.genes_path, DEFAULT_GENES)
        else:
            try:
                existing = json.loads(self.genes_path.read_text("utf-8"))
                existing_ids = {g.get("id") for g in existing if isinstance(g, dict)}
                added = 0
                for seed in DEFAULT_GENES:
                    if seed["id"] not in existing_ids:
                        existing.append(seed)
                        added += 1
                if added:
                    self._write_json(self.genes_path, existing)
                    logger.info(f"Evolver: 合并入 {added} 个新默认 gene")
            except Exception as e:
                logger.debug(f"合并默认 gene 失败（忽略）: {e}")

        if not self.capsules_path.exists():
            self._write_json(self.capsules_path, DEFAULT_CAPSULES)
        if not self.personality_path.exists():
            self._write_json(
                self.personality_path,
                asdict(PersonalityState()),
            )

    @staticmethod
    def _write_json(p: Path, obj) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 读写

    def load_genes(self) -> list[Gene]:
        try:
            raw = json.loads(self.genes_path.read_text("utf-8"))
        except Exception:
            return []
        out = []
        for g in raw:
            try:
                out.append(Gene(**g))
            except TypeError:
                logger.warning(f"gene 结构不完整，跳过: {g}")
        return out

    def load_capsules(self) -> list[Capsule]:
        try:
            raw = json.loads(self.capsules_path.read_text("utf-8"))
        except Exception:
            return []
        return [Capsule(**c) for c in raw if "id" in c]

    def load_personality(self) -> PersonalityState:
        try:
            raw = json.loads(self.personality_path.read_text("utf-8"))
            return PersonalityState(**raw)
        except Exception:
            return PersonalityState()

    def save_personality(self, ps: PersonalityState) -> None:
        self._write_json(self.personality_path, asdict(ps))

    def load_recent_events(self, limit: int = 20) -> list[EvolutionEvent]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text("utf-8").strip().splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(EvolutionEvent(**json.loads(line)))
            except Exception:
                continue
        return out

    def append_event(self, ev: EvolutionEvent) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

    # Signal 扫描

    def scan_signals(self, log_limit: int = 500) -> list[Signal]:
        """扫描并汇总可进化信号。"""
        signals: list[Signal] = []

        # 源 1：tool_log.jsonl
        tail: list[str] = []
        if self.tool_log_path.exists():
            try:
                tail = (
                    self.tool_log_path.read_text("utf-8")
                    .strip().splitlines()[-log_limit:]
                )
            except Exception:
                tail = []

        # 失败计数
        fail_count: dict[str, int] = {}
        error_samples: dict[str, str] = {}
        for line in tail:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "error":
                name = rec.get("tool") or rec.get("skill") or "unknown"
                fail_count[name] = fail_count.get(name, 0) + 1
                # 保留首条错误详情
                error_samples.setdefault(name, str(rec.get("error", "")))

        for name, cnt in fail_count.items():
            if cnt >= 3:  # 连续失败 3 次即发 signal
                signals.append(Signal(
                    kind="skill_failure",
                    source=name,
                    detail=error_samples.get(name, ""),
                    occurrences=cnt,
                ))

        # stagnation：24h 内重复出现
        recent_events = self.load_recent_events(limit=50)
        sig_counter: dict[str, int] = {}
        now = datetime.datetime.utcnow()
        for ev in recent_events:
            try:
                ev_ts = datetime.datetime.fromisoformat(ev.ts.replace("Z", ""))
            except Exception:
                continue
            if (now - ev_ts).total_seconds() > 86400:
                continue
            key = (ev.signal or {}).get("source", "") + "|" + (ev.signal or {}).get("kind", "")
            sig_counter[key] = sig_counter.get(key, 0) + 1
        for key, cnt in sig_counter.items():
            if cnt >= 3:
                src, kind = key.split("|", 1)
                signals.append(Signal(
                    kind="stagnation",
                    source=src,
                    detail=f"同一 {kind} 在 24h 内触发 {cnt} 次，请跳出重复修复回路",
                    occurrences=cnt,
                ))

        # procedural_usage：高命中规则
        if self.memory_store is not None:
            try:
                procs = self.memory_store.list_recent(
                    limit=50, kinds=["procedural"],
                )
                for m in procs:
                    if getattr(m, "hits", 0) >= 5:
                        signals.append(Signal(
                            kind="procedural_usage",
                            source=f"memory:{m.id}",
                            detail=m.text,
                            occurrences=int(m.hits),
                        ))
            except Exception as e:
                logger.debug(f"扫 procedural 记忆失败（忽略）: {e}")

        return signals

    # Gene 匹配

    def select_gene(
        self, signal: Signal, strategy: str, genes: list[Gene],
    ) -> Optional[Gene]:
        """按 strategy 偏好从 genes 里选最佳匹配。"""
        candidates = [g for g in genes if g.kind == signal.kind]
        if not candidates:
            # fallback: 匹配 kind="any"
            candidates = [g for g in genes if g.kind == "any"]
        if not candidates:
            return None

        def score(g: Gene) -> int:
            s = 0
            # pattern 命中加分
            try:
                if re.search(g.pattern, signal.detail, re.IGNORECASE):
                    s += 5
            except re.error:
                pass
            # strategy 偏好加分
            if strategy == "repair-only" and g.intent == "repair":
                s += 3
            elif strategy == "harden" and g.intent == "harden":
                s += 3
            elif strategy == "innovate" and g.intent == "innovate":
                s += 3
            elif strategy == "balanced":
                s += 1  # balanced 模式均匀加分
            return s

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    # 生成 prompt

    def emit_prompt(self, gene: Gene, signal: Signal) -> str:
        """填充 gene 的 prompt_template，产出结构化进化提示"""
        tmpl = gene.prompt_template
        return tmpl.format(
            source=signal.source,
            detail=signal.detail[:500],
            kind=signal.kind,
            occurrences=signal.occurrences,
            skill=signal.source,  # 常见别名
        )

    # 主循环

    def run_cycle(
        self,
        strategy: str = "balanced",
        apply: bool = False,
    ) -> list[EvolutionEvent]:
        """执行一轮进化。返回本轮生成的所有 event。"""
        if strategy not in STRATEGIES:
            raise ValueError(
                f"未知 strategy: {strategy}（可选 {STRATEGIES}）"
            )

        signals = self.scan_signals()
        if not signals:
            logger.info("Evolver: 当前没有可进化信号，返回空")
            return []

        # 24h 内同指纹不重复生成
        seen = self._recent_fingerprints()
        signals = [s for s in signals if s.fingerprint() not in seen]
        if not signals:
            logger.info("Evolver: 所有 signal 均已处理过（去重）")
            return []

        genes = self.load_genes()
        if not genes:
            logger.warning("Evolver: gene 库为空，无法生成进化提示")
            return []

        events: list[EvolutionEvent] = []
        now_iso = datetime.datetime.utcnow().isoformat() + "Z"
        for sig in signals[:5]:  # 单轮上限 5 条，避免淹没
            gene = self.select_gene(sig, strategy, genes)
            if gene is None:
                continue
            prompt = self.emit_prompt(gene, sig)
            ev = EvolutionEvent(
                ts=now_iso,
                strategy=strategy,
                signal=asdict(sig),
                gene_id=gene.id,
                prompt=prompt,
                applied=apply,
            )
            self.append_event(ev)
            events.append(ev)

        # 更新 personality
        if events:
            ps = self.load_personality()
            ps.mutation_count += len(events)
            ps.last_evolution_ts = now_iso
            self.save_personality(ps)

        return events

    async def apply_event(self, ev: EvolutionEvent, agent) -> dict:
        """把一个 EvolutionEvent 应用到 agent：派生新 session 并以 prompt 为首轮消息。

        应用后追加一条标记事件（applied=True 的副本），便于审计。
        """
        title = f"Evolution: {ev.gene_id}"
        sess = agent.session_mgr.create_session(title)
        # 标注 session 元信息，便于事后筛选
        sess.metadata["evolver"] = "applied"
        sess.metadata["gene_id"] = ev.gene_id

        # 这里直接调 agent.chat
        resp, effective_sid = await agent.chat(ev.prompt, sess.id)

        applied_ev = EvolutionEvent(
            ts=datetime.datetime.utcnow().isoformat() + "Z",
            strategy=ev.strategy,
            signal=ev.signal,
            gene_id=ev.gene_id,
            prompt=ev.prompt,
            applied=True,
            note=f"applied in session {effective_sid}",
        )
        self.append_event(applied_ev)

        return {
            "session_id": effective_sid,
            "reply": resp,
            "gene_id": ev.gene_id,
        }

    def _recent_fingerprints(self) -> set[str]:
        """最近 24h 内已处理过的 signal fingerprint 集合"""
        out: set[str] = set()
        for ev in self.load_recent_events(limit=100):
            try:
                ev_ts = datetime.datetime.fromisoformat(ev.ts.replace("Z", ""))
            except Exception:
                continue
            if (datetime.datetime.utcnow() - ev_ts).total_seconds() > 86400:
                continue
            sig = Signal(**ev.signal) if ev.signal else None
            if sig:
                out.add(sig.fingerprint())
        return out


# 种子资产

DEFAULT_GENES = [
    {
        "id": "gene.skill_failure_repair",
        "kind": "skill_failure",
        "pattern": ".*",
        "intent": "repair",
        "prompt_template": (
            "# 进化提示: 修复 skill `{skill}`\n\n"
            "Skill `{skill}` 在最近连续失败 {occurrences} 次。错误样本:\n\n"
            "```\n{detail}\n```\n\n"
            "**修复方向**:\n"
            "1. 检查 `workspace/skills/{skill}/` 下的 Python 实现，定位抛错位置\n"
            "2. 验证该 skill 的 SKILL.md description 与实际参数签名是否一致\n"
            "3. 如果是外部依赖问题（网络/权限/配置），在 description 中标注前置条件\n"
            "4. 修好后手动或用 `/restart` 让 Agent 重建索引\n\n"
            "**约束**:\n"
            "- 不要改 `server/` 下的核心代码\n"
            "- 仅可改 `workspace/skills/{skill}/`\n"
        ),
        "tags": ["repair", "skill"],
    },
    {
        "id": "gene.missing_skill_innovate",
        "kind": "missing_skill",
        "pattern": ".*",
        "intent": "innovate",
        "prompt_template": (
            "# 进化提示: 提议新 skill `{source}`\n\n"
            "用户意图 `{detail}` 最近被反复提及（{occurrences} 次），"
            "但当前 skills 召回不到合适工具。\n\n"
            "**建议**:\n"
            "1. 在 `workspace/skills/` 下新建一个名为 `{source}` 的 skill\n"
            "2. 提供 `SKILL.md` 描述 + 一个简单 Python 实现\n"
            "3. 用 list_dir / read_file 查看已有 skill 结构作为模板\n\n"
            "**约束**: 仅创建新文件，不修改 server/ 下代码。\n"
        ),
        "tags": ["innovate", "skill"],
    },
    {
        "id": "gene.stagnation_breaker",
        "kind": "stagnation",
        "pattern": ".*",
        "intent": "harden",
        "prompt_template": (
            "# 进化提示: 跳出修复回路\n\n"
            "针对 `{source}` 的同类 signal `{kind}` 在 24h 内已触发 {occurrences} 次，"
            "当前修复思路可能陷入回路。\n\n"
            "**对策**:\n"
            "1. 在 `EvolutionEvent` 中回看最近对该 source 的历史 prompt，"
            "   判断是否始终绕着同一假设打转\n"
            "2. 改变切入角度：从「修 skill」切换到「改 prompt」，"
            "   或从「加代码」切换到「加日志定位」\n"
            "3. 如果连续 5 次都未解决，记一笔「需人工介入」事件并等待 operator\n\n"
            "**约束**: 优先加观测/日志而非再改代码。\n"
        ),
        "tags": ["harden", "stagnation"],
    },
    {
        "id": "gene.error_pattern_generic",
        "kind": "error_pattern",
        "pattern": ".*(timeout|connection|refused|unauthorized).*",
        "intent": "harden",
        "prompt_template": (
            "# 进化提示: 网络/权限类错误加固\n\n"
            "在 `{source}` 观察到错误模式 `{detail}`。\n\n"
            "**建议加固**:\n"
            "1. 在 skill 入口加超时保护（默认 10s）+ 重试（2 次，指数退避）\n"
            "2. 若是权限问题，在 SKILL.md 里显式声明所需 env/api key\n"
            "3. 用 try/except 捕获并返回对用户友好的错误说明而非原始 traceback\n"
        ),
        "tags": ["harden", "network"],
    },
    {
        "id": "gene.procedural_promotion",
        "kind": "procedural_usage",
        "pattern": ".*",
        "intent": "harden",
        "prompt_template": (
            "# 进化提示: 固化高频 procedural 规则\n\n"
            "Memory 记忆 `{source}` 已被命中 {occurrences} 次，内容：\n\n"
            "> {detail}\n\n"
            "**建议**：将此规则显式写入 `workspace/AGENTS.md` 或 `workspace/USER.md`，"
            "让它变成 system prompt 永久段而非依赖 memory recall 每次注入。\n\n"
            "**执行**:\n"
            "1. 用 `read_file workspace/AGENTS.md` 看现有内容\n"
            "2. 用 `edit_file` 在「规则 / 约定」小节追加一行\n"
            "3. 固化完成后可以 `/forget {source}` 从 memory 里移除\n\n"
            "**约束**: 仅改 `workspace/`，不碰 `server/` 代码。\n"
        ),
        "tags": ["harden", "procedural", "memory"],
    },
]

DEFAULT_CAPSULES = [
    {
        "id": "capsule.basic_self_repair",
        "name": "Basic Self-Repair Kit",
        "description": "最小可用的 skill 自修复工具集：skill_failure + stagnation + error_pattern",
        "gene_ids": [
            "gene.skill_failure_repair",
            "gene.stagnation_breaker",
            "gene.error_pattern_generic",
        ],
    },
]
