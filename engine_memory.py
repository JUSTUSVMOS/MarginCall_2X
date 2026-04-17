import copy
import json
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.tools import tool

logger = logging.getLogger(__name__)

# 定義 Brain 資料夾路徑
BRAIN_DIR = Path(__file__).resolve().parent / ".brain"
BRAIN_DIR.mkdir(exist_ok=True)
BRAIN_FILE = BRAIN_DIR / "commit.json"
FRONTAL_LOBE_FILE = BRAIN_DIR / "frontal-lobe.md"
EMOTION_FILE = BRAIN_DIR / "emotion-log.json"
MARKET_REGIME_FILE = BRAIN_DIR / "market-regime.md"
HEARTBEAT_FILE = BRAIN_DIR / "heartbeat.json"
SNAPSHOT_FILE = BRAIN_DIR / "snapshot.json"

FRONTAL_LOBE_FIELDS = (
    "Market View",
    "Core Levels",
    "Portfolio Health",
    "Next Round",
)

FRONTAL_LOBE_SECTION_ALIASES = {
    "Market View": ["Market View", "市場視角", "市場觀點"],
    "Core Levels": ["Core Levels", "Key Levels", "核心點位", "關鍵點位"],
    "Portfolio Health": ["Portfolio Health", "持倉評估", "Position Health"],
    "Next Round": ["Next Round", "下一回合", "Plan", "Action Plan"],
    "Context Note": ["Context Note", "補充說明", "Additional Context"],
}

FRONTAL_LOBE_KEYWORDS = {
    "Market View": ["bullish", "bearish", "neutral", "看漲", "看跌", "盤整", "多頭", "空頭", "reversal", "range", "trend"],
    "Core Levels": ["support", "resistance", "watch", "break", "hold", "ma", "均線", "支撐", "壓力", "關鍵", "spx", "qqq", "nvda", "tsla"],
    "Portfolio Health": ["portfolio", "position", "health", "risk", "exposure", "drawdown", "underwater", "倉位", "風險", "槓桿", "持倉"],
    "Next Round": ["next round", "if", "plan", "will", "trim", "cut", "add", "hedge", "如果", "若", "打算", "會"],
}

FRONTAL_LOBE_WRITE_GUIDE = """When calling update_frontal_lobe, always write a professional trading note with these labeled fields:
- Market View: Bullish / Bearish / Neutral + one-sentence thesis
- Core Levels: the key support / resistance / MA levels being watched
- Portfolio Health: whether current positions are healthy or over-risked
- Next Round: if A happens, I will do B

Low-quality placeholder notes will be rejected. Avoid vague one-liners like "觀望", "waiting for CPI", or unlabeled thoughts with no levels / plan.

Example:
Market View: Bearish - Market shows signs of reversal after SPX rejected 5250 resistance.
Core Levels: Watch SPX 5200 support and 5250 resistance.
Portfolio Health: Current longs are slightly underwater but still above 20MA; risk is manageable.
Next Round: If SPX breaks below 5180, I will cut exposure and wait for confirmation before re-adding.
"""

def generate_commit_hash(content: dict) -> str:
    """產生類似 Git 的 Commit Hash (SHA256 取前 8 碼)"""
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:8]

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _shorten(text: Optional[str], limit: int = 120) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit - 3].rstrip() + "..."

def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

def _merge_defaults(defaults: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    if not isinstance(incoming, dict):
        return merged
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged

def _default_market_regime() -> Dict[str, Any]:
    return {
        "summary": "",
        "state": "未初始化",
        "riskScore": None,
        "updatedAt": None,
        "source": "",
        "watchpoints": [],
        "reasons": [],
        "signals": {}
    }

def _default_heartbeat() -> Dict[str, Any]:
    return {
        "lastMacroSyncAt": None,
        "lastMacroChangeAt": None,
        "lastSyncStatus": "never_synced",
        "lastSyncMessage": ""
    }

def _default_state() -> Dict[str, Any]:
    return {
        "frontalLobe": "",
        "emotion": "neutral",
        "marketRegime": _default_market_regime(),
        "heartbeat": _default_heartbeat()
    }

def _extract_labeled_value(content: str, aliases: List[str]) -> str:
    for raw_line in content.splitlines():
        line = raw_line.strip().replace("：", ":")
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        if label.strip().lower() in {alias.lower() for alias in aliases}:
            return value.strip()
    return ""

def _split_sentences(content: str) -> List[str]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    parts = re.split(r"\n+|(?<=[。！？!?\.])\s+|;\s+", normalized)
    return [_shorten(part.strip(" -\t"), 180) for part in parts if part and part.strip(" -\t")]

def _pick_sentences(sentences: List[str], keywords: List[str], limit: int = 2) -> str:
    lowered_keywords = [keyword.lower() for keyword in keywords]
    hits: List[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in lowered_keywords) and sentence not in hits:
            hits.append(sentence)
    return " | ".join(hits[:limit])

def _infer_market_view(content: str, sentences: List[str]) -> str:
    lowered = content.lower()
    if any(token in lowered for token in ["bearish", "看跌", "空頭", "risk off", "downtrend"]):
        stance = "Bearish"
    elif any(token in lowered for token in ["bullish", "看漲", "多頭", "uptrend", "risk on"]):
        stance = "Bullish"
    else:
        stance = "Neutral"

    thesis = _pick_sentences(sentences, FRONTAL_LOBE_KEYWORDS["Market View"], limit=1)
    if not thesis and sentences:
        thesis = sentences[0]
    if not thesis:
        return f"{stance} - Thesis not explicitly stated."
    if stance.lower() in thesis.lower():
        return thesis
    return f"{stance} - {thesis}"

def _infer_core_levels(content: str, sentences: List[str]) -> str:
    explicit = _pick_sentences(sentences, FRONTAL_LOBE_KEYWORDS["Core Levels"], limit=2)
    if explicit:
        return explicit
    numeric = [sentence for sentence in sentences if re.search(r"\b\d{3,5}(?:\.\d+)?\b", sentence)]
    if numeric:
        return " | ".join(numeric[:2])
    return "No critical support / resistance level was explicitly stated."

def _infer_portfolio_health(sentences: List[str]) -> str:
    explicit = _pick_sentences(sentences, FRONTAL_LOBE_KEYWORDS["Portfolio Health"], limit=2)
    if explicit:
        return explicit
    return "Portfolio health / sizing was not explicitly stated; re-check exposure before adding risk."

def _infer_next_round(sentences: List[str]) -> str:
    explicit = _pick_sentences(sentences, FRONTAL_LOBE_KEYWORDS["Next Round"], limit=2)
    if explicit:
        return explicit
    return "If the thesis weakens, reduce risk first and wait for confirmation before re-entering."

def normalize_frontal_lobe_note(content: str) -> str:
    cleaned = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        raise ValueError("Frontal lobe content cannot be empty.")

    sections = {
        field: _extract_labeled_value(cleaned, FRONTAL_LOBE_SECTION_ALIASES[field])
        for field in FRONTAL_LOBE_FIELDS
    }
    context_note = _extract_labeled_value(cleaned, FRONTAL_LOBE_SECTION_ALIASES["Context Note"])
    sentences = _split_sentences(cleaned)
    structured_input = any(sections.values())

    if not sections["Market View"]:
        sections["Market View"] = _infer_market_view(cleaned, sentences)
    if not sections["Core Levels"]:
        sections["Core Levels"] = _infer_core_levels(cleaned, sentences)
    if not sections["Portfolio Health"]:
        sections["Portfolio Health"] = _infer_portfolio_health(sentences)
    if not sections["Next Round"]:
        sections["Next Round"] = _infer_next_round(sentences)

    lines = [f"{field}: {sections[field]}" for field in FRONTAL_LOBE_FIELDS]
    if not structured_input:
        compact_original = _shorten(cleaned, 220)
        if compact_original and compact_original not in " ".join(lines):
            context_note = compact_original
    if context_note:
        lines.append(f"Context Note: {context_note}")
    return "\n".join(lines)

def parse_frontal_lobe_note(content: str) -> Dict[str, str]:
    sections = {field: "" for field in FRONTAL_LOBE_FIELDS}
    sections["Context Note"] = ""
    normalized = content.replace("：", ":")
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip()
        value = value.strip()
        if label in sections:
            sections[label] = value
    return sections

def _coerce_frontal_lobe_sections(content: str) -> Dict[str, str]:
    if not content or not content.strip():
        return parse_frontal_lobe_note("")
    return parse_frontal_lobe_note(normalize_frontal_lobe_note(content))

def _format_signal_value(key: str, value: Any) -> str:
    if isinstance(value, float):
        if key in {"vixZ", "tnxZ", "dxyZ", "sentimentScore", "yieldCurve10Y2Y"}:
            return f"{value:+.2f}"
        if key == "gexBillions":
            return f"{value:+.2f}"
        return f"{value:.2f}"
    return str(value)

def format_key_signals(signals: Optional[Dict[str, Any]], max_items: int = 4) -> str:
    if not signals:
        return ""
    signal_map = [
        ("gexBillions", "GEX", "B"),
        ("vixZ", "VIX_Z", ""),
        ("dixPr", "DIX_PR", ""),
        ("spx", "SPX", ""),
        ("yieldCurve10Y2Y", "YC_10Y2Y", ""),
        ("sentimentScore", "SENT", ""),
    ]
    parts = []
    for key, label, suffix in signal_map:
        value = signals.get(key)
        if value in (None, "", []):
            continue
        parts.append(f"{label}={_format_signal_value(key, value)}{suffix}")
        if len(parts) >= max_items:
            break
    return ", ".join(parts)

class Brain:
    """
    對標 OpenAlice 的 Git-like Cognitive State Management (大腦與額葉模組)。
    負責追蹤工作記憶 (Frontal Lobe)、情緒狀態 (Emotion)、宏觀市場體感 (Market Regime) 的變化，
    並將每一次的異動建立 Commit 儲存為連貫的認知變化鏈。
    """
    _PLACEHOLDER_PATTERNS = (
        "暫無明確",
        "觀望",
        "no clear",
        "not explicitly stated",
        "thesis not explicitly",
        "waiting for",
        "尚未建立",
        "re-check exposure before adding risk",
        "wait for confirmation before re-entering",
    )

    def __init__(self):
        self.state: Dict[str, Any] = _default_state()
        self.commits: List[Dict[str, Any]] = []
        self.head: Optional[str] = None
        self._load()

    def _load(self):
        """從本地端讀取持久化記憶"""
        if BRAIN_FILE.exists():
            try:
                with open(BRAIN_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.state = _merge_defaults(_default_state(), data.get('state', {}))
                    self.commits = []
                    for commit in data.get('commits', []):
                        normalized = self._normalize_loaded_commit(commit)
                        if normalized:
                            self.commits.append(normalized)
                    self.head = data.get('head') or (self.commits[-1]["hash"] if self.commits else None)
            except Exception as e:
                logger.error(f"Failed to load brain state: {e}")
        self._persist_views()

    def _normalize_loaded_commit(self, commit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(commit, dict):
            return None

        normalized = {
            "hash": commit.get("hash"),
            "parent_hash": commit.get("parent_hash") or commit.get("parentHash"),
            "timestamp": commit.get("timestamp") or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "type": commit.get("type", "unknown"),
            "summary": commit.get("summary") or "",
            "key_signals": commit.get("key_signals") or "",
            "frontal_lobe_ref": commit.get("frontal_lobe_ref") or "",
        }

        if commit.get("source"):
            normalized["source"] = commit.get("source")
        if commit.get("delta") is not None:
            normalized["delta"] = commit.get("delta")

        if not normalized["summary"]:
            legacy_state = _merge_defaults(_default_state(), commit.get("stateAfter", {}))
            legacy_market = legacy_state.get("marketRegime", _default_market_regime())
            legacy_frontal_lobe = legacy_state.get("frontalLobe", "")

            if normalized["type"] == "frontal_lobe":
                legacy_note_source = legacy_frontal_lobe or commit.get("message", "")
                note = normalize_frontal_lobe_note(legacy_note_source) if legacy_note_source else ""
                sections = parse_frontal_lobe_note(note)
                normalized["summary"] = self._build_frontal_lobe_commit_summary(sections)
                normalized["key_signals"] = self._compose_market_signal_summary(legacy_market)
                normalized["frontal_lobe_ref"] = self._build_frontal_lobe_ref(note)
                normalized["delta"] = {
                    "market_view": sections.get("Market View", ""),
                    "core_levels": sections.get("Core Levels", ""),
                    "portfolio_health": sections.get("Portfolio Health", ""),
                    "next_round": sections.get("Next Round", ""),
                }
            elif normalized["type"] == "emotion":
                current = legacy_state.get("emotion") or "unknown"
                reason = commit.get("message", "")
                normalized["summary"] = f"🧠 EMOTION: {current} | {_shorten(reason, 80)}"
                normalized["key_signals"] = self._compose_market_signal_summary(legacy_market)
                normalized["frontal_lobe_ref"] = self._build_frontal_lobe_ref(legacy_frontal_lobe)
                normalized["delta"] = {"to": current, "reason": reason}
            elif normalized["type"] == "market_regime":
                normalized["summary"] = self._build_market_regime_commit_summary(_default_market_regime(), legacy_market)
                normalized["key_signals"] = format_key_signals(legacy_market.get("signals", {}))
                normalized["frontal_lobe_ref"] = self._build_frontal_lobe_ref(legacy_frontal_lobe)
                normalized["delta"] = {
                    "risk_score_to": legacy_market.get("riskScore"),
                    "state_to": legacy_market.get("state"),
                    "summary": legacy_market.get("summary", ""),
                }
            else:
                normalized["summary"] = commit.get("message") or "Legacy commit"
                normalized["frontal_lobe_ref"] = self._build_frontal_lobe_ref(legacy_frontal_lobe)

        if not normalized["hash"]:
            normalized["hash"] = generate_commit_hash({
                "type": normalized["type"],
                "summary": normalized["summary"],
                "delta": normalized.get("delta", {}),
                "parent_hash": normalized["parent_hash"],
                "timestamp": normalized["timestamp"],
            })
        return normalized

    def _contains_placeholder_phrase(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(pattern.lower() in lowered for pattern in self._PLACEHOLDER_PATTERNS)

    def _is_placeholder_content(self, note: str) -> bool:
        if not note or len(note.strip()) < 30:
            return True

        sections = parse_frontal_lobe_note(note)
        meaningful_sections = sum(
            1
            for field in FRONTAL_LOBE_FIELDS
            if sections.get(field, "").strip() and not self._contains_placeholder_phrase(sections[field])
        )
        if meaningful_sections < 2:
            return True

        placeholder_sections = sum(
            1
            for field in FRONTAL_LOBE_FIELDS
            if self._contains_placeholder_phrase(sections.get(field, ""))
        )
        return placeholder_sections >= 2

    def _save(self):
        """將狀態持久化至本地端"""
        try:
            BRAIN_DIR.mkdir(exist_ok=True)
            with open(BRAIN_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "state": self.state,
                    "commits": self.commits,
                    "head": self.head
                }, f, ensure_ascii=False, indent=2)
            self._persist_views()
        except Exception as e:
            logger.error(f"Failed to save brain state: {e}")

    def _read_persisted_head(self) -> Optional[str]:
        if not BRAIN_FILE.exists():
            return None
        try:
            with open(BRAIN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read persisted brain head: {e}")
            return self.head

        if not isinstance(data, dict):
            return self.head
        if data.get("head"):
            return data.get("head")

        commits = data.get("commits", [])
        if commits and isinstance(commits[-1], dict):
            return commits[-1].get("hash")
        return None

    def _persist_views(self):
        BRAIN_DIR.mkdir(exist_ok=True)
        market = self.state.get("marketRegime", _default_market_regime())
        heartbeat = self.state.get("heartbeat", _default_heartbeat())
        recent_emotions = self.get_emotion()

        FRONTAL_LOBE_FILE.write_text(
            "# Frontal Lobe\n\n" + (self.state.get("frontalLobe") or "(empty)"),
            encoding='utf-8'
        )
        EMOTION_FILE.write_text(
            json.dumps(recent_emotions, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        HEARTBEAT_FILE.write_text(
            json.dumps(heartbeat, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        SNAPSHOT_FILE.write_text(
            json.dumps(self.get_brain_snapshot(), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        MARKET_REGIME_FILE.write_text(self._render_market_regime_markdown(market, heartbeat), encoding='utf-8')

    def _public_commit_view(self, commit: Dict[str, Any], include_chain: bool = False) -> Dict[str, Any]:
        public_view = {
            "timestamp": commit.get("timestamp"),
            "type": commit.get("type"),
            "summary": commit.get("summary") or commit.get("message") or "",
        }
        key_signals = commit.get("key_signals", "")
        if key_signals:
            public_view["key_signals"] = key_signals
        plan_ref = commit.get("frontal_lobe_ref", "")
        if plan_ref:
            public_view["plan_ref"] = plan_ref
        if commit.get("source"):
            public_view["source"] = commit.get("source")
        if include_chain:
            public_view["hash"] = commit.get("hash")
            public_view["parent_hash"] = commit.get("parent_hash") or commit.get("parentHash")
        return public_view

    def _compose_market_signal_summary(self, market: Optional[Dict[str, Any]]) -> str:
        if not market:
            return ""
        parts = []
        if market.get("riskScore") is not None:
            parts.append(f"RISK={market['riskScore']}")
        if market.get("state"):
            parts.append(f"REGIME={market['state']}")
        signal_text = format_key_signals(market.get("signals", {}), max_items=3)
        if signal_text:
            parts.append(signal_text)
        return " | ".join(parts[:3])

    def _build_frontal_lobe_ref(self, content: Optional[str] = None) -> str:
        note = content or self.state.get("frontalLobe", "")
        if not note:
            return ""
        sections = parse_frontal_lobe_note(note)
        ref_parts = []
        if sections.get("Market View"):
            ref_parts.append(_shorten(sections["Market View"], 80))
        if sections.get("Next Round"):
            ref_parts.append(f"Next: {_shorten(sections['Next Round'], 80)}")
        elif sections.get("Core Levels"):
            ref_parts.append(_shorten(sections["Core Levels"], 80))
        ref = " | ".join(ref_parts)
        return ref or _shorten(note, 120)

    def _build_frontal_lobe_commit_summary(self, sections: Dict[str, str]) -> str:
        market_view = sections.get("Market View") or "Frontal lobe refreshed"
        core_levels = sections.get("Core Levels") or ""
        summary = f"🧠 NOTE: {_shorten(market_view, 80)}"
        if core_levels:
            summary += f" | {_shorten(core_levels, 60)}"
        return summary

    def _pick_market_highlight(self, market: Dict[str, Any]) -> str:
        for item in market.get("reasons", []) + market.get("watchpoints", []):
            if item:
                return _shorten(item, 60)
        return _shorten(market.get("summary", ""), 60)

    def _build_market_regime_commit_summary(self, previous_market: Dict[str, Any], new_market: Dict[str, Any]) -> str:
        old_score = previous_market.get("riskScore")
        new_score = new_market.get("riskScore")
        if old_score is None and new_score is not None:
            movement = "RISK INIT"
        elif old_score is None or new_score is None or new_score == old_score:
            movement = "RISK HOLD"
        elif new_score > old_score:
            movement = "RISK UP"
        else:
            movement = "RISK DOWN"
        old_display = old_score if old_score is not None else "N/A"
        new_display = new_score if new_score is not None else "N/A"
        highlight = self._pick_market_highlight(new_market)
        summary = f"{new_market.get('state', '未初始化')} {movement}: {old_display} -> {new_display}"
        if highlight:
            summary += f" | {highlight}"
        return summary

    def _create_commit(
        self,
        commit_type: str,
        summary: str,
        delta: Optional[Dict[str, Any]] = None,
        key_signals: str = "",
        frontal_lobe_ref: str = "",
        source: str = "",
        expected_head: Optional[str] = None
    ) -> bool:
        """建立新的認知變化節點 (Commit)"""
        if expected_head is not None and expected_head != self.head:
            logger.warning(
                f"[Brain] Optimistic lock conflict! expected={expected_head}, actual={self.head}. "
                "Another in-memory write landed first - aborting this commit."
            )
            return False

        if expected_head is not None:
            persisted_head = self._read_persisted_head()
            if persisted_head != expected_head:
                logger.warning(
                    f"[Brain] Optimistic lock conflict! expected={expected_head}, actual={persisted_head}. "
                    "Another persisted write landed first - aborting this commit."
                )
                return False

        payload = {
            "type": commit_type,
            "summary": summary,
            "delta": delta or {},
            "key_signals": key_signals,
            "frontal_lobe_ref": frontal_lobe_ref,
            "parent_hash": self.head,
            "timestamp": _utc_now_iso()
        }

        commit_hash = generate_commit_hash(payload)

        commit = {
            "hash": commit_hash,
            "parent_hash": self.head,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "type": commit_type,
            "summary": summary,
            "key_signals": key_signals,
            "frontal_lobe_ref": frontal_lobe_ref
        }
        if delta:
            commit["delta"] = delta
        if source:
            commit["source"] = source

        self.commits.append(commit)
        self.head = commit_hash
        self._save()
        return True

    def _render_market_regime_markdown(self, market: Dict[str, Any], heartbeat: Dict[str, Any]) -> str:
        signal_labels = {
            "yieldCurve10Y2Y": "Yield Curve 10Y-2Y",
            "fedFundsRate": "Fed Funds",
            "dixPr": "DIX_PR",
            "gexBillions": "GEX (B)",
            "sentimentScore": "Sentiment Score",
            "sentimentLabel": "Sentiment Label",
            "spx": "SPX",
            "spx10Ma": "SPX 10MA",
            "spx20Ma": "SPX 20MA",
            "spx200Ma": "SPX 200MA",
            "dxyZ": "DXY Z",
            "tnxZ": "TNX Z",
            "vixZ": "VIX Z",
            "skewPr": "SKEW PR"
        }
        watchpoints = market.get("watchpoints") or ["無"]
        reasons = market.get("reasons") or ["無"]
        signals = market.get("signals") or {}
        signal_lines = [
            f"- {signal_labels.get(key, key)}: {value}"
            for key, value in signals.items()
        ] or ["- 無"]

        lines = [
            "# Persistent Macro Regime",
            "",
            f"- Updated: {market.get('updatedAt') or '尚未同步'}",
            f"- Regime: {market.get('state') or '未初始化'}",
            f"- Risk Score: {market.get('riskScore') if market.get('riskScore') is not None else 'N/A'}",
            f"- Source: {market.get('source') or 'N/A'}",
            f"- Heartbeat: {heartbeat.get('lastSyncStatus', 'unknown')} @ {heartbeat.get('lastMacroSyncAt') or 'N/A'}",
            "",
            "## Summary",
            market.get("summary") or "尚未建立",
            "",
            "## Watchpoints",
            *[f"- {item}" for item in watchpoints],
            "",
            "## Reasons",
            *[f"- {item}" for item in reasons],
            "",
            "## Signals",
            *signal_lines
        ]
        return "\n".join(lines)

    def _market_regime_changed(self, new_market: Dict[str, Any]) -> bool:
        current = self.state.get("marketRegime", _default_market_regime())
        keys = ["summary", "state", "riskScore", "watchpoints", "reasons", "signals"]
        return any(current.get(key) != new_market.get(key) for key in keys)

    # ==================== Queries (讀取記憶) ====================

    def get_frontal_lobe(self) -> str:
        return self.state["frontalLobe"]

    def get_emotion(self) -> Dict[str, Any]:
        emotion_commits = [c for c in self.commits if c["type"] == "emotion"]
        recent = emotion_commits[-10:]
        recent.reverse()
        return {
            "current": self.state["emotion"],
            "recentChanges": [self._public_commit_view(commit) for commit in recent]
        }

    def get_market_regime(self, max_age_minutes: int = 180) -> Dict[str, Any]:
        regime = copy.deepcopy(self.state["marketRegime"])
        regime["isStale"] = self.needs_market_regime_refresh(max_age_minutes)
        return regime

    def get_brain_snapshot(self, max_age_minutes: int = 180) -> Dict[str, Any]:
        snapshot_state = copy.deepcopy(self.state)
        snapshot_state["marketRegime"]["isStale"] = self.needs_market_regime_refresh(max_age_minutes)
        return {
            "state": snapshot_state,
            "head": self.head,
            "recentCommits": self.log(10)
        }

    def needs_market_regime_refresh(self, max_age_minutes: int = 180) -> bool:
        updated_at = self.state["marketRegime"].get("updatedAt")
        parsed = _parse_iso_timestamp(updated_at)
        if parsed is None:
            return True
        age_seconds = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
        return age_seconds > max_age_minutes * 60

    def get_cognitive_context(self, max_age_minutes: int = 180) -> str:
        market = self.get_market_regime(max_age_minutes=max_age_minutes)
        watchpoints = market.get("watchpoints") or ["無"]
        reasons = market.get("reasons") or ["無"]
        signals = market.get("signals") or {}
        frontal_lobe = self.state["frontalLobe"] or "空白 (首次運行)"

        signal_summary = ", ".join([
            f"{key}={value}" for key, value in signals.items() if value not in (None, "", [])
        ]) or "無"
        watchpoints_text = "\n".join([f"  - {item}" for item in watchpoints])
        reasons_text = "\n".join([f"  - {item}" for item in reasons[:5]])
        frontal_lobe_text = "\n".join([f"  {line}" for line in frontal_lobe.splitlines()])
        stale_note = "（可能過期，等待 heartbeat 刷新）" if market.get("isStale") else ""

        return (
            "\n\n## Current Brain State\n"
            f"- Emotion: {self.state['emotion']}\n"
            f"- Frontal Lobe:\n{frontal_lobe_text}\n"
            "\n## Persistent Macro / Market Regime\n"
            f"- Updated: {market.get('updatedAt') or '尚未同步'} {stale_note}\n"
            f"- Regime: {market.get('state') or '未初始化'}\n"
            f"- Risk Score: {market.get('riskScore') if market.get('riskScore') is not None else 'N/A'}\n"
            f"- Summary: {market.get('summary') or '尚未建立'}\n"
            f"- Watchpoints:\n{watchpoints_text}\n"
            f"- Reasons:\n{reasons_text}\n"
            f"- Key Signals: {signal_summary}\n"
        )

    def log(self, limit: int = 10) -> List[Dict[str, Any]]:
        recent = self.commits[-limit:]
        recent.reverse()
        return [self._public_commit_view(commit) for commit in recent]

    # ==================== Mutations (寫入記憶) ====================

    def update_lobe_section(self, section_name: str, new_content: str, source: str = "system") -> Dict[str, Any]:
        """精準更新額葉的特定區塊，保留其他部分"""
        if section_name not in FRONTAL_LOBE_FIELDS:
            return {"success": False, "message": f"Invalid section: {section_name}"}
        
        current_note = self.state.get("frontalLobe") or ""
        sections = _coerce_frontal_lobe_sections(current_note)
        
        # 更新指定區塊
        sections[section_name] = new_content.strip()
        
        # 重新組裝 Markdown 格式的內容
        lines = [f"{field}: {sections[field]}" for field in FRONTAL_LOBE_FIELDS]
        if sections.get("Context Note"):
            lines.append(f"Context Note: {sections['Context Note']}")
        
        normalized_note = "\n".join(lines)
        self.state["frontalLobe"] = normalized_note
        
        # 建立 commit
        summary = f"🧠 {section_name.upper()} AUTO-UPDATE: {_shorten(new_content, 80)}"
        delta_key = section_name.lower().replace(" ", "_")
        self._create_commit(
            "frontal_lobe_patch",
            summary,
            delta={delta_key: new_content},
            key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
            frontal_lobe_ref=self._build_frontal_lobe_ref(normalized_note),
            source=source
        )
        return {"success": True, "message": f"Frontal lobe section '{section_name}' updated."}

    def update_frontal_lobe(self, content: str) -> Dict[str, Any]:
        normalized_note = normalize_frontal_lobe_note(content)
        if self._is_placeholder_content(normalized_note):
            logger.warning("[Brain] Rejected placeholder-quality frontal lobe write.")
            return {"success": False, "message": "Rejected: content is too vague to persist."}

        snapshot_head = self.head
        self.state["frontalLobe"] = normalized_note
        sections = parse_frontal_lobe_note(normalized_note)
        summary = self._build_frontal_lobe_commit_summary(sections)
        committed = self._create_commit(
            "frontal_lobe",
            summary,
            delta={
                "market_view": sections.get("Market View", ""),
                "core_levels": sections.get("Core Levels", ""),
                "portfolio_health": sections.get("Portfolio Health", ""),
                "next_round": sections.get("Next Round", ""),
            },
            key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
            frontal_lobe_ref=self._build_frontal_lobe_ref(normalized_note),
            source="frontal_lobe_write",
            expected_head=snapshot_head
        )
        if not committed:
            self._load()
            return {"success": False, "message": "Rejected: concurrent frontal lobe update detected."}
        return {"success": True, "message": "Frontal lobe updated successfully"}

    def update_emotion(self, emotion: str, reason: str) -> Dict[str, Any]:
        old_emotion = self.state["emotion"]
        self.state["emotion"] = emotion
        self._create_commit(
            "emotion",
            f"🧠 EMOTION: {old_emotion} -> {emotion} | {_shorten(reason, 80)}",
            delta={"from": old_emotion, "to": emotion, "reason": reason},
            key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
            frontal_lobe_ref=self._build_frontal_lobe_ref(),
            source="emotion_update"
        )
        return {"success": True, "message": f"Emotion: {old_emotion} -> {emotion}"}

    def update_market_regime(
        self,
        summary: str,
        regime: Optional[str] = None,
        risk_score: Optional[int] = None,
        watchpoints: Optional[List[str]] = None,
        reasons: Optional[List[str]] = None,
        signals: Optional[Dict[str, Any]] = None,
        source: str = "manual",
        updated_at: Optional[str] = None
    ) -> Dict[str, Any]:
        previous_market = copy.deepcopy(self.state["marketRegime"])
        cleaned_market = {
            "summary": summary.strip(),
            "state": regime or self.state["marketRegime"].get("state") or "未初始化",
            "riskScore": risk_score if risk_score is not None else self.state["marketRegime"].get("riskScore"),
            "updatedAt": updated_at or _utc_now_iso(),
            "source": source,
            "watchpoints": [item.strip() for item in (watchpoints or []) if item and item.strip()],
            "reasons": [item.strip() for item in (reasons or []) if item and item.strip()],
            "signals": dict(signals or {})
        }
        if not cleaned_market["watchpoints"]:
            cleaned_market["watchpoints"] = self.state["marketRegime"].get("watchpoints", [])
        if not cleaned_market["reasons"]:
            cleaned_market["reasons"] = self.state["marketRegime"].get("reasons", [])
        if not cleaned_market["signals"]:
            cleaned_market["signals"] = self.state["marketRegime"].get("signals", {})

        changed = self._market_regime_changed(cleaned_market)
        self.state["marketRegime"] = cleaned_market
        heartbeat = self.state["heartbeat"]
        heartbeat["lastMacroSyncAt"] = _utc_now_iso()
        heartbeat["lastSyncMessage"] = cleaned_market["summary"]

        if changed:
            heartbeat["lastMacroChangeAt"] = heartbeat["lastMacroSyncAt"]
            heartbeat["lastSyncStatus"] = "updated"
            self._create_commit(
                "market_regime",
                self._build_market_regime_commit_summary(previous_market, cleaned_market),
                delta={
                    "risk_score_from": previous_market.get("riskScore"),
                    "risk_score_to": cleaned_market.get("riskScore"),
                    "state_from": previous_market.get("state"),
                    "state_to": cleaned_market.get("state"),
                    "summary": cleaned_market.get("summary", ""),
                },
                key_signals=format_key_signals(cleaned_market.get("signals", {})),
                frontal_lobe_ref=self._build_frontal_lobe_ref(),
                source=source
            )
        else:
            heartbeat["lastSyncStatus"] = "no_change"
            self._save()

        return {
            "success": True,
            "changed": changed,
            "message": "Market regime updated successfully" if changed else "Macro heartbeat completed with no material change"
        }

    def sync_market_snapshot(self, snapshot: Dict[str, Any], source: str = "macro_heartbeat") -> Dict[str, Any]:
        signals = snapshot.get("signals", {})
        watchpoints = []
        if signals.get("yieldCurve10Y2Y") is not None:
            watchpoints.append(f"殖利率曲線 10Y-2Y: {signals['yieldCurve10Y2Y']}")
        if signals.get("gexBillions") is not None:
            watchpoints.append(f"GEX: {signals['gexBillions']}B")
        if signals.get("sentimentLabel"):
            watchpoints.append(f"市場情緒: {signals['sentimentLabel']} ({signals.get('sentimentScore', 'N/A')})")
        if signals.get("spx") is not None:
            watchpoints.append(
                f"SPX: {signals['spx']} / MA20 {signals.get('spx20Ma', 'N/A')} / MA200 {signals.get('spx200Ma', 'N/A')}"
            )
        for reason in snapshot.get("reasons", [])[:2]:
            if reason not in watchpoints:
                watchpoints.append(reason)

        return self.update_market_regime(
            summary=snapshot.get("summary", ""),
            regime=snapshot.get("state"),
            risk_score=snapshot.get("riskScore"),
            watchpoints=watchpoints[:5],
            reasons=snapshot.get("reasons", []),
            signals=signals,
            source=source,
            updated_at=snapshot.get("generatedAt")
        )

    def record_heartbeat_error(self, message: str):
        heartbeat = self.state["heartbeat"]
        heartbeat["lastMacroSyncAt"] = _utc_now_iso()
        heartbeat["lastSyncStatus"] = "error"
        heartbeat["lastSyncMessage"] = message
        self._save()

# ==============================================================================
# 工具包裝層 (對標 OpenAlice src/tool/brain.ts) - 專為 LLM 設計的工具函數
# ==============================================================================

def get_frontal_lobe_write_guide() -> str:
    return FRONTAL_LOBE_WRITE_GUIDE

# 全域唯一的 Brain 實例
_global_brain = Brain()

@tool()
def get_frontal_lobe() -> str:
    """
    Retrieves the "Working Memory Space" left from the previous session.
    
    This is your Frontal Lobe, where you stored:
    - Market trend outlook (bullish/bearish/uncertain)
    - Portfolio health assessments
    - Key predictions or expectations for upcoming rounds
    - Critical reminders (e.g., "Watch BTC support at $95k")
    
    CALL THIS TOOL FIRST at the start of every analysis to maintain cognitive continuity.
    Returns: Your previous self-assessment string.
    """
    return _global_brain.get_frontal_lobe()

@tool(mode="write")
def update_frontal_lobe(content: str) -> str:
    """
    Updates your "Frontal Lobe" memory space with a disciplined professional trading note.

    CALL THIS TOOL SILENTLY before ending a session if there are significant updates to:
    - Market View: Bullish / Bearish / Neutral + one-sentence thesis
    - Core Levels: key support / resistance / MA levels being watched
    - Portfolio Health: whether current positions are healthy or over-risked
    - Next Round: if A happens, you will do B
    - Low-quality placeholder notes will be rejected instead of persisted

    This memory survives restarts. Keep it concise, structured, and decision-oriented.
    Example:
    Market View: Bearish - SPX rejected 5250 resistance and momentum is fading.
    Core Levels: Watch SPX 5200 support and 5250 resistance.
    Portfolio Health: Long exposure is slightly underwater but still controlled above 20MA.
    Next Round: If SPX breaks below 5180, I will cut exposure and wait for confirmation.
    """
    res = _global_brain.update_frontal_lobe(content)
    return res["message"]

@tool()
def get_emotion() -> str:
    """
    Retrieves your current emotional state and recent sentiment trajectory.
    Use this to understand your own cognitive bias.
    """
    return json.dumps(_global_brain.get_emotion(), ensure_ascii=False, indent=2)

@tool(mode="write")
def update_emotion(emotion: str, reason: str) -> str:
    """
    Updates your emotional state when market conditions or confidence levels shift.
    You MUST provide a reason — this creates a permanent commit in your brain log.
    
    Common states: fearful, cautious, neutral, confident, euphoric.
    Example: emotion="cautious", reason="BTC rejected at $100k resistance with declining volume."
    """
    res = _global_brain.update_emotion(emotion, reason)
    return res["message"]

@tool()
def get_market_regime() -> str:
    """
    Retrieves the persistent macro / market regime snapshot carried across restarts.

    This is the long-lived "what kind of tape are we in now?" memory surface:
    - current risk regime
    - risk score
    - heartbeat timestamp
    - watchpoints and key macro signals
    """
    return json.dumps(_global_brain.get_market_regime(), ensure_ascii=False, indent=2)

@tool(mode="write")
def update_market_regime(summary: str, regime: str = "", risk_score: int = -1) -> str:
    """
    Updates the persistent market regime summary when the macro backdrop materially changes.
    Keep it concise but durable: this should survive the next restart.
    """
    normalized_score = None if risk_score < 0 else risk_score
    res = _global_brain.update_market_regime(
        summary=summary,
        regime=regime or None,
        risk_score=normalized_score,
        source="llm_manual"
    )
    return res["message"]

@tool()
def refresh_market_regime(force: bool = False, max_age_minutes: int = 180) -> str:
    """
    Pulls the latest global risk snapshot and syncs it into the persistent brain.
    Use this when you need an updated macro heartbeat rather than relying on older memory.
    """
    result = sync_market_brain(force=force, max_age_minutes=max_age_minutes)
    return result["message"]

@tool()
def get_brain_snapshot() -> str:
    """
    Returns the full persistent cognitive snapshot, including frontal lobe, emotion,
    market regime, heartbeat metadata, and recent commits.
    """
    return json.dumps(_global_brain.get_brain_snapshot(), ensure_ascii=False, indent=2)

@tool()
def get_brain_log(limit: int = 10) -> str:
    """
    Views your Brain Commit History as compact semantic deltas optimized for AI recall.
    Each entry focuses on:
    - summary: one-line explanation of what changed
    - key_signals: the few market numbers that mattered
    - plan_ref: the matching frontal-lobe / next-round reference

    Hash-chain metadata is intentionally hidden from normal output to save tokens.
    """
    return json.dumps(_global_brain.log(limit), ensure_ascii=False, indent=2)

def sync_market_brain(force: bool = False, max_age_minutes: int = 180) -> Dict[str, Any]:
    if not force and not _global_brain.needs_market_regime_refresh(max_age_minutes=max_age_minutes):
        return {
            "success": True,
            "changed": False,
            "message": "Persistent macro regime is still fresh; heartbeat skipped.",
            "marketRegime": _global_brain.get_market_regime(max_age_minutes=max_age_minutes)
        }
    try:
        import engine_risk as risk

        snapshot = risk.get_global_risk_snapshot(force_refresh=force)
        result = _global_brain.sync_market_snapshot(snapshot)
        result["marketRegime"] = _global_brain.get_market_regime(max_age_minutes=max_age_minutes)
        return result
    except Exception as e:
        _global_brain.record_heartbeat_error(str(e))
        raise

def build_cognitive_context(max_age_minutes: int = 180) -> str:
    return _global_brain.get_cognitive_context(max_age_minutes=max_age_minutes)

def patch_frontal_lobe_section(section_name: str, content: str, source: str = "system") -> Dict[str, Any]:
    return _global_brain.update_lobe_section(section_name, content, source=source)

if __name__ == "__main__":
    # 自檢測試
    print("1. 更新額葉記憶...")
    print(update_frontal_lobe(
        "Market View: Neutral - 非農後市場進入事件前整理，等待 CPI 提供方向。\n"
        "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"
        "Portfolio Health: 槓桿偏低，部位可控，但不宜在數據前追價。\n"
        "Next Round: If CPI 低於預期且 SPX 站回 5250，我會小幅加碼；若跌破 5200，先降風險。"
    ))
    print("\n2. 更新情緒狀態...")
    print(update_emotion("cautious", "非農數據強勁，擔心聯準會延後降息，市場波動加劇。"))
    print("\n3. 更新市場狀態...")
    print(update_market_regime("目前處於高利率壓力下的整理盤，先觀察 SPX 與 VIX 是否同步轉穩。", "🟡 整理", 42))
    print("\n4. 讀取最新額葉記憶...")
    print(get_frontal_lobe())
    print("\n5. 查看完整腦快照...")
    print(get_brain_snapshot())
    print("\n6. 查看記憶 Commit 歷史 (Git-like Log)...")
    print(get_brain_log(3))
