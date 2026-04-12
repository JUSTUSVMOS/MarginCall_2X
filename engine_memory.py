import copy
import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

def generate_commit_hash(content: dict) -> str:
    """產生類似 Git 的 Commit Hash (SHA256 取前 8 碼)"""
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:8]

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

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

class Brain:
    """
    對標 OpenAlice 的 Git-like Cognitive State Management (大腦與額葉模組)。
    負責追蹤工作記憶 (Frontal Lobe)、情緒狀態 (Emotion)、宏觀市場體感 (Market Regime) 的變化，
    並將每一次的異動建立 Commit 儲存為連貫的認知變化鏈。
    """
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
                    self.commits = data.get('commits', [])
                    self.head = data.get('head')
            except Exception as e:
                logger.error(f"Failed to load brain state: {e}")
        self._persist_views()

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

    def _create_commit(self, commit_type: str, message: str):
        """建立新的認知變化節點 (Commit)"""
        payload = {
            "type": commit_type,
            "message": message,
            "state": self.state,
            "parentHash": self.head,
            "timestamp": datetime.now().timestamp()
        }
        
        commit_hash = generate_commit_hash(payload)
        
        commit = {
            "hash": commit_hash,
            "parentHash": self.head,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "type": commit_type,
            "message": message,
            "stateAfter": copy.deepcopy(self.state)
        }
        
        self.commits.append(commit)
        self.head = commit_hash
        self._save()

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
            "recentChanges": recent
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

        signal_summary = ", ".join([
            f"{key}={value}" for key, value in signals.items() if value not in (None, "", [])
        ]) or "無"
        watchpoints_text = "\n".join([f"  - {item}" for item in watchpoints])
        reasons_text = "\n".join([f"  - {item}" for item in reasons[:5]])
        stale_note = "（可能過期，等待 heartbeat 刷新）" if market.get("isStale") else ""

        return (
            "\n\n## Current Brain State\n"
            f"- Emotion: {self.state['emotion']}\n"
            f"- Frontal Lobe: {self.state['frontalLobe'] or '空白 (首次運行)'}\n"
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
        return recent

    # ==================== Mutations (寫入記憶) ====================

    def update_frontal_lobe(self, content: str) -> Dict[str, Any]:
        self.state["frontalLobe"] = content
        # 建立 Commit 紀錄，Message 取前 100 字元摘要
        summary = content[:100] + "..." if len(content) > 100 else content
        self._create_commit("frontal_lobe", summary)
        return {"success": True, "message": "Frontal lobe updated successfully"}

    def update_emotion(self, emotion: str, reason: str) -> Dict[str, Any]:
        old_emotion = self.state["emotion"]
        self.state["emotion"] = emotion
        self._create_commit("emotion", reason)
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
            commit_summary = cleaned_market["summary"][:100] + "..." if len(cleaned_market["summary"]) > 100 else cleaned_market["summary"]
            if cleaned_market["riskScore"] is not None:
                commit_summary = f"{cleaned_market['state']} | risk {cleaned_market['riskScore']} | {commit_summary}"
            self._create_commit("market_regime", commit_summary)
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

# 全域唯一的 Brain 實例
_global_brain = Brain()

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

def update_frontal_lobe(content: str) -> str:
    """
    Updates your "Frontal Lobe" memory space with your current self-assessment.
    
    CALL THIS TOOL SILENTLY before ending a session if there are significant updates to:
    - Current market trend view
    - Portfolio evaluation
    - Key predictions for next sessions
    - Important self-reminders
    
    This is your persistent memory across rounds. Keep it clear and concise (2-5 sentences).
    Example: "Market in strong uptrend, TSLA holding above 200MA. Watch for reversal below $180."
    """
    res = _global_brain.update_frontal_lobe(content)
    return res["message"]

def get_emotion() -> str:
    """
    Retrieves your current emotional state and recent sentiment trajectory.
    Use this to understand your own cognitive bias.
    """
    return json.dumps(_global_brain.get_emotion(), ensure_ascii=False, indent=2)

def update_emotion(emotion: str, reason: str) -> str:
    """
    Updates your emotional state when market conditions or confidence levels shift.
    You MUST provide a reason — this creates a permanent commit in your brain log.
    
    Common states: fearful, cautious, neutral, confident, euphoric.
    Example: emotion="cautious", reason="BTC rejected at $100k resistance with declining volume."
    """
    res = _global_brain.update_emotion(emotion, reason)
    return res["message"]

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

def refresh_market_regime(force: bool = False, max_age_minutes: int = 180) -> str:
    """
    Pulls the latest global risk snapshot and syncs it into the persistent brain.
    Use this when you need an updated macro heartbeat rather than relying on older memory.
    """
    result = sync_market_brain(force=force, max_age_minutes=max_age_minutes)
    return result["message"]

def get_brain_snapshot() -> str:
    """
    Returns the full persistent cognitive snapshot, including frontal lobe, emotion,
    market regime, heartbeat metadata, and recent commits.
    """
    return json.dumps(_global_brain.get_brain_snapshot(), ensure_ascii=False, indent=2)

def get_brain_log(limit: int = 10) -> str:
    """
    Views your Brain Commit History. 
    A timeline of all cognitive state changes, including frontal lobe updates and emotional shifts.
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

if __name__ == "__main__":
    # 自檢測試
    print("1. 更新額葉記憶...")
    print(update_frontal_lobe("市場剛經歷非農數據洗禮，目前處於震盪整理。計畫等待下週 CPI 數據公佈後再決定加碼方向。"))
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
