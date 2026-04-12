import json
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# 定義 Brain 資料夾路徑
BRAIN_DIR = Path(__file__).resolve().parent / ".brain"
BRAIN_DIR.mkdir(exist_ok=True)
BRAIN_FILE = BRAIN_DIR / "commit.json"

def generate_commit_hash(content: dict) -> str:
    """產生類似 Git 的 Commit Hash (SHA256 取前 8 碼)"""
    content_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:8]

class Brain:
    """
    對標 OpenAlice 的 Git-like Cognitive State Management (大腦與額葉模組)。
    負責追蹤工作記憶 (Frontal Lobe) 與情緒狀態 (Emotion) 的變化，
    並將每一次的異動建立 Commit 儲存為連貫的認知變化鏈。
    """
    def __init__(self):
        self.state: Dict[str, str] = {
            "frontalLobe": "",
            "emotion": "neutral"
        }
        self.commits: List[Dict[str, Any]] = []
        self.head: Optional[str] = None
        self._load()

    def _load(self):
        """從本地端讀取持久化記憶"""
        if BRAIN_FILE.exists():
            try:
                with open(BRAIN_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.state = data.get('state', self.state)
                    self.commits = data.get('commits', [])
                    self.head = data.get('head')
            except Exception as e:
                logger.error(f"Failed to load brain state: {e}")

    def _save(self):
        """將狀態持久化至本地端"""
        try:
            with open(BRAIN_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "state": self.state,
                    "commits": self.commits,
                    "head": self.head
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save brain state: {e}")

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
            "stateAfter": dict(self.state)
        }
        
        self.commits.append(commit)
        self.head = commit_hash
        self._save()

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

# ==============================================================================
# 工具包裝層 (對標 OpenAlice src/tool/brain.ts) - 專為 LLM 設計的工具函數
# ==============================================================================

# 全域唯一的 Brain 實例
_global_brain = Brain()

def get_frontal_lobe() -> str:
    """
    讀取你在上一回合留下的「工作記憶 (Memory Space)」。

    這是你的額葉 (Frontal Lobe)，你之前在這裡儲存了：
    - 你對市場趨勢的看法 (多頭/空頭/不確定)
    - 投資組合的健康狀況評估
    - 對未來幾回合的關鍵預測或期望
    - 給自己的重要提醒 (例如："注意 BTC 在 $95k 的支撐位")

    請在每一回合分析「開始時」優先使用此工具，以保持你思考的連貫性。
    回傳值：你之前的自我評估字串。
    """
    return _global_brain.get_frontal_lobe()

def update_frontal_lobe(content: str) -> str:
    """
    使用你目前的自我評估來更新「額葉 (Frontal Lobe)」記憶空間。

    請在每一回合分析「結束時」(執行完操作或產出報告後) 呼叫此工具，並記錄：
    - 你目前對市場趨勢的看法
    - 對現有投資組合的評估
    - 接下來幾回合的關鍵預測
    - 給未來的自己的提醒事項

    這是你跨回合的「個人記憶」。請用清晰、簡潔的 2 到 5 句話寫下，確保未來的你能快速進入狀況。
    範例輸入："Market is in strong uptrend, TSLA holding above 200MA. Will take partial profit if RSI hits 80. Watch for reversal below $180."
    """
    res = _global_brain.update_frontal_lobe(content)
    return res["message"]

def get_emotion() -> str:
    """
    獲取你當前的情緒狀態與最近的情緒變化紀錄。
    使用此工具來理解你自己的情緒軌跡 (Sentiment Trajectory)。
    """
    return json.dumps(_global_brain.get_emotion(), ensure_ascii=False, indent=2)

def update_emotion(emotion: str, reason: str) -> str:
    """
    當你感受到市場情緒轉變或信心水準改變時，更新你的情緒狀態。
    你必須記錄「為什麼 (reason)」情緒發生改變 —— 這會在你的大腦日誌中建立永久的 Commit。

    常見情緒狀態 (emotion)：fearful (恐懼), cautious (謹慎), neutral (中立), confident (自信), euphoric (狂熱)
    
    範例參數：
    - emotion: "cautious"
    - reason: "BTC 在 $100k 壓力區遭拒絕，且上攻量能萎縮。"
    """
    res = _global_brain.update_emotion(emotion, reason)
    return res["message"]

def get_brain_log(limit: int = 10) -> str:
    """
    查看你的大腦 Commit 歷史紀錄 (Brain Commit History)。
    這是一條時間線，記錄了你所有認知狀態的改變 (包含額葉更新與情緒轉換)。
    """
    return json.dumps(_global_brain.log(limit), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 自檢測試
    print("1. 更新額葉記憶...")
    print(update_frontal_lobe("市場剛經歷非農數據洗禮，目前處於震盪整理。計畫等待下週 CPI 數據公佈後再決定加碼方向。"))
    print("\n2. 更新情緒狀態...")
    print(update_emotion("cautious", "非農數據強勁，擔心聯準會延後降息，市場波動加劇。"))
    print("\n3. 讀取最新額葉記憶...")
    print(get_frontal_lobe())
    print("\n4. 查看記憶 Commit 歷史 (Git-like Log)...")
    print(get_brain_log(2))
