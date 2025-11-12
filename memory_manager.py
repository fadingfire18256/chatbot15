"""
記憶管理器 - 使用 LangChain 的 ConversationSummaryMemory
"""

from typing import List, Dict, Optional, Tuple
import logging
import uuid
from datetime import datetime

from langchain.memory import ConversationSummaryMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

from config2 import MemoryConfig
from database import DatabaseManager
from prompt_templates2_pro import QuestioningStage

logger = logging.getLogger(__name__)


class MemoryManager:
    """記憶管理器 - 整合 LangChain 記憶系統與資料庫"""
    
    def __init__(self, model, tokenizer, config: MemoryConfig):
        """
        初始化記憶管理器
        
        Args:
            model: Transformers 模型
            tokenizer: Tokenizer
            config: 記憶系統配置
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # 初始化 LangChain 記憶系統
        if self.config.use_summary_memory:
            self.memory = self._init_langchain_memory()
        else:
            self.memory = None
        
        # 初始化資料庫管理器
        self.db_manager = DatabaseManager(config)
        
        # Session 管理
        self.current_session_id = str(uuid.uuid4())
        self.current_user_id = "default_user"  # 可從 UI 設定
        self.conversation_turns = 0
        self.current_stage = QuestioningStage.CLARIFY.value
        
        # 記錄情緒和信念變化
        self.emotion_history: List[str] = []
        self.belief_history: List[str] = []
    
    def _init_langchain_memory(self) -> ConversationSummaryMemory:
        """初始化 LangChain ConversationSummaryMemory"""
        try:
            # 創建 HuggingFace Pipeline for LangChain
            text_gen_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            
            # 包裝為 LangChain LLM
            llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            
            # 創建 ConversationSummaryMemory
            memory = ConversationSummaryMemory(
                llm=llm,
                max_token_limit=self.config.max_token_limit,
                return_messages=True
            )
            
            logger.info("✅ LangChain 記憶系統初始化成功")
            return memory
            
        except Exception as e:
            logger.error(f"❌ LangChain 記憶系統初始化失敗: {e}")
            return None
    
    def add_message(self, role: str, content: str):
        """
        添加訊息到記憶系統
        
        Args:
            role: 角色 (user/assistant)
            content: 訊息內容
        """
        if not self.memory:
            return
        
        try:
            if role == "user":
                self.memory.chat_memory.add_user_message(content)
            elif role == "assistant":
                self.memory.chat_memory.add_ai_message(content)
            
            self.conversation_turns += 1
            logger.debug(f"📝 訊息已加入記憶: {role}")
            
        except Exception as e:
            logger.error(f"❌ 添加訊息失敗: {e}")
    
    def get_summary(self) -> str:
        """
        獲取對話摘要
        
        Returns:
            對話摘要文本
        """
        if not self.memory:
            return "記憶系統未啟用"
        
        try:
            # 使用 LangChain 的 buffer 屬性獲取摘要
            summary = self.memory.buffer
            
            if not summary:
                # 如果沒有摘要，手動生成
                messages = self.memory.chat_memory.messages
                if messages:
                    summary = f"對話共 {len(messages)} 輪，包含 {self.conversation_turns} 個回合。"
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 獲取摘要失敗: {e}")
            return "無法生成摘要"
    
    def get_memory_context(self) -> str:
        """
        獲取記憶上下文（用於注入 Prompt）
        
        Returns:
            記憶上下文文本
        """
        if not self.memory:
            return ""
        
        try:
            # 獲取最近的對話記錄
            messages = self.memory.chat_memory.messages
            if not messages:
                return ""
            
            # 格式化為文本
            context_lines = []
            for msg in messages[-6:]:  # 最近3輪對話
                role = "用戶" if msg.type == "human" else "助理"
                context_lines.append(f"{role}: {msg.content}")
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"❌ 獲取記憶上下文失敗: {e}")
            return ""
    
    def update_analysis(self, analysis: Dict):
        """
        更新分析資訊
        
        Args:
            analysis: 分析結果字典
        """
        # 更新情緒歷史
        if "emotion" in analysis:
            self.emotion_history.append(analysis["emotion"])
        
        # 更新信念歷史
        if "belief" in analysis:
            self.belief_history.append(analysis["belief"])
        
        # 更新當前階段
        if "stage" in analysis:
            stage_text = analysis["stage"]
            # 提取階段名稱（可能包含其他文字）
            for stage in QuestioningStage:
                if stage.value in stage_text:
                    self.current_stage = stage.value
                    break
    
    def is_closure_stage(self, analysis: Dict) -> bool:
        """
        判斷是否進入結案階段
        
        Args:
            analysis: 分析結果字典
            
        Returns:
            是否為結案階段
        """
        if "stage" in analysis:
            stage_text = analysis.get("stage", "")
            return QuestioningStage.CLOSURE.value in stage_text
        return False
    
    def save_session(self) -> bool:
        """
        儲存當前 session 到資料庫
        
        Returns:
            是否儲存成功
        """
        try:
            # 獲取摘要
            summary = self.get_summary()
            
            # 分析情緒趨勢
            emotion_trend = self._analyze_emotion_trend()
            
            # 分析信念變化
            belief_change = self._analyze_belief_change()
            
            # 儲存到資料庫
            success = self.db_manager.save_session_summary(
                user_id=self.current_user_id,
                session_id=self.current_session_id,
                summary_text=summary,
                stage_completed=self.current_stage,
                emotion_trend=emotion_trend,
                belief_change=belief_change,
                total_turns=self.conversation_turns
            )
            
            if success:
                logger.info(f"✅ Session 已儲存: {self.current_session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 儲存 Session 失敗: {e}")
            return False
    
    def _analyze_emotion_trend(self) -> str:
        """分析情緒趨勢"""
        if not self.emotion_history:
            return "無情緒記錄"
        
        # 簡單分析：計算正向、中性、負向數量
        positive_count = sum(1 for e in self.emotion_history if "正向" in e)
        negative_count = sum(1 for e in self.emotion_history if "負向" in e)
        neutral_count = len(self.emotion_history) - positive_count - negative_count
        
        return f"正向:{positive_count} 中性:{neutral_count} 負向:{negative_count}"
    
    def _analyze_belief_change(self) -> str:
        """分析信念變化"""
        if not self.belief_history:
            return "無信念記錄"
        
        # 簡單分析：從非理性到理性的變化
        has_irrational = any("非理性" in b for b in self.belief_history)
        has_rational = any("理性" in b for b in self.belief_history)
        
        if has_irrational and has_rational:
            return "從非理性轉為理性"
        elif has_irrational:
            return "持續非理性信念"
        elif has_rational:
            return "持續理性信念"
        else:
            return "未識別信念類型"
    
    def reset_session(self):
        """重置 session（結案後開始新一輪）"""
        logger.info(f"🔄 重置 Session: {self.current_session_id}")
        
        # 生成新的 session ID
        self.current_session_id = str(uuid.uuid4())
        
        # 重置計數器
        self.conversation_turns = 0
        
        # 重置階段
        self.current_stage = QuestioningStage.CLARIFY.value
        
        # 清空歷史
        self.emotion_history = []
        self.belief_history = []
        
        # 清空記憶（如果需要）
        if self.memory:
            self.memory.clear()
    
    def get_session_info(self) -> Dict:
        """
        獲取當前 session 資訊
        
        Returns:
            session 資訊字典
        """
        return {
            "session_id": self.current_session_id,
            "user_id": self.current_user_id,
            "conversation_turns": self.conversation_turns,
            "current_stage": self.current_stage,
            "emotion_trend": self._analyze_emotion_trend(),
            "belief_change": self._analyze_belief_change(),
            "summary": self.get_summary()
        }
    
    def set_user_id(self, user_id: str):
        """設定使用者 ID"""
        self.current_user_id = user_id
        logger.info(f"👤 使用者 ID 已設定: {user_id}")
    
    def close(self):
        """關閉資料庫連線"""
        if self.db_manager:
            self.db_manager.close()

