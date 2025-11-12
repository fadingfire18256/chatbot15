"""
資料庫管理 - PostgreSQL 操作
"""

import psycopg2
from psycopg2 import pool, extras
from typing import Optional, Dict, List
from datetime import datetime
import logging

from config import MemoryConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """PostgreSQL 資料庫管理器"""
    
    def __init__(self, config: MemoryConfig):
        """
        初始化資料庫連線
        
        Args:
            config: 記憶系統配置
        """
        self.config = config
        self.connection_pool = None
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """初始化連線池"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1,  # 最小連線數
                10,  # 最大連線數
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            logger.info("✅ 資料庫連線池初始化成功")
        except psycopg2.Error as e:
            logger.error(f"❌ 資料庫連線失敗: {e}")
            self.connection_pool = None
    
    def get_connection(self):
        """從連線池獲取連線"""
        if self.connection_pool:
            return self.connection_pool.getconn()
        return None
    
    def return_connection(self, conn):
        """歸還連線到連線池"""
        if self.connection_pool and conn:
            self.connection_pool.putconn(conn)
    
    def create_tables(self):
        """創建必要的資料表"""
        conn = self.get_connection()
        if not conn:
            logger.error("❌ 無法獲取資料庫連線")
            return False
        
        try:
            cursor = conn.cursor()
            
            # 創建 session_summary 資料表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_summary (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    session_id VARCHAR(36) UNIQUE NOT NULL,
                    summary_text TEXT NOT NULL,
                    stage_completed VARCHAR(50),
                    emotion_trend TEXT,
                    belief_change TEXT,
                    total_turns INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # 創建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON session_summary(user_id);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON session_summary(created_at DESC);
            """)
            
            conn.commit()
            cursor.close()
            logger.info("✅ 資料表創建成功")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ 資料表創建失敗: {e}")
            conn.rollback()
            return False
        finally:
            self.return_connection(conn)
    
    def save_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary_text: str,
        stage_completed: str = None,
        emotion_trend: str = None,
        belief_change: str = None,
        total_turns: int = 0
    ) -> bool:
        """
        儲存 session 摘要
        
        Args:
            user_id: 使用者 ID
            session_id: Session ID
            summary_text: 摘要文本
            stage_completed: 完成的階段
            emotion_trend: 情緒趨勢
            belief_change: 信念變化
            total_turns: 對話輪數
            
        Returns:
            是否儲存成功
        """
        conn = self.get_connection()
        if not conn:
            logger.error("❌ 無法獲取資料庫連線")
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO session_summary 
                (user_id, session_id, summary_text, stage_completed, 
                 emotion_trend, belief_change, total_turns, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (session_id) 
                DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    stage_completed = EXCLUDED.stage_completed,
                    emotion_trend = EXCLUDED.emotion_trend,
                    belief_change = EXCLUDED.belief_change,
                    total_turns = EXCLUDED.total_turns,
                    updated_at = NOW();
            """, (user_id, session_id, summary_text, stage_completed, 
                  emotion_trend, belief_change, total_turns))
            
            conn.commit()
            cursor.close()
            logger.info(f"✅ Session 摘要已儲存: {session_id}")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ 儲存 Session 摘要失敗: {e}")
            conn.rollback()
            return False
        finally:
            self.return_connection(conn)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """
        獲取 session 摘要
        
        Args:
            session_id: Session ID
            
        Returns:
            摘要資料字典
        """
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM session_summary
                WHERE session_id = %s
            """, (session_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return dict(result)
            return None
            
        except psycopg2.Error as e:
            logger.error(f"❌ 獲取 Session 摘要失敗: {e}")
            return None
        finally:
            self.return_connection(conn)
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict]:
        """
        獲取使用者的歷史 sessions
        
        Args:
            user_id: 使用者 ID
            limit: 限制數量
            
        Returns:
            session 列表
        """
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM session_summary
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (user_id, limit))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except psycopg2.Error as e:
            logger.error(f"❌ 獲取使用者 Sessions 失敗: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def close(self):
        """關閉連線池"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("✅ 資料庫連線池已關閉")


def init_database(config: MemoryConfig) -> bool:
    """
    初始化資料庫（創建資料表）
    
    Args:
        config: 記憶系統配置
        
    Returns:
        是否初始化成功
    """
    db_manager = DatabaseManager(config)
    success = db_manager.create_tables()
    db_manager.close()
    return success

