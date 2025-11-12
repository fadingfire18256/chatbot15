"""
config2.py  
升級版配置文件 - 保留模型、生成、提示與記憶設定（支援分析與PostgreSQL記憶）
"""

from dataclasses import dataclass, field
import torch


# =========================================================
# 🧠 模型設定
# =========================================================
@dataclass
class ModelConfig:
    """模型設定"""
    model_path: str = "D:/AI/trychatbot15/Qwen2.5-3B"
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    device_map: dict = field(default_factory=lambda: {"": 0})
    trust_remote_code: bool = True


# =========================================================
# ⚙️ 生成設定
# =========================================================
@dataclass
class GenerationConfig:
    """生成參數"""
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: float = 1.2

    def to_generate_kwargs(self, pad_token_id: int, eos_token_id: int) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
        }


# =========================================================
# 💬 Prompt 設定
# =========================================================
@dataclass
class PromptConfig:
    """Prompt 模板設定"""
    use_socratic_template: bool = True
    include_analysis: bool = True           # ✅ 開啟分析功能
    context_window_size: int = 3


# =========================================================
# 🧠 記憶系統設定（PostgreSQL / Docker）
# =========================================================
@dataclass
class MemoryConfig:
    """記憶系統設定"""
    use_summary_memory: bool = True         # ✅ 啟用記憶系統
    max_token_limit: int = 2000

    # Docker PostgreSQL 設定
    db_host: str = "localhost"
    db_port: int = 5433
    db_name: str = "main"
    db_user: str = "postgres"
    db_password: str = "mypassword"

    auto_save_on_closure: bool = True
    compression_strategy: str = "summary"   # 可選 "summary" 或 "buffer"


# =========================================================
# 🖥️ UI 設定（供 Streamlit 使用）
# =========================================================
@dataclass
class UIConfig:
    """UI 顯示設定"""
    page_title: str = "Qwen2.5 Chatbot"
    page_icon: str = "🤖"
    layout: str = "centered"


# =========================================================
# 📦 總設定
# =========================================================
@dataclass
class AppConfig:
    """應用程式設定集合"""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)  # ✅ 加回記憶設定
    ui: UIConfig = field(default_factory=UIConfig)


# =========================================================
# 🏁 工具函式
# =========================================================
def get_default_config() -> AppConfig:
    """回傳預設設定"""
    return AppConfig()
