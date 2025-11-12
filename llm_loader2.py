"""
llm_loader2.py  
中階版本：保留 PromptTemplate、分析功能、完整 PostgreSQL 記憶架構
刪除 UI 與日誌相關程式
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config2 import AppConfig
from prompt_templates2_pro import create_formatter
from memory_manager import MemoryManager  # ✅ 使用原始的 DB 記憶架構


# =========================================================
# 🧠 模型載入器
# =========================================================
class ModelLoader:
    """根據 config 載入模型與 tokenizer"""
    def __init__(self, config: AppConfig):
        self.config = config

    def load(self):
        """載入模型與 tokenizer"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.model.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.config.model.bnb_4bit_compute_dtype
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_path,
            trust_remote_code=self.config.model.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_path,
            quantization_config=bnb_config,
            trust_remote_code=self.config.model.trust_remote_code,
            device_map=self.config.model.device_map
        )
        return model, tokenizer


# =========================================================
# 🤖 Qwen 對話代理（主核心）
# =========================================================
class QwenAgent:
    """具備 PromptTemplate、分析與資料庫記憶功能的對話代理"""
    def __init__(self, model, tokenizer, config: AppConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_formatter = create_formatter()
        self.memory = MemoryManager(model, tokenizer, config.memory)  # ✅ 使用真實 PostgreSQL 記憶管理器
        self.last_analysis = {}

    # -----------------------------------------------------
    # 🌐 主流程：對話回覆
    # -----------------------------------------------------
    def process(self, messages):
        """生成回覆、分析與記憶更新"""
        # 1️⃣ 格式化對話內容
        if self.config.prompt.use_socratic_template:
            current_stage = self.memory.current_stage
            formatted_messages = self.prompt_formatter.format_conversation(
                messages, use_template=True, stage=current_stage
            )
        else:
            formatted_messages = messages

        # 2️⃣ 套用 chat template
        text = self.tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True
        )
        #這邊的tokenizer，是autotokenizer，因此套用Qwen讀得懂的模板讓Qwen讀上下文。

        # 3️⃣ 模型生成
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            #input真正轉成token+生成output
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                repetition_penalty=self.config.generation.repetition_penalty,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # 4️⃣ 解析分析結果
        self.last_analysis = self.extract_analysis(response)

        # 5️⃣ 更新資料庫記憶
        try:
            # 將用戶與 AI 對話加入記憶
            for msg in messages:
                self.memory.add_message(msg["role"], msg["content"])
            self.memory.add_message("assistant", response)
            # 更新情緒、信念、階段等分析資訊
            self.memory.update_analysis(self.last_analysis)
            # 若為結案階段，自動儲存 Session
            if self.memory.is_closure_stage(self.last_analysis):
                self.memory.save_session()
        except Exception as e:
            print(f"[Memory Error] 無法更新資料庫記憶: {str(e)}")

        # 6️⃣ 回傳 LLM 回覆
        torch.cuda.empty_cache()
        return response

    # -----------------------------------------------------
    # 📊 分析模組（維持輕量）
    # -----------------------------------------------------
    def extract_analysis(self, response: str):
        """根據 LLM 回覆解析情緒、信念、階段"""
        analysis = {"emotion": "未知", "context": "未知", "belief": "未知", "stage": "未知"}
        for key in analysis.keys():
            token = f"【{key}】"
            if token in response:
                try:
                    analysis[key] = response.split(token)[1].split("\n")[0].strip()
                except:
                    pass
        return analysis

    def get_analysis(self):
        """供 Streamlit 呼叫顯示分析卡"""
        return self.last_analysis

    def get_memory_info(self):
        """從 PostgreSQL 抓取當前記憶狀態"""
        try:
            return self.memory.get_memory_summary()
        except Exception as e:
            return {"enabled": False, "error": str(e)}


# =========================================================
# 🏭 Agent 工廠（依賴注入）
# =========================================================
class AgentFactory:
    """建立 Agent 實例"""
    @staticmethod
    def create_agent(config: AppConfig) -> QwenAgent:
        loader = ModelLoader(config)
        model, tokenizer = loader.load()
        return QwenAgent(model, tokenizer, config)
