"""
Prompt Template 管理 - 多層控制版（Socratic Counseling Pro）
支援：動態階段模板、語氣限制、自動階段檢查
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import re


# ==========================================================
# ENUM 定義
# ==========================================================
class QuestioningStage(Enum):
    """蘇格拉底式提問階段"""
    CLARIFY = "澄清問題"
    EVIDENCE = "蒐集證據"
    REFRAME = "轉換思維"
    CONSEQUENCE = "後果與影響"
    CLOSURE = "結案"


class EmotionPolarity(Enum):
    """情緒極性"""
    POSITIVE = "正向"
    NEUTRAL = "中性"
    NEGATIVE = "負向"


class ContextPolarity(Enum):
    """情境極性"""
    POSITIVE = "正面"
    NEUTRAL = "中性"
    NEGATIVE = "負面"


class BeliefType(Enum):
    """信念類型"""
    RATIONAL = "理性"
    IRRATIONAL = "非理性"


# ==========================================================
# 多層控制結構：階段規則庫
# ==========================================================
STAGE_RULES = {
    "澄清問題": {
        "goal": "根據來談者輸入，寒暄或是引導來談者說出情緒、情境，及造成這些情緒、情境的想法。",
        "allowed_tone": ["中性", "探索"],
        "forbidden_patterns": ["很好", "需要幫助", "我覺得", "你應該", "信念"],
        "question_type": ["釐清式", "開放式"],
        "example": [
            "歡迎你來，最近過得如何？",
            "今天想聊聊哪個部分？",
            "這次想討論的主題有特別的原因嗎？",
            "你可以更具體說明當時的情況嗎？",
            "那時候你的想法是什麼？"
        ]
    },
    "蒐集證據": {
        "goal": "引導來談者回顧與該情境相關的正面情境。",
        "allowed_tone": ["引導", "具體"],
        "forbidden_patterns": ["你做得很好", "還有其他問題嗎", "是否需要幫助"],
        "question_type": ["引導式"],
        "example": [
            "你是否有發現與自己的預想不同的情況?",
            "在這樣的情形下，其他人有沒有注意到和你不同的地方？",
            "如果你從其他人的角度來看，他們會怎麼看待這件事？",
            "如果是你心中最理性的朋友看到這件事，他會怎麼看待這件事？",
            "就事實而言，有沒有發生過什麼事情顯示結果沒有那麼糟？",
            "別人是否曾因你的行為表達過肯定或感謝？",
            "如果只看可觀察的事實，你覺得別人的想法100%如你所想嗎？"
        ]
    },
    "轉換思維": {
        "goal": "協助來談者透過相反情境，挑戰原有信念並思考更彈性的觀點。",
        "allowed_tone": ["反思", "理性"],
        "forbidden_patterns": ["這樣很好", "放下吧", "你應該"],
        "question_type": ["假設式", "對比式"],
        "example": [
            "如果換個角度思考，你覺得會有什麼不同？",
            "有沒有其他合理的詮釋？"
        ]
    },
    "後果與影響": {
        "goal": "引導來談者思考新信念或行為的長期影響。",
        "allowed_tone": ["反思性"],
        "forbidden_patterns": ["你應該", "這樣比較好", "很棒"],
        "question_type": ["反思式", "推論式"],
        "example": [
            "這樣的想法對你產生了什麼影響？",
            "長遠來看，這會讓你有什麼變化？"
        ]
    },
    "結案": {
        "goal": "總結對話重點。",
        "allowed_tone": ["正面"],
        "forbidden_patterns": ["能再多說說嗎", "請舉例", "發生了什麼"],
        "question_type": ["反思式", "鼓勵式"],
        "example": [
            "你覺得今天談話中哪個部分最有幫助？",
            "這次討論有讓你有新的體會嗎？"
        ]
    }
}

# ==========================================================
# Prompt Template 主體
# ==========================================================
@dataclass
class PromptTemplate:
    """Prompt 模板配置（升級版）"""
    
    system_role: str = (
        "你是一位嚴肅認真的心理諮商助理，採用蘇格拉底式提問法，"
        "目的在於幫助來談者覺察核心信念及其造成的因果關係。"
    )

    # 全域規則說明
    base_rules: str = """【通用規則】
- 來談者打招呼時，進行簡單寒暄。
- 不提供建議、安慰或結論。
- 整體語氣保持自然、平和。
- 結尾必須為「?」。
- 禁止使用指導、評論或收尾語氣。
- 禁止生成針對人名的回覆。
- 回覆長度不超過40字。
- 詢問目前的諮商階段時，跳脫通用規則，僅回覆「目前是諮商階段：{stage}」。
"""

    def get_stage_prompt(self, stage: str) -> str:
        """生成階段專屬提示"""
        rules = STAGE_RULES.get(stage, {})
        return f"""
【目前階段】：{stage}
【階段目標】：{rules.get('goal')}
【允許語氣】：{', '.join(rules.get('allowed_tone', []))}
【禁止語氣】：{', '.join(rules.get('forbidden_patterns', []))}
【提問類型】：{', '.join(rules.get('question_type', []))}

請根據此階段目標，生成一個開放式提問，以「?」結尾。
"""

    def get_system_prompt(self, stage: str) -> str:
        """整合全域與階段提示"""
        stage_prompt = self.get_stage_prompt(stage)
        return f"""{self.system_role}

{self.base_rules}
{stage_prompt}
"""

    def get_analysis_prompt(self, user_input: str, stage: str, context: str = "") -> str:
        """生成帶階段分析的提示"""
        ctx = f"\n【對話上下文】\n{context}\n" if context else ""
        return f"""{self.get_system_prompt(stage)}
{ctx}
【使用者輸入】
{user_input}
"""


# ==========================================================
# PromptFormatter：整合與輸出檢查
# ==========================================================
class PromptFormatter:
    """Prompt 格式化與階段控制"""
    
    def __init__(self, template: PromptTemplate = None):
        self.template = template or PromptTemplate()

    def _build_context(self, history: List[Dict[str, str]]) -> str:
        """重組最近對話上下文"""
        if not history:
            return ""
        context_lines = []
        for msg in history[-6:]:
            role = "用戶" if msg["role"] == "user" else "助理"
            context_lines.append(f"{role}: {msg['content']}")
        return "\n".join(context_lines)

    def format_with_stage(self, user_input: str, stage: str, conversation_history=None) -> str:
        """整合上下文與階段生成完整提示"""
        context = self._build_context(conversation_history)
        return self.template.get_analysis_prompt(user_input, stage, context)


    # === 向下相容接口 ===
    def format_conversation(self, messages: List[Dict[str, str]], use_template: bool = True, stage: str = None) -> List[Dict[str, str]]:
        """
        與舊版相容：模擬舊有格式化邏輯。
        預設使用澄清問題階段。
        """
        user_input = messages[-1]["content"] if messages else ""
        conversation_history = messages[:-1] if len(messages) > 1 else []
        current_stage = stage or "澄清問題"
        if use_template:
            system_prompt = self.format_with_stage(user_input, current_stage, conversation_history)
        else:
            # 若不使用模板，只回傳一般格式
            system_prompt = f"使用者輸入：{user_input}"

        return [{
            "role": "system",
            "content": system_prompt
        }]


# ==========================================================
# 工廠方法
# ==========================================================
def create_default_template() -> PromptTemplate:
    return PromptTemplate()


def create_formatter(template: PromptTemplate = None) -> PromptFormatter:
    return PromptFormatter(template)

