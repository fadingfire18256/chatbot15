"""
main_min.py  
目標：保留依賴注入與 prompt template，但只顯示
1️⃣ 標題  2️⃣ 對話欄  3️⃣ 對話紀錄
"""

import streamlit as st
from config2 import AppConfig, get_default_config
from llm_loader2 import AgentFactory, QwenAgent


# =========================================================
# 🧠 UI 控制類別（精簡版）
# =========================================================
class StreamlitUI:
    """極簡 UI：只保留標題、輸入框與歷史紀錄"""

    def __init__(self, config: AppConfig, agent: QwenAgent):
        self.config = config
        self.agent = agent

    def setup_page(self):
        """設定頁面屬性"""
        #瀏覽器頁面標籤
        st.set_page_config(
            page_title=self.config.ui.page_title,
            page_icon=self.config.ui.page_icon,
            layout="centered",
        )
        #頁面中央的標題
        st.title(f"{self.config.ui.page_icon} {self.config.ui.page_title}")

        # 🔹 新增：顯示目前諮商階段
    def render_chat_history(self):
        """顯示歷史對話"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
            #這邊的.messages其實就是建立一個messages為開頭的session_state dict

        for message in st.session_state.messages:
            #如果已經有一個messages dict在session_state裡面
            with st.chat_message(message["role"]):
                st.write(message["content"])
                #應用於messages:{"role": "user", "content": "你好"}
                #通常chat_message會和write配合，先建立角色對話氣泡，再寫入內容
                #總結來說，render_chat_history會抓handle_user_input建立的messages dict，依據其中session_state存入的對話，以對話氣泡顯示

    def handle_user_input(self, user_input: str):
        """處理使用者輸入並呼叫 Agent"""
        # 顯示使用者訊息
        st.chat_message("user").write(user_input)
        #這個寫法不同於render_chat_hsitory，這邊是處理即時輸入，而只顯示單行訊息
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 呼叫 Agent
        response = self.agent.process(st.session_state.messages)
        #把st.session_state.messages傳入process，作為process的messages。
        #process包含:轉換messages格式變成prompt排版/套用Qwen讀得懂的模板讓Qwen讀上下文(autotokenizer)/input真正轉成token+生成output/
        #解碼第一個output，並切除output中input原本的句長作為真正的response/紀錄response中，經過分析後key的value/暫時關閉紀錄梯度/
        #加入新對話到記憶/累加所有的情緒和信念/更新當下的階段/儲存id、摘要、最後階段、信念及情緒趨勢、總輪次

        # 顯示回覆
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    def run(self):
        """運行主畫面"""
        self.setup_page()
        self.render_chat_history()

        # 輸入欄位
        user_input = st.chat_input("請輸入你的問題...")
        if user_input:
            self.handle_user_input(user_input)
    
# =========================================================
# ⚙️ 快取載入 Agent（避免重複載入模型）
# =========================================================
@st.cache_resource
def load_agent(config: AppConfig) -> QwenAgent:
    with st.spinner("🔄 載入模型中..."):
        return AgentFactory.create_agent(config)


# =========================================================
# 🚀 主函式（Dependency Injection 保留）
# =========================================================
def main():
    config = get_default_config()
    agent = load_agent(config)
    ui = StreamlitUI(config, agent)
    ui.run()


# =========================================================
# 📌 程式入口點
# =========================================================
if __name__ == "__main__":
    main()
