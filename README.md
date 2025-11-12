**專案簡介**  

本專案為一個整合 大型語言模型（LLM）、心理諮商階段模板（Socratic Counseling）、PostgreSQL 記憶管理系統 的 Streamlit 聊天應用。  
系統可依據使用者對話內容進行階段性提問（澄清→蒐證→重構→後果→結案），並將歷史摘要、情緒趨勢與信念變化儲存於資料庫中，以支援持續對談與深層探索。  

---
**專案架構**  

<img width="1717" height="962" alt="image" src="https://github.com/user-attachments/assets/9de08299-d324-47c5-9cc9-26357182b444" />  

**main_min.py – 使用者介面層（UI Layer)**  

以 Streamlit 建立極簡對話介面。  
顯示歷史訊息、接收使用者輸入。  
將完整訊息序列傳入 QwenAgent 進行處理與生成回覆。  

**llm_loader2.py – 模型與代理層（Model & Agent Layer）**

使用 Transformers + BitsAndBytesConfig 載入 4bit 量化的 Qwen2.5 模型。  
QwenAgent 統籌：  
格式化對話模板（PromptFormatter）  
生成 LLM 回覆與分析（emotion, belief, stage）  
透過 MemoryManager 儲存情緒趨勢與會談摘要至 PostgreSQL  

**memory_manager.py – 記憶管理層（Memory Layer）**  

整合 LangChain ConversationSummaryMemory 與資料庫。  
儲存與提取歷史摘要、信念與情緒趨勢。  
判斷階段是否為「結案」，並自動觸發 save_session()。  

**database.py – 資料庫層（Persistence Layer）** 
管理 PostgreSQL Connection Pool。
建立與操作 session_summary 表格。
提供：
save_session_summary()：儲存或更新對話摘要。
get_session_summary()：查詢特定 session。
get_user_sessions()：提取使用者歷史紀錄。

**prompt_templates2_pro.py – 提示模板層（Prompt Layer）**  
定義蘇格拉底五階段提問模板。  
限制語氣、控制提問邏輯、範例問題。  
由 PromptFormatter 將使用者輸入與上下文整合成可供 LLM 理解的 system prompt。  

**config2.py – 全域設定層（Configuration Layer）**  
以 dataclass 結構化設定：  
模型參數（ModelConfig）  
生成參數（GenerationConfig）  
提示控制（PromptConfig）  
記憶與資料庫設定（MemoryConfig）  
UI 顯示設定（UIConfig）  

---  
**執行流程**  

啟動主程式：  
streamlit run main_min.py  

**功能**  

系統載入：  
初始化模型與 tokenizer。  
建立 PostgreSQL 資料庫連線池。  
從資料庫載入上次 session 摘要至記憶。  

使用者對話：  
輸入訊息 → 由 PromptFormatter 生成 system prompt。  
模型生成回覆 → 抽取 emotion/context/belief/stage。  

記憶與分析：  
自動記錄對話輪次。  
在「結案階段」自動呼叫 save_session() 儲存歷史摘要。  
