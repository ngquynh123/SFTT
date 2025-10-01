# -*- coding: utf-8 -*-
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_models import ChatOllama

try:
    from langchain_core.chat_history import InMemoryChatMessageHistory  # >=0.2
except Exception:
    from langchain_community.chat_message_histories import ChatMessageHistory as InMemoryChatMessageHistory

_SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]

def build_memory_chain(model_id: str, base_url: str = "http://localhost:11434", temperature: float = 0.2):
    llm = ChatOllama(model=model_id, base_url=base_url, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Bạn là trợ lý AI về LÝ THUYẾT LÁI XE Ô TÔ tại Việt Nam. "
         "Chỉ dựa vào NỘI DUNG THAM CHIẾU nếu có. Trả lời ngắn gọn, trung lập."),
        MessagesPlaceholder(variable_name="history"),
        ("system", "NỘI DUNG THAM CHIẾU (nếu có):\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    return with_history
