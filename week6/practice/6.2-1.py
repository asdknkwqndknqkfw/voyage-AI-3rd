from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

import streamlit as st
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"openai_api_key: {openai_api_key[:5]}")


def streamlit_test2():
    model = ChatOpenAI(model="gpt-4o-mini")

    st.title("GPT Bot")

    # Session state 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 만약 app이 rerun하면 message들을 다시 UI에 띄우기
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 유저의 메시지를 받았을 때, response로 GPT API의 답변
    if prompt := st.chat_input("placeholder"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            msgs = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    msgs.append(HumanMessage(content=msg["content"]))
                else:
                    msgs.append(AIMessage(content=msg["content"]))

            res = model.invoke(msgs)
            resp = res.content

            st.markdown(resp)

            st.session_state.messages.append({"role": "assistance", "content": resp})


if __name__ == "__main__":
    streamlit_test2()
