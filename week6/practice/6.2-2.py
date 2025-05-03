from dotenv import load_dotenv
from huggingface_hub import login
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

import streamlit as st
import os
import torch

load_dotenv()

torch.classes.__path__ = []


@st.cache_resource
def get_model():
    hf_key = os.getenv("HF_TOKEN")
    print(f"hf_key: {hf_key[:5]}")

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-2b-it",
        task="text-generation",
        device=0,
        pipeline_kwargs={"max_new_tokens": 256, "do_sample": False},
    )
    model = ChatHuggingFace(llm=llm)
    return model


def streamlit_test3():
    st.title("HF Bot")

    # Session state 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 만약 app이 rerun하면 message들을 다시 UI에 띄우기
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    model = get_model()

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
            print(f"res.content: {res.content}")
            resp = res.content

            st.markdown(resp)
            st.session_state.messages.append({"role": "assistance", "content": resp})


if __name__ == "__main__":
    streamlit_test3()
