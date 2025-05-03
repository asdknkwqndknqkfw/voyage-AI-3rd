from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

import os
import streamlit as st
import base64

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f'OPENAI_API_KEY: {OPENAI_API_KEY[:7]}')

model_name = "gpt-4o-mini"
model = ChatOpenAI(model=model_name)

st.title("Fashion Recommendation Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Question what you want")

# file uploader
if images := st.file_uploader(
    "본인의 전신이 보이는 사진을 올려주세요!", 
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True  # 여러 이미지를 입력으로 받기
):
    image_contents = []
    
    for _image in images:
        st.image(_image)
        image_data = base64.b64encode(_image.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_data}"
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": image_url},
        })
    
    if prompt:
        # 과거 메시지 복원
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 주어진 사진을 기준으로 패션 아이템을 추천할 수 있도록 prompt를 준비
        with st.chat_message("assistant"):
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    *image_contents
                ],
            )
            result = model.invoke([message])
            response = result.content
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
