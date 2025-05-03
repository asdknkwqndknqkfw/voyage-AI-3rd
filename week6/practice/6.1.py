import streamlit as st


def streamlit_test1():
    st.title("ChatBot")

    if "messages" not in st.session_state:
        st.session_state.messages = []  # 유저의 AI 간의 대화를 저장

    # messages에 저장되어 있는 메시지들을 UI에 띄우는 코드
    for message in st.session_state.messages:
        print(f"message: {message}")
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "What is up?"
    ):  # User의 input을 기다립니다. Placeholder로 "What is up?"라는 문구를 사용합니다.
        with st.chat_message("user"):
            st.markdown(prompt)  # User의 메시지를 기록
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )  # User의 메시지를 session_state에 저장

        response = f"Echo: {prompt}"  # 우리가 내놓을 답변으로 user가 보낸 메시지를 사용

        # User의 메시지를 처리
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistance", "content": response}
            )


if __name__ == "__main__":
    streamlit_test1()
