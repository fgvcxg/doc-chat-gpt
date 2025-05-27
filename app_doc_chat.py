import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
doc_dir = "docs"
last_update_file = "last_update.txt"

st.set_page_config(page_title="📚 문서 기반 GPT 챗봇", layout="wide")
st.title("📚 로컬 문서 기반 GPT 챗봇")

# ✅ 문서 준비 상태 표시
if "ready" not in st.session_state:
    st.session_state["ready"] = False
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "gpt-3.5-turbo"

# 모델 선택 드롭다운
model_name = st.selectbox("🤖 사용할 GPT 모델을 선택하세요", ["gpt-3.5-turbo", "gpt-4o"], index=0)
st.session_state["model_name"] = model_name

if st.session_state["ready"]:
    st.success("✅ 문서 로드 완료! 지금 바로 질문해보세요.")
else:
    st.warning("⏳ 문서를 불러오는 중입니다. 잠시만 기다려 주세요...")

# 문서 자동 업데이트 및 로딩 함수
def load_documents():
    with st.spinner("📁 문서를 불러오는 중입니다..."):
        all_docs = []
        updated_files = []
        last_time = 0
        if os.path.exists(last_update_file):
            with open(last_update_file, "r") as f:
                last_time = float(f.read())

        for file in os.listdir(doc_dir):
            if file.endswith(".pdf"):
                path = os.path.join(doc_dir, file)
                if os.path.getmtime(path) > last_time:
                    loader = PyPDFLoader(path)
                    all_docs.extend(loader.load())
                    updated_files.append(file)

        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(all_docs)
            vectordb = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=api_key))
            st.session_state["vectordb"] = vectordb
            st.success(f"🔄 {len(updated_files)}개의 문서를 새로 불러왔습니다.")
        else:
            st.info("✅ 업데이트된 문서가 없어 기존 데이터를 사용합니다.")

        with open(last_update_file, "w") as f:
            f.write(str(time.time()))

        st.session_state["ready"] = True

# 최초 실행 시 문서 자동 로드
if not st.session_state["ready"]:
    load_documents()

# 대화 UI 출력
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력창 (문서 로드 완료 여부에 따라 활성/비활성)
if st.session_state["ready"]:
    user_input = st.chat_input("💬 질문을 입력하세요:")
else:
    st.chat_input("⏳ 문서를 불러오는 중입니다... (입력이 비활성화됩니다)", disabled=True)
    user_input = None

if user_input and st.session_state["ready"]:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 GPT가 답변 중입니다..."):
            # 유사도 점수 포함 검색
            docs_and_scores = st.session_state["vectordb"].similarity_search_with_score(user_input, k=5)
            docs = [doc for doc, score in docs_and_scores if score > 0.3]  # 점수 필터링

            # 커스텀 프롬프트 추가
            system_prompt = f"""
            너는 문서를 기반으로 정보를 제공하는 전문가야.
            아래 문서들을 참고해서 사용자 질문에 대해 항상 한국어로 자세히 설명해줘.
            문서가 영어로 되어 있어도 답변은 반드시 한국어로 해.
            그리고 한국어로 질문해도 문서들이 영어일 경우 답변이 안될 수 있으니까 한국어로 질문해도 한번 영어로 변경해서 답변을 찾아봐줘.
            질문: {user_input}
            """

            llm = ChatOpenAI(model_name=st.session_state["model_name"], temperature=0, openai_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=system_prompt)
            st.markdown(response)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
