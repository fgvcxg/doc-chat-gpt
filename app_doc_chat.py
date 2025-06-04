import os
import time
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# ✅ 환경 변수 로드 및 진단
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
helicone_key = os.getenv("HELICONE_API_KEY")

# 🔍 환경 변수 진단 출력
st.sidebar.title("🧪 .env 진단 결과")
st.sidebar.code(f"OPENAI_API_KEY: {api_key[:10]}********" if api_key else "❌ api_key 없음")
st.sidebar.code(f"HELICONE_API_KEY: {helicone_key[:10]}********" if helicone_key else "❌ helicone_key 없음")

# 📁 기본 설정
doc_dir = "docs"
last_update_file = "last_update.txt"

st.set_page_config(page_title="📚 문서 기반 GPT 챗봇", layout="wide")
st.title("📚 로컬 문서 기반 GPT 챗봇")

if "ready" not in st.session_state:
    st.session_state["ready"] = False
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "gpt-3.5-turbo"

model_name = st.selectbox("🤖 사용할 GPT 모델을 선택하세요", ["gpt-3.5-turbo", "gpt-4o"], index=0)
st.session_state["model_name"] = model_name

need_reload = False
latest_mtime = 0
modified_files = []

for file in os.listdir(doc_dir):
    if file.endswith(".pdf"):
        path = os.path.join(doc_dir, file)
        mtime = os.path.getmtime(path)
        if mtime > latest_mtime:
            latest_mtime = mtime
        modified_files.append((file, mtime))

last_saved_time = 0
if os.path.exists(last_update_file):
    with open(last_update_file, "r") as f:
        last_saved_time = float(f.read())

if latest_mtime > last_saved_time:
    need_reload = True

if need_reload or not os.path.exists(last_update_file):
    with st.spinner("📁 문서를 불러오는 중입니다..."):
        all_docs = []
        for file in os.listdir(doc_dir):
            if file.endswith(".pdf"):
                path = os.path.join(doc_dir, file)
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(all_docs)
        vectordb = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=api_key))
        st.session_state["vectordb"] = vectordb

        with open(last_update_file, "w") as f:
            f.write(str(time.time()))

        st.session_state["ready"] = True
else:
    if not st.session_state["vectordb"]:
        all_docs = []
        for file in os.listdir(doc_dir):
            if file.endswith(".pdf"):
                path = os.path.join(doc_dir, file)
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(all_docs)
        vectordb = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=api_key))
        st.session_state["vectordb"] = vectordb
        st.session_state["ready"] = True

if st.session_state["ready"]:
    st.success("✅ 문서 로드 완료! 지금 바로 질문해보세요.")
else:
    st.warning("⏳ 문서를 불러오는 중입니다. 잠시만 기다려 주세요...")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("💬 질문을 입력하세요:") if st.session_state["ready"] else None

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 GPT가 답변 중입니다..."):
            docs_and_scores = st.session_state["vectordb"].similarity_search_with_score(user_input, k=5)
            docs = [doc for doc, score in docs_and_scores if score > 0.3]

            llm = ChatOpenAI(
                model_name=st.session_state["model_name"],
                temperature=0,
                openai_api_key=api_key,
                base_url="https://oai.helicone.ai/v1",
                default_headers={
                    "Helicone-Auth": f"Bearer {helicone_key}"
                }
            )

            system_prompt = f"""
            너는 문서를 기반으로 정보를 제공하는 전문가야.
            아래 문서들을 참고해서 사용자 질문에 대해 항상 한국어로 자세히 설명해줘.
            문서가 영어로 되어 있어도 답변은 반드시 한국어로 해.
            그리고 한국어로 질문해도 문서들이 영어일 경우 답변이 안될 수 있으니까 한국어로 질문해도 한번 영어로 변경해서 답변을 찾아봐줘.
            질문: {user_input}
            """

            chain = load_qa_chain(llm, chain_type="stuff")

            response = ""
            try:
                if docs:
                    response = chain.run(input_documents=docs, question=system_prompt)
                else:
                    response = "⚠️ 관련된 문서를 찾을 수 없습니다. 질문을 다르게 표현해 보세요."
            except Exception as e:
                response = f"❌ GPT 처리 중 오류가 발생했습니다:\n\n{str(e)}"

            st.markdown(response)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
