import os
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

st.set_page_config(page_title="📚 문서 기반 GPT 챗봇", layout="wide")
st.title("📚 로컬 문서 기반 GPT 챗봇")

if "ready" not in st.session_state:
    st.session_state["ready"] = False
if "vectordb" not in st.session_state:
    st.session_state["vectordb"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 문서 로딩 함수
def load_documents():
    with st.spinner("📁 문서를 불러오는 중입니다..."):
        all_docs = []
        for file in os.listdir(doc_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(doc_dir, file))
                all_docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(all_docs)
        vectordb = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=api_key))
        st.session_state["vectordb"] = vectordb
        st.session_state["ready"] = True

# 최초 실행 시 문서 자동 로드
if not st.session_state["ready"]:
    load_documents()

# 대화 UI 출력
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("💬 질문을 입력하세요:")

if user_input and st.session_state["ready"]:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 GPT가 답변 중입니다..."):
            docs = st.session_state["vectordb"].similarity_search(user_input, k=5)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_input)
            st.markdown(response)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
