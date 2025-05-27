import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# 문서 디렉토리 설정
doc_dir = "docs"

# Streamlit UI
st.set_page_config(page_title="📚 문서 기반 GPT 챗봇", layout="wide")
st.title("📚 로컬 문서 기반 GPT 챗봇")

# 문서 자동 로드 함수
def load_documents():
    with st.spinner("📁 문서를 불러오는 중입니다..."):
        all_docs = []
        for file in os.listdir(doc_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(doc_dir, file))
                all_docs.extend(loader.load())
#            elif file.endswith(".pptx"):
#                loader = UnstructuredPowerPointLoader(os.path.join(doc_dir, file))
#                all_docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(all_docs)
        vectordb = FAISS.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=api_key))
        st.session_state["vectordb"] = vectordb
        st.session_state["ready"] = True

# 앱 실행 시 한 번만 문서 로드
if "ready" not in st.session_state:
    load_documents()

# 질문 UI
if st.session_state.get("ready"):
    question = st.text_input("💬 질문을 입력하세요:")
    if question:
        with st.spinner("🤖 GPT가 답변 중입니다..."):
            docs = st.session_state["vectordb"].similarity_search(question, k=5)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=question)
            st.markdown("### 🧠 답변")
            st.write(response)
else:
    st.info("🔄 문서를 불러오는 중입니다... 질문은 잠시만 기다려 주세요.")
