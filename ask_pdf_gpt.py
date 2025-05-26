import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import getpass

# GPT API Key 입력 받기 (처음 1회만)
openai_api_key = input("OpenAI API Key 입력 (복붙 가능): ")

# 1. PDF 경로 설정
pdf_dir = "D:/toyproj"
all_docs = []

# 2. PDF 파일 로드 및 분할
print("📄 PDF 불러오는 중...")
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, filename))
        docs = loader.load()
        all_docs.extend(docs)

# 3. 텍스트 분할
print("🔍 문서 분할 및 임베딩 중...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(all_docs)

# 4. Chroma 벡터 DB 생성
db = Chroma.from_documents(split_docs, OpenAIEmbeddings(openai_api_key=openai_api_key))

# 5. 질의 응답 체인 생성
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    retriever=db.as_retriever(),
    return_source_documents=True
)

# 6. 질문 루프
print("\n✅ 설정 완료! PDF에 대해 질문하세요. (종료하려면 'exit')")
while True:
    query = input("\n질문: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = qa({"query": query})
    print("\n📌 답변:")
    print(result["result"])

    print("\n🔎 출처 문서 일부:")
    for doc in result["source_documents"]:
        print(f"→ {doc.metadata['source']}")
