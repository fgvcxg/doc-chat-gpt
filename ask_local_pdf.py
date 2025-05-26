import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import FakeEmbeddings

# 1. 문서 로딩
pdf_dir = "D:/toyproj"
all_docs = []

print("📄 PDF 문서 불러오는 중...")
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, filename))
        docs = loader.load()
        all_docs.extend(docs)

# 2. 텍스트 분할
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(all_docs)

# 3. 벡터 DB에 저장 (FakeEmbeddings 사용)
print("🔍 벡터 인덱싱 중... (GPT 없이 작동)")
embedding = FakeEmbeddings(size=768)  # 실제 임베딩은 하지 않지만 구조만 유지
db = Chroma.from_documents(split_docs, embedding)

retriever = db.as_retriever()

# 4. 검색 루프 시작
print("\n✅ 설정 완료! PDF 문서 내용 검색 테스트 시작 (종료하려면 'exit')")
while True:
    query = input("\n질문 키워드: ")
    if query.lower() in ["exit", "quit"]:
        break

    results = retriever.get_relevant_documents(query)
    
    if not results:
        print("❌ 관련된 문서를 찾을 수 없습니다.")
    else:
        print(f"\n🔎 관련된 문서 조각 {len(results)}개:")
        for i, doc in enumerate(results):
            print(f"\n--- 문서 {i+1} ---")
            print(doc.page_content.strip()[:500] + "...")
