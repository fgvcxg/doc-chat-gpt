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

        # ✅ 문서가 실제로 업데이트되었을 경우만 목록 출력
        if need_reload:
            st.subheader("📄 업데이트된 문서 목록")
            for name, ts in modified_files:
                if ts > last_saved_time:
                    download_link = os.path.join(doc_dir, name)
                    st.markdown(f"- [{name}]({download_link}) (수정: {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')})")
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

# ✅ 문서 준비 상태 표시
if st.session_state["ready"]:
    st.success("✅ 문서 로드 완료! 지금 바로 질문해보세요.")
else:
    st.warning("⏳ 문서를 불러오는 중입니다. 잠시만 기다려 주세요...")
