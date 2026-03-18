
# install 
```commandline
pip install sentence-transformers
pip install chromadb
```
# example
## python demo
```commandline
if __name__ == "__main__":
    DB_FILE = "./my_chroma_db"  # 数据会存在这个文件夹里
    MODEL = "BAAI/bge-small-zh-v1.5"

    # 初始化
    rag = ChromaRAG(db_path=DB_FILE, model_name=MODEL)

    # 测试添加数据
    docs = [
        "ChromaDB 是一个轻量级的向量数据库，无需配置服务器。",
        "sqlite-vec 在 Windows 上可能会遇到扩展加载权限问题。",
        "Python 3.12 的 sqlite3 模块默认禁用了 load_extension。",
        "使用 ONNX 后端可以加速 BGE 模型的推理速度。"
    ]
    # 添加一些元数据
    metas = [{"source": "test.txt"} for _ in docs]

    rag.add_documents(docs, metadatas=metas)

    # 测试搜索
    query = "Windows 上 sqlite3 报错怎么办？"
    print(f"\n🔍 搜索查询：{query}")
    results = rag.search(query, top_k=2)

    for res in results:
        print(f"- [距离: {res['distance']:.4f}] {res['content']}")
        if res['metadata']:
            print(f"  来源：{res['metadata'].get('source')}")

```