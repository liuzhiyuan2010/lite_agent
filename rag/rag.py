import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置 HuggingFace 镜像加速 (如果在中国大陆)
import chromadb
#from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any
import uuid
import re

def parse_qa_pair(content: str) -> dict:
    if not content:
        return None

    # 定义正则模式
    # ^User:\s*      : 匹配开头的 "User:" 及其后的空白
    # (.*?)          : 非贪婪匹配用户内容 (Group 1)
    # \n             : 匹配中间的换行符 (作为分隔)
    # Assistant:\s*  : 匹配 "Assistant:" 及其后的空白
    # (.*)           : 匹配剩余的助理内容 (Group 2)，包括可能的多行内容
    # re.DOTALL      : 让 '.' 也能匹配换行符 (防止助理回答有多段话被截断)
    pattern = r"^user:\s*(.*?)\nassistant:\s*(.*)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        user_content = match.group(1).strip()
        assistant_content = match.group(2).strip()
        return [
            {'role': 'user', 'content': f'{user_content}'},
            {'role': 'assistant', 'content': f'{assistant_content}'}
        ]

    return None

class ChromaRAG:
    def __init__(self, db_path="./chroma_db", model_name="BAAI/bge-small-zh-v1.5"):
        self.db_path = db_path
        self.model_name = model_name
        # 1. 初始化 Chroma 客户端 (持久化模式)
        self.client = chromadb.PersistentClient(path=self.db_path)
        # 2. 获取或创建集合
        # metadata={"hnsw:space": "cosine"} 指定使用余弦相似度 (底层计算的是余弦距离)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",metadata={"hnsw:space": "cosine"}
        )

        # 3. 加载嵌入模型
        print(f"🚀 正在加载模型：{model_name} ...")
        self.model = SentenceTransformer( model_name, device='cpu',local_files_only=True )
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ 模型加载完成，维度：{self.dimension}")

    def add_documents(self, texts: List[str],
                      ids: Optional[List[str]] = None,
                      metadatas: Optional[List[Dict[str, Any]]] = None,
                      similarity_threshold: float = 0.9):
        """
        添加文档到向量库，带有去重功能。
        参数:
            texts: 列表，包含文档字符串
            ids: 列表，唯一标识符 (如果为 None，自动生成)
            metadatas: 列表，字典，包含额外信息
            similarity_threshold: 相似度阈值 (0~1)。如果新文档与库中已有文档相似度高于此值，则丢弃。
                                  默认 0.9 表示非常相似才丢弃。
        """
        if not texts:
            print("⚠️ 没有提供文档，跳过添加。")
            return

        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        # 如果用户传了 IDs，但长度不够，也补充 UUID (防止报错)
        elif len(ids) < len(texts):
            missing_count = len(texts) - len(ids)
            ids += [str(uuid.uuid4()) for _ in range(missing_count)]
            print(f"⚠️ 提供的 IDs 数量不足，已自动补充 {missing_count} 个随机 ID。")

        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas 的长度必须与 texts 长度一致")

        print(f"⏳ 正在处理 {len(texts)} 条文档 (相似度阈值：{similarity_threshold})...")

        # 批量生成所有待添加文档的嵌入向量
        all_embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()

        # 收集最终需要添加的数据
        final_texts = []
        final_ids = []
        final_metadatas = []
        final_embeddings = []
        skipped_count = 0

        # 逐个检查文档是否重复
        for i, (text, doc_id, embedding) in enumerate(zip(texts, ids, all_embeddings)):
            hits = self.search( query=text,top_k=1, min_similarity=similarity_threshold)

            if hits:
                # 找到了相似文档
                hit = hits[0]
                #print(f"  ⚠️ 跳过重复文档 [ID: {hit['id']}] (相似度: {hit['similarity']:.4f}): '{text[:30]}...'")
                skipped_count += 1
            else:
                # 没找到，可以添加

                final_texts.append(text)
                final_ids.append(doc_id)
                final_metadatas.append(metadatas[i] if metadatas else None)
                final_embeddings.append(embedding)

        # 2. 批量添加通过筛选的文档
        if final_texts:
                self.collection.add(
                    ids=final_ids,
                    embeddings=final_embeddings,
                    documents=final_texts,
                    metadatas=final_metadatas
                )
                print(f"✅ 成功添加 {len(final_texts)} 条新文档 (跳过了 {skipped_count} 条重复文档)。")
                # 打印一个示例 ID 让用户知道生成了什么
                #print(f"   示例生成的新 ID: {final_ids[0]}")
        else:
            print(f"✅ 没有新文档需要添加 (全部 {skipped_count} 条均为重复文档)。")


    def search(self, query, top_k=5,min_similarity = 0.6):
        query_embedding = self.model.encode([query], convert_to_numpy=True).tolist()
        results = self.collection.query(query_embeddings=query_embedding,n_results=top_k, include=["documents", "metadatas", "distances"] )
        hits = []
        if results['documents'] and results['documents'][0]:

            for i, doc in enumerate(results['documents'][0]):

                similarity = 1 - results['distances'][0][i]  # 转换为相似度
                # 【核心逻辑】：如果相似度低于阈值，直接丢弃该条结果
                # 注意：因为结果是按相似度从高到低排列的，
                # 一旦遇到一个低于阈值的，后面的肯定也都低于阈值，可以直接 break 提高效率
                if similarity < min_similarity:
                    #print(f"  ⚠️ 检索结果相似度 ({similarity:.4f}) 低于阈值 ({min_similarity})，已丢弃。")
                    break
                hits.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else None,
                    "similarity": similarity if results['distances'] else None,
                    "id": results['ids'][0][i]
                })
        return hits

    def clear(self):
        """清空数据库"""
        self.client.delete_collection("rag_collection")
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",metadata={"hnsw:space": "cosine"}
        )
        print("🧹 数据库已清空")


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    DB_FILE = "./my_chroma_db"
    MODEL = "BAAI/bge-small-zh-v1.5"

    # 初始化
    rag = ChromaRAG(db_path=DB_FILE, model_name=MODEL)

    # 1. 首次添加数据
    docs_batch_1 = [
        "ChromaDB 是一个轻量级的向量数据库，无需配置服务器。",
        "sqlite-vec 在 Windows 上可能会遇到扩展加载权限问题。",
        "Python 3.12 的 sqlite3 模块默认禁用了 load_extension。"
    ]
    print("\n--- 第一批数据添加 ---")
    rag.add_documents(docs_batch_1, similarity_threshold=0.85)

    # 2. 尝试添加包含重复内容的第二批数据
    docs_batch_2 = [
        "ChromaDB 是一个轻量级的向量数据库，无需配置服务器。",  # 与第一条几乎完全一样
        "使用 ONNX 后端可以加速 BGE 模型的推理速度。",  # 新内容
        "ChromaDB 是个很轻的向量库，不用配服务器。"  # 语义非常相似，应该被过滤
    ]
    print("\n--- 第二批数据添加 (包含重复) ---")
    rag.add_documents(docs_batch_2, similarity_threshold=0.85)

    # 3. 测试搜索
    query = "Windows 上 sqlite3 报错怎么办？"
    print(f"\n🔍 搜索查询：{query}")
    results = rag.search(query, top_k=3)

    for res in results:
        print(f"- [ID: {res['id']}, 距离: {res['similarity']:.4f}] {res['content']}")