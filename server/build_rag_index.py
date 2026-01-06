"""
RAG (Retrieval-Augmented Generation) 模块
使用 FAISS 向量检索 + 医疗知识库
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 尝试导入 RAG 相关库
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
    logger.info("RAG dependencies loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"RAG dependencies not available: {e}")


class RAGModule:
    """RAG 检索增强生成模块"""
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 index_path: str = "data/rag_index",
                 knowledge_base_path: str = "data/knowledge_base.json",
                 device: str = "cpu",
                 top_k: int = 3,
                 min_score: float = 0.5):
        """
        初始化 RAG 模块
        
        Args:
            embedding_model: Embedding 模型名称
            index_path: FAISS 索引保存路径
            knowledge_base_path: 知识库 JSON 文件路径
            device: 运行设备
            top_k: 默认检索数量
            min_score: 相似度阈值，低于此值的结果将被过滤
        """
        self.top_k = top_k
        self.min_score = min_score
        self.index_path = Path(index_path)
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents = []  # 存储原始文档
        self.index = None    # FAISS 索引
        self.model = None    # Embedding 模型
        self.knowledge_graph = None  # 知识图谱模块
        
        if not RAG_AVAILABLE:
            logger.warning("RAG module initialized but dependencies not available")
            return
        
        try:
            # 加载 Embedding 模型
            logger.info(f"Loading embedding model: {embedding_model}")
            
            # 优先使用 ModelScope 下载模型
            if 'BAAI' in embedding_model or 'bge' in embedding_model.lower():
                try:
                    from modelscope import snapshot_download
                    # 从 ModelScope 下载 bge 模型
                    models_dir = Path(__file__).parent.parent / "models" / "embedding"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    ms_model_name = "AI-ModelScope/bge-small-zh-v1.5"
                    logger.info(f"Downloading {ms_model_name} from ModelScope...")
                    model_path = snapshot_download(ms_model_name, cache_dir=str(models_dir))
                    logger.info(f"Embedding model downloaded to: {model_path}")
                    self.model = SentenceTransformer(model_path, device=device)
                except ImportError:
                    logger.warning("modelscope not available, falling back to HuggingFace")
                    self.model = SentenceTransformer(embedding_model, device=device)
                except Exception as e:
                    logger.warning(f"ModelScope download failed: {e}, falling back to HuggingFace")
                    self.model = SentenceTransformer(embedding_model, device=device)
            else:
                self.model = SentenceTransformer(embedding_model, device=device)
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded, dimension: {self.embedding_dim}")
            
            # 尝试加载已有索引，否则从知识库构建
            if self._load_index():
                logger.info("Loaded existing FAISS index")
            elif self.knowledge_base_path.exists():
                self._build_index_from_knowledge_base()
            else:
                logger.warning(f"Knowledge base not found: {self.knowledge_base_path}")
                self._create_empty_index()
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG module: {e}")
            self.model = None
    
    def _create_empty_index(self):
        """创建空的 FAISS 索引"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积（余弦相似度）
        self.documents = []
        
    def _load_index(self) -> bool:
        """加载已保存的 FAISS 索引"""
        index_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "documents.json"
        
        if index_file.exists() and docs_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                
                print("\n" + "="*50)
                print(f"📚 [RAG] 成功加载索引")
                print(f"   - 文档数量: {len(self.documents)}")
                print(f"   - 向量数量: {self.index.ntotal}")
                print("="*50 + "\n")
                
                return True
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
        return False
    
    def _save_index(self):
        """保存 FAISS 索引"""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        index_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "documents.json"
        
        faiss.write_index(self.index, str(index_file))
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Index saved to {self.index_path}")
    
    def _build_index_from_knowledge_base(self):
        """从知识库文件构建索引"""
        logger.info(f"Building index from {self.knowledge_base_path}")
        
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
        
        # 提取文档内容
        self.documents = []
        texts = []
        
        for item in knowledge_base:
            content = item.get('content', '')
            if content:
                self.documents.append({
                    'id': item.get('id', len(self.documents)),
                    'content': content,
                    'metadata': item.get('metadata', {})
                })
                texts.append(content)
        
        if not texts:
            logger.warning("No documents found in knowledge base")
            self._create_empty_index()
            return
        
        # 生成 embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # 创建 FAISS 索引
        self._create_empty_index()
        self.index.add(embeddings.astype(np.float32))
        
        # 保存索引
        self._save_index()
        logger.info(f"Index built with {len(self.documents)} documents")
    
    def add_documents(self, documents: List[Dict]):
        """
        添加文档到索引
        
        Args:
            documents: 文档列表，每个文档包含 'content' 字段
        """
        if not self.model:
            logger.error("RAG model not initialized")
            return
        
        texts = []
        for doc in documents:
            content = doc.get('content', '')
            if content:
                self.documents.append({
                    'id': doc.get('id', len(self.documents)),
                    'content': content,
                    'metadata': doc.get('metadata', {})
                })
                texts.append(content)
        
        if texts:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            self.index.add(embeddings.astype(np.float32))
            self._save_index()
            logger.info(f"Added {len(texts)} documents to index")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            
        Returns:
            检索到的文档列表
        """
        if not self.model or not self.index or self.index.ntotal == 0:
            return []
        
        top_k = top_k or self.top_k
        top_k = min(top_k, self.index.ntotal)
        
        # 生成查询向量
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # 检索
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # 打印检索信息到终端 (RAG 调试信息)
        print("\n" + "="*50)
        print(f"🔍 [RAG 检索] 查询: {query}")
        print(f"   相似度阈值: {self.min_score}")
        print("-" * 50)
        
        # 返回结果（应用相似度阈值过滤）
        results = []
        filtered_count = 0
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                # 检查是否满足相似度阈值
                if score < self.min_score:
                    filtered_count += 1
                    if i < 3:
                        content_preview = self.documents[idx]['content'].replace('\n', ' ')[:60]
                        label = self.documents[idx].get('metadata', {}).get('label', '未知')
                        print(f"  [✗] (相似度: {score:.3f} < {self.min_score}) [{label}]")
                        print(f"      {content_preview}... (已过滤)")
                    continue
                
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
                
                # 打印前3条有效检索结果
                if len(results) <= 3:
                    content_preview = doc['content'].replace('\n', ' ')[:100]
                    label = doc.get('metadata', {}).get('label', '未知')
                    print(f"  [✓] (相似度: {score:.3f}) [{label}]")
                    print(f"      {content_preview}...")
        
        if filtered_count > 0:
            print(f"\n   ⚠ 已过滤 {filtered_count} 条低相似度结果")
        if not results:
            print("   📭 无有效检索结果（所有结果相似度低于阈值）")
        
        print("="*50 + "\n")
        
        return results
    
    def build_context(self, query: str, top_k: int = None) -> str:
        """
        构建 RAG 上下文（向量检索 + 知识图谱）
        
        Args:
            query: 用户查询
            top_k: 检索数量
            
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 1. 向量检索
        retrieved = self.retrieve(query, top_k)
        if retrieved:
            context_parts.append("【向量检索结果】")
            for i, doc in enumerate(retrieved[:3], 1):
                context_parts.append(f"参考{i}：{doc['content']}")
        
        # 2. 知识图谱补充
        if self.knowledge_graph and self.knowledge_graph.enabled:
            kg_context = self.knowledge_graph.build_context_from_query(query)
            if kg_context:
                context_parts.append("\n【知识图谱补充】")
                context_parts.append(kg_context)
        
        return "\n".join(context_parts)
    
    def get_info(self) -> Dict:
        """获取模块信息"""
        return {
            "available": RAG_AVAILABLE and self.model is not None,
            "document_count": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim if self.model else None,
            "top_k": self.top_k
        }


# 简化版 RAG（用于测试或依赖不可用时）
class SimpleRAGModule:
    """简化的 RAG 模块（基于关键词匹配）"""
    
    def __init__(self, knowledge_base_path: str = "data/knowledge_base.json"):
        self.documents = []
        self.knowledge_base_path = Path(knowledge_base_path)
        
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
                self.documents = [item.get('content', '') for item in kb if item.get('content')]
                logger.info(f"SimpleRAG loaded {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """基于关键词的简单检索"""
        results = []
        
        for i, doc in enumerate(self.documents):
            # 计算关键词匹配得分
            score = sum(1 for char in query if char in doc)
            if score > 0:
                results.append({
                    'id': i,
                    'content': doc,
                    'score': score
                })
        
        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def build_context(self, query: str, top_k: int = 3) -> str:
        """构建上下文"""
        retrieved = self.retrieve(query, top_k)
        
        if not retrieved:
            return ""
        
        context_parts = [f"参考{i}：{doc['content']}" for i, doc in enumerate(retrieved, 1)]
        return "\n".join(context_parts)
    
    def get_info(self) -> Dict:
        return {
            "available": True,
            "document_count": len(self.documents),
            "type": "simple_keyword_matching"
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试 RAG 模块
    rag = RAGModule()
    
    print(f"RAG Info: {rag.get_info()}")
    
    # 测试检索
    query = "感冒怎么办"
    results = rag.retrieve(query)
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  - {r['content'][:50]}... (score: {r['score']:.3f})")