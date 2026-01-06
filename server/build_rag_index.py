"""
构建 RAG 知识库索引
从 Huatuo26M-Lite 数据集构建 FAISS 向量索引
"""

import json
import os
import sys
import logging
from pathlib import Path

# 设置 HuggingFace 镜像（在导入其他库之前）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if 'HF_HOME' not in os.environ:
    script_dir = Path(__file__).parent
    os.environ['HF_HOME'] = str(script_dir / 'models')
if 'SENTENCE_TRANSFORMERS_HOME' not in os.environ:
    script_dir = Path(__file__).parent
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(script_dir / 'models')

from tqdm import tqdm
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def build_rag_index(
    input_file: str = "data/Huatuo26M-Lite/format_data.jsonl",
    output_dir: str = "data/rag_index",
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    max_samples: int = None,  # 设置为 None 使用全部数据，或指定数量
    batch_size: int = 64,
    device: str = "cpu"
):
    """
    从 Huatuo 数据集构建 RAG 索引
    
    Args:
        input_file: JSONL 数据文件路径
        output_dir: 输出目录
        embedding_model: Embedding 模型名称
        max_samples: 最大样本数（用于测试）
        batch_size: 批处理大小
        device: 计算设备
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        logger.error("请安装依赖: pip install sentence-transformers faiss-cpu")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载 Embedding 模型
    logger.info(f"加载 Embedding 模型: {embedding_model}")
    # 设置缓存目录，优先使用本地已下载的模型
    script_dir = Path(__file__).parent
    cache_folder = str(script_dir / 'models')
    logger.info(f"使用模型缓存目录: {cache_folder}")
    logger.info(f"HuggingFace 镜像: {os.environ.get('HF_ENDPOINT', '未设置')}")
    
    try:
        model = SentenceTransformer(
            embedding_model, 
            device=device,
            cache_folder=cache_folder
        )
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.info("提示：请确保已设置 HuggingFace 镜像环境变量")
        logger.info("运行: source setup_mirrors.sh")
        raise
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding 维度: {embedding_dim}")
    
    # 2. 读取数据
    logger.info(f"读取数据文件: {input_file}")
    documents = []
    texts = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="读取数据")):
            if max_samples and i >= max_samples:
                break
            
            try:
                item = json.loads(line.strip())
                
                # 构建文档内容：问题 + 答案
                question = item.get('question', '')
                answer = item.get('answer', '')
                label = item.get('label', '')
                related_diseases = item.get('related_diseases', '')
                
                # 组合成检索文本
                content = f"问题：{question}\n答案：{answer}"
                
                documents.append({
                    'id': item.get('id', i),
                    'content': content,
                    'question': question,
                    'answer': answer,
                    'metadata': {
                        'label': label,
                        'related_diseases': related_diseases,
                        'score': item.get('score', 0),
                        'source': 'Huatuo26M-Lite'
                    }
                })
                texts.append(content)
                
            except json.JSONDecodeError:
                continue
    
    logger.info(f"加载了 {len(documents)} 条文档")
    
    if not documents:
        logger.error("没有加载到文档")
        return
    
    # 3. 生成 Embeddings（分批处理）
    logger.info(f"生成 Embeddings（批大小: {batch_size}）...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="生成向量"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # 4. 创建 FAISS 索引
    logger.info("创建 FAISS 索引...")
    
    # 使用 IndexFlatIP（内积，因为向量已归一化，等同于余弦相似度）
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    logger.info(f"索引包含 {index.ntotal} 个向量")
    
    # 5. 保存索引和文档
    index_path = output_path / "index.faiss"
    docs_path = output_path / "documents.json"
    
    logger.info(f"保存索引到: {index_path}")
    faiss.write_index(index, str(index_path))
    
    logger.info(f"保存文档到: {docs_path}")
    with open(docs_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    # 6. 保存元数据
    meta_path = output_path / "metadata.json"
    metadata = {
        'embedding_model': embedding_model,
        'embedding_dim': embedding_dim,
        'document_count': len(documents),
        'source': input_file,
        'index_type': 'IndexFlatIP'
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 50)
    logger.info("RAG 知识库构建完成！")
    logger.info(f"文档数量: {len(documents)}")
    logger.info(f"索引路径: {index_path}")
    logger.info(f"文档路径: {docs_path}")
    logger.info("=" * 50)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建 RAG 知识库索引")
    parser.add_argument("--input", default="Huatuo26M-Lite/format_data.jsonl", help="输入数据文件")
    parser.add_argument("--output", default="data/rag_index", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数（用于测试）")
    parser.add_argument("--batch-size", type=int, default=64, help="批处理大小")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    build_rag_index(
        input_file=args.input,
        output_dir=args.output,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device
    )
