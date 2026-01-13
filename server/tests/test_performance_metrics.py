"""
性能指标测试脚本
测试 RAG 检索延迟、知识图谱查询、LLM 推理时间
"""

import time
import sys
import os

# 获取 server 目录和项目根目录的绝对路径
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(SERVER_DIR)  # voice_assistant 根目录
sys.path.insert(0, SERVER_DIR)
os.chdir(SERVER_DIR)  # 切换工作目录到 server

from typing import Dict, List
import statistics


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.results = {
            'rag_retrieval': [],
            'knowledge_graph': [],
            'llm_inference': []
        }
    
    def measure_time(self, func, *args, **kwargs):
        """测量函数执行时间（毫秒）"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
        return result, elapsed
    
    def add_result(self, metric_name: str, value: float):
        """添加测试结果"""
        if metric_name in self.results:
            self.results[metric_name].append(value)
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        summary = {}
        for name, values in self.results.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
        return summary
    
    def print_report(self):
        """打印性能报告"""
        print("\n" + "=" * 60)
        print("📊 性能指标测试报告")
        print("=" * 60)
        
        summary = self.get_summary()
        
        metrics_info = {
            'rag_retrieval': ('RAG 检索延迟', '<100ms', 'FAISS 向量检索'),
            'knowledge_graph': ('知识图谱查询', '<200ms', 'Neo4j Cypher 执行'),
            'llm_inference': ('LLM 推理时间', '~1.5s', '512 token 生成')
        }
        
        print(f"\n{'指标':<20} | {'平均值':>10} | {'最小值':>10} | {'最大值':>10} | {'目标':>10} | {'状态':>6}")
        print("-" * 80)
        
        for name, info in metrics_info.items():
            if name in summary:
                s = summary[name]
                avg = s['avg']
                
                # 判断是否达标
                if name == 'rag_retrieval':
                    passed = avg < 100
                elif name == 'knowledge_graph':
                    passed = avg < 200
                else:  # llm_inference
                    passed = avg < 2000  # 2秒内
                
                status = "✅ 达标" if passed else "❌ 超时"
                
                if name == 'llm_inference':
                    print(f"{info[0]:<18} | {avg/1000:>8.2f}s | {s['min']/1000:>8.2f}s | {s['max']/1000:>8.2f}s | {info[1]:>10} | {status}")
                else:
                    print(f"{info[0]:<18} | {avg:>8.2f}ms | {s['min']:>8.2f}ms | {s['max']:>8.2f}ms | {info[1]:>10} | {status}")
            else:
                print(f"{info[0]:<18} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {info[1]:>10} | ⚠️ 未测试")
        
        print("=" * 60)


def test_rag_retrieval(metrics: PerformanceMetrics, rag_module, queries: List[str], n_runs: int = 5):
    """测试 RAG 检索延迟"""
    print("\n🔍 测试 RAG 检索延迟...")
    
    for i, query in enumerate(queries):
        for run in range(n_runs):
            _, elapsed = metrics.measure_time(rag_module.retrieve, query)
            metrics.add_result('rag_retrieval', elapsed)
            print(f"  Query {i+1}, Run {run+1}: {elapsed:.2f}ms")


def test_knowledge_graph(metrics: PerformanceMetrics, kg_module, queries: List[str], n_runs: int = 5):
    """测试知识图谱查询延迟"""
    print("\n🌐 测试知识图谱查询...")
    
    for i, query in enumerate(queries):
        for run in range(n_runs):
            _, elapsed = metrics.measure_time(kg_module.smart_query, query)
            metrics.add_result('knowledge_graph', elapsed)
            print(f"  Query {i+1}, Run {run+1}: {elapsed:.2f}ms")


def test_llm_inference(metrics: PerformanceMetrics, dialogue_module, queries: List[str], n_runs: int = 3):
    """测试 LLM 推理时间"""
    print("\n🧠 测试 LLM 推理时间...")
    
    for i, query in enumerate(queries):
        for run in range(n_runs):
            _, elapsed = metrics.measure_time(
                dialogue_module.chat, 
                query=query, 
                session_id="perf_test",
                use_rag=False
            )
            metrics.add_result('llm_inference', elapsed)
            print(f"  Query {i+1}, Run {run+1}: {elapsed/1000:.2f}s")


def main():
    """主测试函数"""
    print("🚀 NLP 模块性能测试")
    print("=" * 40)
    
    metrics = PerformanceMetrics()
    
    # 测试查询
    test_queries = [
        "感冒有什么症状",
        "高血压吃什么药",
        "头痛应该挂什么科"
    ]
    
    # ========================================
    # 1. 测试 RAG 模块
    # ========================================
    try:
        from modules.core.rag import RAGModule
        import yaml
        
        with open(os.path.join(PROJECT_DIR, 'config/config.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        rag_config = config.get('rag', {})
        rag = RAGModule(
            embedding_model=rag_config.get('embedding_model', 'BAAI/bge-small-zh-v1.5'),
            index_path=rag_config.get('index_path', 'data/rag_index'),
            device=rag_config.get('device', 'cpu'),
            top_k=rag_config.get('top_k', 3),
            min_score=rag_config.get('min_score', 0.5)
        )
        
        # 检查 RAG 索引是否可用
        if rag.index is None or (hasattr(rag.index, 'ntotal') and rag.index.ntotal == 0):
            print("⚠️ RAG 模块测试跳过: FAISS 索引未构建（需要先运行 build_rag_index.py）")
        else:
            print(f"📚 RAG 索引已加载: {rag.index.ntotal} 条文档")
            test_rag_retrieval(metrics, rag, test_queries)
        
    except Exception as e:
        print(f"⚠️ RAG 模块测试跳过: {e}")
    
    # ========================================
    # 2. 测试知识图谱模块
    # ========================================
    try:
        from modules.knowledge.knowledge_graph import KnowledgeGraphModule
        
        kg = KnowledgeGraphModule(
            host="localhost",
            port=7474,
            user="neo4j",
            password="12345"
        )
        if kg.enabled:
            test_knowledge_graph(metrics, kg, test_queries)
        else:
            print("⚠️ 知识图谱模块测试跳过: Neo4j 未连接")
        
    except Exception as e:
        print(f"⚠️ 知识图谱模块测试跳过: {e}")
    
    # ========================================
    # 3. 测试 LLM 模块 (使用 0.5B 模型)
    # ========================================
    try:
        from modules.core.dialogue import DialogueModule
        import yaml
        
        with open(os.path.join(PROJECT_DIR, 'config/config.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        dialogue_cfg = config.get('dialogue', {})
        device = dialogue_cfg.get('device', 'cuda')  # 默认使用 cuda
        model_name = dialogue_cfg.get('model', 'Qwen/Qwen2.5-0.5B-Instruct')
        
        print(f"\n⏳ 正在加载 {model_name}，设备: {device}...")
        # DialogueModule 需要传入 model 和 device
        dialogue = DialogueModule(
            model_name=model_name,
            device=device,
            max_length=dialogue_cfg.get('max_length', 512),
            temperature=dialogue_cfg.get('temperature', 0.7)
        )
        print(f"✅ 模型已加载到: {dialogue.device}")
        test_llm_inference(metrics, dialogue, test_queries, n_runs=2)
        
    except Exception as e:
        import traceback
        print(f"⚠️ LLM 模块测试跳过: {e}")
        traceback.print_exc()
    
    # ========================================
    # 打印报告
    # ========================================
    metrics.print_report()
    
    # 保存结果到文件
    import json
    report_path = 'tests/performance_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': metrics.get_summary(),
            'raw_results': metrics.results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()

