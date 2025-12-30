"""
模型下载脚本
下载项目所需的所有模型到对应文件夹
如果模型已存在则跳过
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 模型配置
MODELS_DIR = Path(__file__).parent / "models"

# 需要下载的模型列表
MODELS = {
    # 对话模型 (Qwen3-4B from ModelScope)
    "dialogue": {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "source": "modelscope",
        "path": MODELS_DIR / "Qwen" / "Qwen3-4B-Instruct-2507",
        "description": "对话生成模型 (4B参数)"
    },
    # ASR模型 (FunASR Paraformer)
    "asr": {
        "name": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "source": "modelscope",
        "path": MODELS_DIR / "models" / "iic" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "description": "语音识别模型 (Paraformer)"
    },
    # VAD模型
    "vad": {
        "name": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "source": "modelscope",
        "path": MODELS_DIR / "models" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "description": "语音活动检测模型"
    },
    # 标点模型
    "punc": {
        "name": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "source": "modelscope",
        "path": MODELS_DIR / "models" / "iic" / "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "description": "标点恢复模型"
    },
    # 声纹识别模型
    "speaker": {
        "name": "iic/speech_campplus_sv_zh-cn_16k-common",
        "source": "modelscope",
        "path": MODELS_DIR / "models" / "iic" / "speech_campplus_sv_zh-cn_16k-common",
        "description": "声纹识别模型"
    },
    # SenseVoice (可选)
    "sensevoice": {
        "name": "iic/SenseVoiceSmall",
        "source": "modelscope",
        "path": MODELS_DIR / "models" / "iic" / "SenseVoiceSmall",
        "description": "多语言语音识别模型"
    },
    # Embedding模型 (for RAG)
    "embedding": {
        "name": "BAAI/bge-small-zh-v1.5",
        "source": "huggingface",
        "path": None,  # HuggingFace会自动缓存
        "description": "RAG向量嵌入模型"
    }
}


def check_model_exists(model_config):
    """检查模型是否已存在"""
    path = model_config.get("path")
    if path is None:
        return False
    
    if path.exists():
        # 检查目录是否有内容
        files = list(path.iterdir()) if path.is_dir() else []
        return len(files) > 0
    return False


def download_from_modelscope(model_name, cache_dir):
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        
        path = snapshot_download(model_name, cache_dir=str(cache_dir))
        return path
    except Exception as e:
        logger.error(f"ModelScope 下载失败: {e}")
        return None


def download_from_huggingface(model_name):
    """从 HuggingFace 下载模型"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # 对于 embedding 模型使用 sentence_transformers
        if "bge" in model_name.lower():
            model = SentenceTransformer(model_name)
            return True
        
        # 其他模型使用 transformers
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        return True
        
    except Exception as e:
        logger.error(f"HuggingFace 下载失败: {e}")
        return None


def download_model(key, config):
    """下载单个模型"""
    name = config["name"]
    source = config["source"]
    description = config["description"]
    
    print(f"\n{'='*50}")
    print(f"📦 {description}")
    print(f"   模型: {name}")
    print(f"   来源: {source}")
    print(f"{'='*50}")
    
    # 检查是否已存在
    if check_model_exists(config):
        logger.info(f"✓ 模型已存在，跳过下载")
        return "已存在"
    
    # 下载模型
    logger.info(f"开始下载...")
    
    if source == "modelscope":
        result = download_from_modelscope(name, MODELS_DIR)
    elif source == "huggingface":
        result = download_from_huggingface(name)
    else:
        logger.error(f"未知的模型来源: {source}")
        return "失败"
    
    if result:
        logger.info(f"✓ 下载完成")
        return "成功"
    else:
        return "失败"


def setup_directories():
    """创建必要的目录"""
    directories = [
        MODELS_DIR,
        MODELS_DIR / "Qwen",
        MODELS_DIR / "models" / "iic",
        Path(__file__).parent.parent / "data",
        Path(__file__).parent.parent / "data" / "rag_index",
        Path(__file__).parent / "logs",
        Path(__file__).parent / "temp",
    ]
    
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
    
    logger.info("目录结构已创建")


def main():
    """主函数"""
    print()
    print("="*60)
    print("  🏥 医疗语音助手 - 模型下载工具")
    print("="*60)
    print()
    
    # 创建目录
    setup_directories()
    
    # 选择要下载的模型
    print("可用模型:")
    for i, (key, config) in enumerate(MODELS.items(), 1):
        status = "✓" if check_model_exists(config) else "○"
        print(f"  {i}. [{status}] {config['description']}")
    
    print()
    print("选项:")
    print("  a - 下载所有模型")
    print("  s - 只下载缺失的模型")
    print("  q - 退出")
    print()
    
    choice = input("请选择 [s]: ").strip().lower() or "s"
    
    if choice == "q":
        print("已取消")
        return
    
    # 确定要下载的模型
    if choice == "a":
        models_to_download = MODELS
    else:  # 默认只下载缺失的
        models_to_download = {
            k: v for k, v in MODELS.items() 
            if not check_model_exists(v)
        }
    
    if not models_to_download:
        print("\n✓ 所有模型已就绪！")
        return
    
    print(f"\n将下载 {len(models_to_download)} 个模型...")
    
    # 下载模型
    results = {}
    for key, config in models_to_download.items():
        results[key] = download_model(key, config)
    
    # 显示结果
    print("\n" + "="*60)
    print("下载结果:")
    print("="*60)
    
    for key, status in results.items():
        icon = "✓" if status in ["成功", "已存在"] else "✗"
        print(f"  {icon} {MODELS[key]['description']}: {status}")
    
    print("="*60)
    
    # 检查关键模型
    critical = ["dialogue", "asr"]
    all_ok = all(
        results.get(k, "失败") in ["成功", "已存在"] or check_model_exists(MODELS[k])
        for k in critical
    )
    
    if all_ok:
        print("\n✓ 核心模型已就绪，可以启动服务器！")
        print("  运行: ./start_server.sh")
    else:
        print("\n⚠ 部分关键模型未就绪，请检查错误信息")
    
    print()


if __name__ == "__main__":
    main()
