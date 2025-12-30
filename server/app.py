"""
语音助手服务器主应用
提供RESTful API接口
"""

import os
import sys
import warnings
import logging

# 抑制第三方库的详细输出（必须在导入其他库之前设置）
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 抑制tokenizers警告
os.environ['FUNASR_DISABLE_TQDM'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'  # 40 = logging.ERROR

# 设置所有第三方库日志级别为ERROR
for logger_name in ['modelscope', 'funasr', 'transformers', 'torch', 
                    'urllib3', 'filelock', 'tqdm', 'httpx', 'httpcore']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# 抑制root logger的WARNING
logging.getLogger().setLevel(logging.ERROR)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yaml
from pathlib import Path
import tempfile
from datetime import datetime

# 设置模型缓存目录到项目的 models 文件夹
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE_DIR
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
# 设置 HuggingFace 镜像源（如果未设置）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入各个模块
from modules.asr import ASRModule
from modules.emotion import EmotionModule
from modules.speaker import SpeakerModule
from modules.dialogue import DialogueModule, SimplDialogueModule
from modules.tts import TTSModule, SimpleTTSModule
from modules.rag import RAGModule, SimpleRAGModule
from modules.triage import TriageModule
from modules.diagnosis_assistant import DiagnosisAssistant
from modules.medication import MedicationModule

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/voice_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 全局变量存储模块实例
modules = {}
config = {}


def load_config(config_path: str = "config/config.yaml"):
    """加载配置文件"""
    try:
        # 尝试多个可能的配置文件路径
        possible_paths = [
            config_path,  # 当前目录
            os.path.join('..', config_path),  # 上级目录
            os.path.join(os.path.dirname(__file__), '..', config_path),  # 相对于脚本位置
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    logger.info(f"Loaded config from: {path}")
                    return yaml.safe_load(f)
        
        logger.warning(f"Config file not found in any of: {possible_paths}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


def initialize_modules():
    """初始化所有模块"""
    global modules, config
    
    # 加载配置
    config = load_config()
    
    # 创建必要的目录
    for path_key in ['models', 'data', 'logs', 'temp']:
        path = Path(config.get('paths', {}).get(path_key, path_key))
        path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing modules...")
    
    try:
        # 初始化ASR模块 (Paraformer-Large)
        asr_config = config.get('asr', {})
        modules['asr'] = ASRModule(
            model_name=asr_config.get('model', 'paraformer-large'),
            device=asr_config.get('device', 'cpu'),
            language=asr_config.get('language', 'zh')
        )
        logger.info("ASR module initialized (Paraformer-Large)")
        
    except Exception as e:
        logger.error(f"Failed to initialize ASR: {e}")
        modules['asr'] = None
    
    try:
        # 初始化情感识别模块 (SenseVoice)
        emotion_config = config.get('emotion', {})
        modules['emotion'] = EmotionModule(
            model_path=emotion_config.get('model'),
            device=emotion_config.get('device', 'cpu')
        )
        logger.info("Emotion module initialized (SenseVoice)")
        
    except Exception as e:
        logger.error(f"Failed to initialize Emotion: {e}")
        modules['emotion'] = None
    
    try:
        # 初始化声纹识别模块 (Cam++)
        speaker_config = config.get('speaker', {})
        modules['speaker'] = SpeakerModule(
            db_path=speaker_config.get('db_path', 'data/speaker_db.pkl'),
            threshold=speaker_config.get('threshold', 0.75),
            device=speaker_config.get('device', 'cpu')
        )
        logger.info("Speaker module initialized (Cam++)")
        
    except Exception as e:
        logger.error(f"Failed to initialize Speaker: {e}")
        modules['speaker'] = None
    
    # 初始化 RAG 模块（如果启用）
    rag_module = None
    rag_config = config.get('rag', {})
    if rag_config.get('enabled', False):
        # 处理相对路径（相对于脚本位置的上级目录）
        kb_path = rag_config.get('knowledge_base', 'config/knowledge_base.json')
        if not os.path.isabs(kb_path):
            kb_path = os.path.join(os.path.dirname(__file__), '..', kb_path)
        
        index_path = rag_config.get('index_path', 'data/rag_index')
        if not os.path.isabs(index_path):
            index_path = os.path.join(os.path.dirname(__file__), index_path)
        
        try:
            rag_module = RAGModule(
                embedding_model=rag_config.get('embedding_model', 'BAAI/bge-small-zh-v1.5'),
                index_path=index_path,
                knowledge_base_path=kb_path,
                device=rag_config.get('device', 'cpu'),
                top_k=rag_config.get('top_k', 3)
            )
            modules['rag'] = rag_module
            logger.info(f"RAG module initialized with {rag_module.get_info().get('document_count', 0)} documents")
            
            # 初始化知识图谱模块并与 RAG 集成
            try:
                from modules.knowledge_graph import KnowledgeGraphModule
                kg_config = config.get('knowledge_graph', {})
                kg_module = KnowledgeGraphModule(
                    host=kg_config.get('host', '172.24.30.243'),
                    port=kg_config.get('port', 7474),
                    user=kg_config.get('user', 'neo4j'),
                    password=kg_config.get('password', '12345')
                )
                if kg_module.enabled:
                    rag_module.knowledge_graph = kg_module
                    modules['knowledge_graph'] = kg_module
                    logger.info("Knowledge Graph integrated with RAG")
                else:
                    logger.warning("Knowledge Graph not available, RAG will work without it")
            except Exception as kg_e:
                logger.warning(f"Knowledge Graph initialization failed: {kg_e}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize RAG, trying SimpleRAG: {e}")
            try:
                rag_module = SimpleRAGModule(
                    knowledge_base_path=kb_path
                )
                modules['rag'] = rag_module
                logger.info("SimpleRAG module initialized")
            except Exception as inner_e:
                logger.error(f"Failed to initialize any RAG module: {inner_e}")
                rag_module = None
    
    try:
        # 初始化对话模块 (Qwen2.5-0.5B-Instruct)
        dialogue_config = config.get('dialogue', {})
        try:
            modules['dialogue'] = DialogueModule(
                model_name=dialogue_config.get('model', 'Qwen/Qwen2.5-0.5B-Instruct'),
                device=dialogue_config.get('device', 'cpu'),
                max_length=dialogue_config.get('max_length', 512),
                temperature=dialogue_config.get('temperature', 0.7),
                top_p=dialogue_config.get('top_p', 0.9),
                history_length=dialogue_config.get('history_length', 10),
                rag_module=rag_module  # 传入 RAG 模块
            )
            logger.info("Dialogue module initialized (Qwen2.5-0.5B)" + (" with RAG" if rag_module else ""))
        except Exception as inner_e:
            logger.warning(f"Using simplified dialogue module: {inner_e}")
            modules['dialogue'] = SimplDialogueModule()
        
    except Exception as e:
        logger.error(f"Failed to initialize Dialogue: {e}")
        modules['dialogue'] = SimplDialogueModule()
    
    try:
        # 初始化TTS模块
        tts_config = config.get('tts', {})
        
        # 如果配置了 fast_mode，使用 macOS say 命令（即时响应）
        if tts_config.get('fast_mode', False):
            logger.info("TTS fast_mode enabled, using macOS say")
            modules['tts'] = SimpleTTSModule()
        else:
            try:
                tts_module = TTSModule(
                    model_name=tts_config.get('model'),
                    device=tts_config.get('device', 'cpu')
                )
                # 检查模型是否真正可用
                if tts_module.model is not None:
                    modules['tts'] = tts_module
                    logger.info("TTS module initialized (CosyVoice)")
                else:
                    raise RuntimeError("CosyVoice model not available")
            except Exception as e:
                logger.warning(f"Using simplified TTS module (macOS say): {e}")
                modules['tts'] = SimpleTTSModule()
        
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        modules['tts'] = SimpleTTSModule()
    
    # 初始化医疗模块（传入 RAG 和对话模块以支持智能诊断）
    try:
        knowledge_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'medical_knowledge.json')
        
        # 患者导诊模块（支持 RAG + LLM）
        modules['triage'] = TriageModule(
            knowledge_path=knowledge_path,
            rag_module=rag_module,
            dialogue_module=modules.get('dialogue')
        )
        logger.info("Triage module initialized" + (" with RAG+LLM" if rag_module else ""))
        
        # 医生辅助诊断模块（支持 RAG + LLM）
        modules['diagnosis'] = DiagnosisAssistant(
            knowledge_path=knowledge_path,
            rag_module=rag_module,
            dialogue_module=modules.get('dialogue')
        )
        logger.info("Diagnosis Assistant module initialized" + (" with RAG+LLM" if rag_module else ""))
        
        # 用药查询模块
        modules['medication'] = MedicationModule(knowledge_path=knowledge_path)
        logger.info("Medication module initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize medical modules: {e}")
        modules['triage'] = None
        modules['diagnosis'] = None
        modules['medication'] = None
    
    logger.info("All modules initialized successfully")


# ============= API路由 =============

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            name: module is not None
            for name, module in modules.items()
        }
    })


@app.route('/asr', methods=['POST'])
def asr_endpoint():
    """语音识别接口"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # 保存临时文件
        temp_path = Path(tempfile.mktemp(suffix='.wav'))
        audio_file.save(temp_path)
        
        # 执行识别
        result = modules['asr'].transcribe(audio_path=str(temp_path))
        
        # 删除临时文件
        temp_path.unlink()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"ASR endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/emotion', methods=['POST'])
def emotion_endpoint():
    """情感识别接口"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # 保存临时文件
        temp_path = Path(tempfile.mktemp(suffix='.wav'))
        audio_file.save(temp_path)
        
        # 执行情感识别
        result = modules['emotion'].predict(audio_path=str(temp_path))
        
        # 删除临时文件
        temp_path.unlink()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Emotion endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/speaker/register', methods=['POST'])
def speaker_register():
    """声纹注册接口"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        speaker_id = request.form.get('speaker_id')
        
        if not speaker_id:
            return jsonify({"error": "speaker_id is required"}), 400
        
        # 保存临时文件
        temp_path = Path(tempfile.mktemp(suffix='.wav'))
        audio_file.save(temp_path)
        
        # 注册声纹
        result = modules['speaker'].register_speaker(
            speaker_id=speaker_id,
            audio_path=str(temp_path)
        )
        
        # 删除临时文件
        temp_path.unlink()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Speaker register error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/speaker/recognize', methods=['POST'])
def speaker_recognize():
    """声纹识别接口"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # 保存临时文件
        temp_path = Path(tempfile.mktemp(suffix='.wav'))
        audio_file.save(temp_path)
        
        # 识别声纹
        result = modules['speaker'].recognize_speaker(
            audio_path=str(temp_path),
            return_all=request.args.get('return_all', 'false').lower() == 'true'
        )
        
        # 删除临时文件
        temp_path.unlink()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Speaker recognize error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/speaker/list', methods=['GET'])
def speaker_list():
    """列出所有注册的说话人"""
    try:
        speakers = modules['speaker'].list_speakers()
        return jsonify({"speakers": speakers})
    except Exception as e:
        logger.error(f"Speaker list error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/dialogue', methods=['POST'])
def dialogue_endpoint():
    """对话接口"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "query is required"}), 400
        
        query = data['query']
        session_id = data.get('session_id', 'default')
        reset = data.get('reset', False)
        
        # 执行对话
        result = modules['dialogue'].chat(
            query=query,
            session_id=session_id,
            reset=reset
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Dialogue endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/tts', methods=['POST'])
def tts_endpoint():
    """语音合成接口"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "text is required"}), 400
        
        text = data['text']
        
        # 执行语音合成
        result = modules['tts'].synthesize(text=text)
        logger.info(f"TTS result: {result}")
        
        if result.get('output_path'):
            output_path = result['output_path']
            # 根据文件扩展名确定mimetype
            if output_path.endswith('.aiff'):
                mimetype = 'audio/aiff'
                download_name = 'speech.aiff'
            else:
                mimetype = 'audio/wav'
                download_name = 'speech.wav'
            
            # 返回音频文件
            return send_file(
                output_path,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_name
            )
        else:
            logger.error(f"TTS synthesis returned no output_path: {result}")
            return jsonify({"error": "TTS synthesis failed", "details": result.get('error', 'unknown')}), 500
        
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """完整对话流程接口（ASR + Emotion + Speaker + Dialogue + TTS）"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('session_id', 'default')
        
        # 保存临时文件
        temp_path = Path(tempfile.mktemp(suffix='.wav'))
        audio_file.save(temp_path)
        
        # 1. 语音识别
        asr_result = modules['asr'].transcribe(audio_path=str(temp_path))
        text = asr_result.get('text', '')
        
        if not text:
            temp_path.unlink()
            return jsonify({"error": "No speech detected"}), 400
        
        # 2. 情感识别
        if modules.get('emotion') is None:
            emotion_result = {"emotion": "unknown", "error": "Emotion module not available"}
        else:
            emotion_result = modules['emotion'].predict(audio_path=str(temp_path))
        
        # 3. 声纹识别
        if modules.get('speaker') is None:
            speaker_result = {"speaker_id": "unknown", "similarity": 0.0, "recognized": False, "error": "Speaker module not available"}
            logger.warning("Speaker module not available, returning unknown speaker")
        else:
            speaker_result = modules['speaker'].recognize_speaker(audio_path=str(temp_path))
        
        # 删除临时音频文件
        temp_path.unlink()
        
        # 4. 对话生成
        dialogue_result = modules['dialogue'].chat(
            query=text,
            session_id=session_id
        )
        response_text = dialogue_result.get('response', '')
        
        # 5. 语音合成
        tts_result = modules['tts'].synthesize(text=response_text)
        
        # 返回完整结果
        result = {
            "asr": asr_result,
            "emotion": emotion_result,
            "speaker": speaker_result,
            "dialogue": dialogue_result,
            "tts": {
                "output_path": tts_result.get('output_path'),
                "duration": tts_result.get('duration')
            }
        }
        
        # 如果有音频文件，返回音频
        if tts_result.get('output_path'):
            from flask import make_response
            from urllib.parse import quote
            response = make_response(send_file(
                tts_result['output_path'],
                mimetype='audio/wav',
                as_attachment=True,
                download_name='response.wav'
            ))
            # 添加自定义响应头（URL编码中文字符）
            response.headers['X-ASR-Text'] = quote(text, safe='')
            response.headers['X-Response-Text'] = quote(response_text, safe='')
            response.headers['X-Emotion'] = emotion_result.get('emotion', '')
            response.headers['X-Speaker'] = speaker_result.get('speaker_id', '')
            response.headers['X-RAG-Used'] = str(dialogue_result.get('rag_used', False))
            return response
        else:
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/info', methods=['GET'])
def info_endpoint():
    """获取系统信息"""
    try:
        info = {
            "config": config,
            "modules": {}
        }
        
        # 获取各模块信息
        if modules.get('asr'):
            info['modules']['asr'] = modules['asr'].get_model_info()
        
        if modules.get('speaker'):
            info['modules']['speaker'] = modules['speaker'].get_statistics()
        
        if modules.get('dialogue') and hasattr(modules['dialogue'], 'get_model_info'):
            info['modules']['dialogue'] = modules['dialogue'].get_model_info()
        
        if modules.get('tts') and hasattr(modules['tts'], 'get_model_info'):
            info['modules']['tts'] = modules['tts'].get_model_info()
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Info endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


# ============= 医疗功能API =============

@app.route('/patient/triage', methods=['POST'])
def patient_triage():
    """
    患者导诊接口（RAG + LLM 模式）
    
    输入参数：
    - query: 自然语言症状描述
    - age: 年龄（可选）
    - gender: 性别（可选）
    """
    try:
        data = request.json
        query = data.get('query', '')
        age = data.get('age')
        gender = data.get('gender')
        
        if not query:
            return jsonify({"error": "缺少症状描述，请提供 query 参数"}), 400
        
        triage_module = modules.get('triage')
        if not triage_module:
            return jsonify({"error": "导诊模块未初始化"}), 500
        
        result = triage_module.triage(query, age, gender)
        
        return jsonify({
            "status": "success",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Triage endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/doctor/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    """
    医生端：辅助诊断接口（RAG + LLM 模式）
    
    输入参数：
    - query: 自然语言症状描述
    - symptoms: 症状列表（可选，如果没有query则使用）
    - patient_info: 患者信息（年龄、性别、病史等，可选）
    """
    try:
        data = request.json
        query = data.get('query', '')
        symptoms = data.get('symptoms', [])
        patient_info = data.get('patient_info', {})
        
        if not query and not symptoms:
            return jsonify({"error": "缺少症状信息，请提供 query 或 symptoms"}), 400
        
        diagnosis_module = modules.get('diagnosis')
        if not diagnosis_module:
            return jsonify({"error": "诊断模块未初始化"}), 500
        
        result = diagnosis_module.diagnose(
            query=query,
            symptoms=symptoms if not query else None,
            patient_info=patient_info
        )
        
        return jsonify({
            "status": "success",
            "analysis": result
        })
        
    except Exception as e:
        logger.error(f"Symptom analysis endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/medication/query', methods=['POST'])
def query_medication():
    """
    查询药品信息
    """
    try:
        data = request.json
        med_name = data.get('medication', '')
        
        if not med_name:
            return jsonify({"error": "缺少药品名称"}), 400
        
        med_module = modules.get('medication')
        if not med_module:
            return jsonify({"error": "用药模块未初始化"}), 500
        
        result = med_module.query_medication(med_name)
        
        if result:
            return jsonify({
                "status": "success",
                "medication": result
            })
        else:
            return jsonify({
                "status": "not_found",
                "message": f"未找到药品：{med_name}"
            }), 404
        
    except Exception as e:
        logger.error(f"Medication query endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/medication/check-interactions', methods=['POST'])
def check_interactions():
    """
    检查药物相互作用
    """
    try:
        data = request.json
        medications = data.get('medications', [])
        
        if len(medications) < 2:
            return jsonify({"error": "至少需要两个药品"}), 400
        
        med_module = modules.get('medication')
        if not med_module:
            return jsonify({"error": "用药模块未初始化"}), 500
        
        warnings = med_module.check_interactions(medications)
        
        return jsonify({
            "status": "success",
            "medications": medications,
            "warnings": warnings,
            "safe": len(warnings) == 0
        })
        
    except Exception as e:
        logger.error(f"Interaction check endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/medication/dosage-recommendation', methods=['POST'])
def dosage_recommendation():
    """
    获取用药剂量建议
    """
    try:
        data = request.json
        med_name = data.get('medication', '')
        patient_info = data.get('patient_info', {})
        
        if not med_name:
            return jsonify({"error": "缺少药品名称"}), 400
        
        med_module = modules.get('medication')
        if not med_module:
            return jsonify({"error": "用药模块未初始化"}), 500
        
        result = med_module.get_dosage_recommendation(med_name, patient_info)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Dosage recommendation endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/medication/check-contraindications', methods=['POST'])
def check_contraindications():
    """
    检查用药禁忌
    """
    try:
        data = request.json
        med_name = data.get('medication', '')
        patient_info = data.get('patient_info', {})
        
        if not med_name:
            return jsonify({"error": "缺少药品名称"}), 400
        
        med_module = modules.get('medication')
        if not med_module:
            return jsonify({"error": "用药模块未初始化"}), 500
        
        result = med_module.check_contraindications(med_name, patient_info)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Contraindication check endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/medication/search-by-indication', methods=['POST'])
def search_by_indication():
    """
    根据适应症搜索药品
    """
    try:
        data = request.json
        indication = data.get('indication', '')
        
        if not indication:
            return jsonify({"error": "缺少适应症"}), 400
        
        med_module = modules.get('medication')
        if not med_module:
            return jsonify({"error": "用药模块未初始化"}), 500
        
        results = med_module.search_by_indication(indication)
        
        return jsonify({
            "status": "success",
            "indication": indication,
            "medications": results
        })
        
    except Exception as e:
        logger.error(f"Search by indication endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/departments/list', methods=['GET'])
def list_departments():
    """
    列出所有科室
    """
    try:
        triage_module = modules.get('triage')
        if not triage_module:
            return jsonify({"error": "导诊模块未初始化"}), 500
        
        departments = triage_module.list_departments()
        
        return jsonify({
            "status": "success",
            "departments": departments
        })
        
    except Exception as e:
        logger.error(f"List departments endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 初始化所有模块
    initialize_modules()
    
    # 启动服务器
    server_config = config.get('server', {})
    app.run(
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 5000),
        debug=server_config.get('debug', False)
    )
