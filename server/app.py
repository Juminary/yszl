"""
语音助手服务器主应用
提供RESTful API接口
"""

import os
import sys
import warnings
import logging
from urllib.parse import quote

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

from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
import yaml
from pathlib import Path
import tempfile
from datetime import datetime, time
import queue
import threading
import json

# 设置模型缓存目录到项目的 models 文件夹
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ['MODELSCOPE_CACHE'] = MODEL_CACHE_DIR
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
# 设置 HuggingFace 镜像源（如果未设置）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入各个模块 - 按分类组织
# Core modules
from modules.core.asr import ASRModule
from modules.core.dialogue import DialogueModule, SimplDialogueModule
from modules.core.tts import TTSModule, SimpleTTSModule
from modules.core.rag import RAGModule, SimpleRAGModule

# Audio modules  
from modules.audio.emotion import EmotionModule
from modules.audio.speaker import SpeakerModule

# Medical modules
from modules.medical.triage import TriageModule
from modules.medical.diagnosis_assistant import DiagnosisAssistant
from modules.medical.medication import MedicationModule

# ACI modules
from modules.aci.consultation_session import ConsultationSession
from modules.aci.clinical_entity_extractor import ClinicalEntityExtractor
from modules.aci.soap_generator import SOAPGenerator
from modules.aci.hallucination_detector import HallucinationDetector
from modules.aci.emergency_detector import EmergencyDetector


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

# 创建Flask应用（配置静态文件服务）
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# 全局变量存储模块实例
modules = {}
config = {}

# ========================================
# 消息广播系统 (SSE - Server-Sent Events)
# 用于客户端和网页之间同步消息
# ========================================
message_subscribers = []  # 存储所有SSE订阅者的队列

# ========================================
# 会话模式存储
# 存储每个 session_id 的当前模式 (patient/doctor/consultation)
# ========================================
session_modes = {}  # {session_id: 'patient' | 'doctor' | 'consultation'}
consultation_sessions = {}  # {session_id: ConsultationSession 实例}
session_voice_preferences = {}  # {session_id: {"current": str | None, "awaiting_selection": bool}}

def broadcast_message(msg_type: str, data: dict):
    """广播消息给所有订阅者"""
    message = {
        "type": msg_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    # 复制列表以避免迭代时修改
    for q in message_subscribers[:]:
        try:
            q.put_nowait(message)
        except:
            pass


def get_voice_state(session_id: str) -> dict:
    """获取或初始化音色偏好状态"""
    state = session_voice_preferences.get(session_id)
    if state is None:
        state = {"current": None, "awaiting_selection": False}
        session_voice_preferences[session_id] = state
    return state


def get_available_voice_clones() -> list:
    """获取当前可用的音色克隆列表"""
    tts_module = modules.get('tts')
    if not tts_module:
        return []
    try:
        if hasattr(tts_module, 'list_voice_clones'):
            clones = tts_module.list_voice_clones() or []
            return clones
        if hasattr(tts_module, 'voice_clones'):
            return list(tts_module.voice_clones.keys())
    except Exception as exc:
        logger.warning(f"Failed to fetch voice clone list: {exc}")
    return []


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
        # 初始化ASR模块 (SenseVoice - 多语言/方言)
        asr_config = config.get('asr', {})
        modules['asr'] = ASRModule(
            model_name=asr_config.get('model', 'sensevoice'),
            device=asr_config.get('device', 'cpu'),
            language=asr_config.get('language', 'auto')
        )
        logger.info("ASR module initialized (SenseVoice)")
        
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
                top_k=rag_config.get('top_k', 3),
                min_score=rag_config.get('min_score', 0.5)
            )
            modules['rag'] = rag_module
            logger.info(f"RAG module initialized with {rag_module.get_info().get('document_count', 0)} documents")
            
            # 初始化知识图谱模块并与 RAG 集成
            kg_config = config.get('knowledge_graph', {})
            if kg_config.get('enabled', False):
                try:
                    from modules.knowledge.knowledge_graph import KnowledgeGraphModule
                    
                    print("\n" + "-"*50, flush=True)
                    print("🔗 [知识图谱] 正在初始化...", flush=True)
                    print(f"   Neo4j: {kg_config.get('host', 'localhost')}:{kg_config.get('port', 7474)}", flush=True)
                    
                    kg_module = KnowledgeGraphModule(
                        host=kg_config.get('host', 'localhost'),
                        port=kg_config.get('port', 7474),
                        user=kg_config.get('user', 'neo4j'),
                        password=kg_config.get('password', '12345')
                    )
                    
                    if kg_module.enabled:
                        rag_module.knowledge_graph = kg_module
                        modules['knowledge_graph'] = kg_module
                        # 显示 NLU 模块状态
                        nlu_info = kg_module.get_info().get('nlu_modules', {})
                        print(f"   NLU模块: 词典={nlu_info.get('medical_dict')}, 意图={nlu_info.get('intent_classifier')}, Cypher={nlu_info.get('cypher_generator')}", flush=True)
                        print("-"*50 + "\n", flush=True)
                        logger.info("Knowledge Graph integrated with RAG")
                    else:
                        print("   ✗ Neo4j 连接失败，知识图谱功能禁用", flush=True)
                        print("   （RAG 向量检索仍然可用）", flush=True)
                        print("-"*50 + "\n", flush=True)
                        logger.warning("Knowledge Graph not available, RAG will work without it")
                except Exception as kg_e:
                    print(f"   ✗ 知识图谱初始化异常: {kg_e}", flush=True)
                    print("-"*50 + "\n", flush=True)
                    logger.warning(f"Knowledge Graph initialization failed: {kg_e}")
            else:
                logger.info("Knowledge Graph disabled in config")
                
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
        # 初始化对话模块
        dialogue_config = config.get('dialogue', {})
        provider = dialogue_config.get('provider', 'transformers')
        
        if provider == 'gguf':
            # 使用 GGUF 量化模型
            try:
                from modules.core.gguf_dialogue import GGUFDialogueModule, download_gguf_model
                
                gguf_repo = dialogue_config.get('gguf_repo', 'unsloth/Qwen3-4B-GGUF')
                gguf_file = dialogue_config.get('gguf_file', 'Qwen3-4B-Q4_K_M.gguf')
                gguf_source = dialogue_config.get('gguf_source', 'huggingface')
                gguf_dir = os.path.join(os.path.dirname(__file__), 'models', 'gguf')
                
                # 检查并下载模型
                model_path = os.path.join(gguf_dir, gguf_file)
                if not os.path.exists(model_path):
                    model_path = download_gguf_model(gguf_repo, gguf_file, gguf_dir, gguf_source)
                
                modules['dialogue'] = GGUFDialogueModule(
                    model_path=model_path,
                    temperature=dialogue_config.get('temperature', 0.7),
                    top_p=dialogue_config.get('top_p', 0.9),
                    max_tokens=dialogue_config.get('max_length', 512),
                    system_prompt=dialogue_config.get('system_prompt'),
                    rag_module=rag_module
                )
                logger.info(f"Dialogue module initialized (GGUF: {gguf_file})" + (" with RAG" if rag_module else ""))
                
            except Exception as e:
                logger.warning(f"GGUF failed, falling back to transformers: {e}")
                provider = 'transformers'  # 回退到 transformers
        
        if provider == 'transformers':
            # 使用 Transformers 模型
            try:
                modules['dialogue'] = DialogueModule(
                    model_name=dialogue_config.get('model', 'Qwen/Qwen2.5-0.5B-Instruct'),
                    device=dialogue_config.get('device', 'cpu'),
                    max_length=dialogue_config.get('max_length', 512),
                    temperature=dialogue_config.get('temperature', 0.7),
                    top_p=dialogue_config.get('top_p', 0.9),
                    history_length=dialogue_config.get('history_length', 10),
                    rag_module=rag_module,
                    system_prompt=dialogue_config.get('system_prompt')
                )
                logger.info("Dialogue module initialized (Transformers)" + (" with RAG" if rag_module else ""))
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
        
        # 患者导诊模块（支持 RAG + LLM + 实体提取）
        from modules.medical.triage import TriageService
        hospital_db_path = os.path.join(os.path.dirname(__file__), 'data', 'hospital.db')
        
        modules['triage'] = TriageService(
            db_path=hospital_db_path,
            dialogue_module=modules.get('dialogue'),
            rag_module=rag_module,
            entity_extractor=entity_extractor
        )
        logger.info(f"Triage module initialized with SQLite: {hospital_db_path}")
        
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


# ============= Web界面路由 =============

@app.route('/')
def index():
    """返回Web界面首页"""
    return send_from_directory('static', 'index.html')


@app.route('/consultation')
def consultation_page():
    """返回会诊病历生成器页面"""
    return send_from_directory('static', 'consultation.html')


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


@app.route('/events', methods=['GET'])
def events_stream():
    """SSE事件流 - 用于实时推送客户端消息到网页"""
    def generate():
        q = queue.Queue()
        message_subscribers.append(q)
        try:
            # 发送连接成功事件
            yield f"data: {json.dumps({'type': 'connected', 'message': '已连接到消息流'})}\n\n"
            
            while True:
                try:
                    # 等待消息，超时30秒发送心跳
                    message = q.get(timeout=30)
                    yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    # 发送心跳保持连接
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            message_subscribers.remove(q)
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


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
    """声纹注册接口（同时注册音色克隆）"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        speaker_id = request.form.get('speaker_id')
        prompt_text = request.form.get('prompt_text')  # 获取提示文本（可选）
        
        if not speaker_id:
            return jsonify({"error": "speaker_id is required"}), 400
        
        # 保存临时文件
        temp_path = Path(tempfile.mktemp(suffix='.wav'))
        audio_file.save(temp_path)
        
        # 准备 metadata（包含 voice_clone_path 和 prompt_text）
        # 注意：这里先准备 metadata，但 voice_clone_path 会在后面生成
        # 所以先注册声纹，然后再更新 metadata
        result = modules['speaker'].register_speaker(
            speaker_id=speaker_id,
            audio_path=str(temp_path),
            metadata={}  # 先传空，后面会更新
        )
        
        # 如果声纹注册成功，同时注册音色克隆
        if result.get('status') == 'success' and modules.get('tts'):
            # 将音频保存到专门目录用于音色克隆
            voice_clone_dir = Path(__file__).parent / "data" / "voice_clones"
            voice_clone_dir.mkdir(parents=True, exist_ok=True)
            voice_clone_path = voice_clone_dir / f"{speaker_id}.wav"
            
            # 复制音频文件到音色克隆目录
            import shutil
            shutil.copy2(temp_path, voice_clone_path)
            
            # 如果没有提供 prompt_text，使用默认值
            if not prompt_text:
                prompt_text = "你好，我是医生，很高兴为您服务。"
            
            # 更新 speaker_db 中的 metadata，保存 voice_clone_path 和 prompt_text
            try:
                speaker_db_path = Path(__file__).parent / "data" / "speaker_db.pkl"
                if speaker_db_path.exists():
                    import pickle
                    with open(speaker_db_path, 'rb') as f:
                        speaker_db = pickle.load(f)
                    
                    if speaker_id in speaker_db:
                        if 'metadata' not in speaker_db[speaker_id]:
                            speaker_db[speaker_id]['metadata'] = {}
                        speaker_db[speaker_id]['metadata']['voice_clone_path'] = str(voice_clone_path)
                        speaker_db[speaker_id]['metadata']['prompt_text'] = prompt_text
                        
                        with open(speaker_db_path, 'wb') as f:
                            pickle.dump(speaker_db, f)
                        logger.info(f"Updated speaker_db metadata for {speaker_id}")
            except Exception as e:
                logger.warning(f"Failed to update speaker_db metadata: {e}")
            
            # 注册音色克隆到TTS模块
            try:
                voice_clone_success = modules['tts'].register_voice_clone(
                    speaker_id=speaker_id,
                    audio_path=str(voice_clone_path),
                    prompt_text=prompt_text
                )
                
                if voice_clone_success:
                    result['voice_clone_registered'] = True
                    logger.info(f"Voice clone registered for speaker: {speaker_id}")
                else:
                    result['voice_clone_registered'] = False
                    result['voice_clone_error'] = "Failed to register voice clone in CosyVoice"
                    logger.warning(f"Voice clone registration failed for speaker: {speaker_id}")
            except Exception as e:
                result['voice_clone_registered'] = False
                result['voice_clone_error'] = str(e)
                logger.error(f"Voice clone registration error for speaker {speaker_id}: {e}", exc_info=True)
        
        # 删除临时文件
        temp_path.unlink()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Speaker register error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/voice-clone/list', methods=['GET'])
def list_voice_clones():
    """列出所有可用的音色克隆"""
    try:
        if modules.get('tts') and hasattr(modules['tts'], 'list_voice_clones'):
            # 直接返回已加载的音色克隆列表，不重新加载（避免超时）
            # 音色克隆在启动时和注册时已经加载到内存中
            voice_clones = modules['tts'].list_voice_clones()
            logger.info(f"List voice clones: {len(voice_clones)} clones available")
            return jsonify({
                "status": "success",
                "voice_clones": voice_clones,
                "count": len(voice_clones)
            })
        else:
            return jsonify({
                "status": "error",
                "error": "TTS module not available or does not support voice cloning"
            }), 500
    except Exception as e:
        logger.error(f"List voice clones error: {e}")
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


@app.route('/speaker/delete', methods=['POST'])
def speaker_delete():
    """删除说话人接口（同时删除音色克隆）"""
    try:
        # 支持 JSON 和 form-data 两种方式
        if request.is_json:
            data = request.json
            speaker_id = data.get('speaker_id')
        else:
            speaker_id = request.form.get('speaker_id')
        
        if not speaker_id:
            return jsonify({"error": "speaker_id is required"}), 400
        
        # 检查说话人是否存在
        if modules.get('speaker') is None:
            return jsonify({"error": "Speaker module not available"}), 500
        
        speakers = modules['speaker'].list_speakers()
        speaker_ids = [s.get('speaker_id') for s in speakers]
        if speaker_id not in speaker_ids:
            return jsonify({"error": f"Speaker {speaker_id} not found"}), 404
        
        # 删除说话人（同时删除音色克隆文件）
        result = modules['speaker'].delete_speaker(speaker_id, delete_voice_clone=True)
        
        if result.get('status') == 'error':
            return jsonify(result), 400
        
        # 从TTS模块中注销音色克隆
        if modules.get('tts'):
            try:
                modules['tts'].unregister_voice_clone(speaker_id)
                result['voice_clone_unregistered'] = True
                logger.info(f"Unregistered voice clone for speaker: {speaker_id}")
            except Exception as e:
                result['voice_clone_unregistered'] = False
                result['voice_clone_error'] = str(e)
                logger.warning(f"Failed to unregister voice clone for speaker {speaker_id}: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Speaker delete error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/dialogue', methods=['POST'])
def dialogue_endpoint():
    """对话接口 - 支持语音命令切换模式"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "query is required"}), 400
        
        query = data['query']
        session_id = data.get('session_id', 'default')
        reset = data.get('reset', False)
        mode = data.get('mode', 'patient')  # patient | doctor | consultation
        emotion = data.get('emotion', 'neutral')  # 用户情绪
        emotional_mode = data.get('emotional_mode', False)  # 是否启用情感感知模式
        
        # ========================================
        # 语音命令模式切换检测
        # ========================================
        mode_switch_commands = {
            'patient': ['切换到患者模式', '患者模式', '切换患者模式', '进入患者模式', '我是患者'],
            'doctor': ['切换到医生模式', '医生模式', '切换医生模式', '进入医生模式', '我是医生'],
            'consultation': ['切换到会诊模式', '会诊模式', '开始会诊', '进入会诊模式', '启动会诊']
        }
        
        # 检查是否是模式切换命令
        query_clean = query.strip().replace(' ', '')
        for target_mode, commands in mode_switch_commands.items():
            for cmd in commands:
                if cmd.replace(' ', '') in query_clean or query_clean in cmd.replace(' ', ''):
                    # 检测到模式切换命令
                    mode_names = {'patient': '患者', 'doctor': '医生', 'consultation': '会诊'}
                    response_text = f"好的，已切换到{mode_names[target_mode]}模式。"
                    
                    if target_mode == 'patient':
                        response_text += "请描述您的症状，我会为您提供导诊建议。"
                    elif target_mode == 'doctor':
                        response_text += "我将为您提供专业的辅助诊断建议。"
                    elif target_mode == 'consultation':
                        response_text += "会诊模式已启动，我会记录对话并生成病历。"
                    
                    return jsonify({
                        "text": response_text,
                        "mode": target_mode,
                        "mode_switched": True,
                        "previous_mode": mode
                    })
        
        # ========================================
        # 正常对话处理
        # ========================================
        
        # 根据模式选择不同的系统提示词
        mode_prompts = {
            'patient': """你是一个智能医疗助手，帮助患者分析症状、给出初步的健康生活建议，并建议应该挂什么科室。
你的职责是，根据患者描述的症状，分析可能的健康问题，提供初步的日常护理建议（如休息、饮食、补水等），并建议患者应该去哪个科室就诊。
你的回答将被直接用于语音合成朗读，因此必须遵守以下格式要求，
只用纯中文回答，禁止英文和数字。
只用中文逗号和句号，禁止其他标点。
禁止使用列表和编号格式，必须写成连贯的一段话。
态度温和友好，像一个耐心的专业护士。
在提供建议的同时，必须告知患者这不替代医生面诊，必要时及时就医。""",

            'doctor': """你是医生的AI诊断辅助助手，帮助医生分析病情、提供鉴别诊断和治疗方案建议。
你应该使用专业的医学术语，提供基于循证医学的建议。
你的职责包括，分析患者症状提供鉴别诊断，建议必要的检查项目，提供治疗方案参考，提示潜在的风险和禁忌症。
你的回答将被直接用于语音合成朗读，因此必须遵守以下格式要求，
只用纯中文回答，禁止英文和数字。
只用中文逗号和句号，禁止其他标点。
禁止使用列表和编号格式，必须写成连贯的一段话。
态度专业严谨，像一个经验丰富的主治医师在与同事讨论病例。""",

            'consultation': """你是会诊记录助手，正在记录医患对话。
请简洁回应确认你正在记录，不需要提供医疗建议。
回复简短即可，如"好的，已记录"或"继续"。"""
        }
        
        system_prompt = mode_prompts.get(mode, mode_prompts['patient'])
        
        # ========================================
        # 患者模式：根据意图决定是否导诊
        # ========================================
        if mode == 'patient' and 'triage' in modules:
            # 判断是否需要导诊的关键词
            triage_keywords = [
                '挂什么科', '看什么科', '去哪个科', '哪个科室',
                '挂号', '看医生', '去医院', '要不要去医院',
                '应该挂', '建议挂', '推荐科室', '推荐医生',
                '帮我挂', '需要看医生', '想看医生'
            ]
            
            # 症状描述词（表示可能需要导诊）
            symptom_patterns = [
                '疼', '痛', '发烧', '发热', '咳嗽', '头晕', '恶心', '呕吐',
                '拉肚子', '腹泻', '胸闷', '心慌', '不舒服', '难受',
                '三天', '一周', '几天了', '好久了'
            ]
            
            # 非导诊请求的关键词
            non_triage_keywords = [
                '吃什么', '怎么办', '注意什么', '食疗', '食物',
                '如何预防', '怎么治', '怎么调理', '有什么偏方',
                '你是谁', '你好', '谢谢', '再见'
            ]
            
            # 检测是否明确请求导诊
            needs_triage = any(kw in query for kw in triage_keywords)
            
            # 如果没有明确请求导诊，但有症状描述且没有非导诊关键词，也触发导诊
            has_symptoms = any(kw in query for kw in symptom_patterns)
            has_non_triage = any(kw in query for kw in non_triage_keywords)
            
            if not needs_triage and has_symptoms and not has_non_triage:
                # 症状描述，可能需要导诊，但不确定
                # 可以考虑提示用户是否需要导诊，这里暂时触发导诊
                needs_triage = True
            
            print(f"\n[DEBUG] 意图分析: 需要导诊={needs_triage}, 有症状={has_symptoms}, 非导诊={has_non_triage}")
            
            if needs_triage:
                try:
                    print(f"[DEBUG] 调用导诊服务，输入: {query}")
                    triage_result = modules['triage'].analyze(query)
                    print(f"[DEBUG] 导诊结果: 科室={triage_result.get('department', {}).get('name', '未匹配')}")
                    
                    # 只有当匹配到科室时才使用导诊回复
                    if triage_result.get('department') and triage_result.get('response'):
                        print(f"[DEBUG] 使用导诊服务回复")
                        result = {
                            'response': triage_result['response'],
                            'mode': mode,
                            'mode_switched': False,
                            'triage': {
                                'department': triage_result.get('department', {}),
                                'doctors': triage_result.get('doctors', []),
                                'diseases': triage_result.get('diseases', [])
                            }
                        }
                        return jsonify(result)
                    else:
                        print(f"[DEBUG] 导诊未匹配到科室，使用普通对话")
                except Exception as e:
                    print(f"[DEBUG] 导诊服务失败: {e}")
                    logger.warning(f"Triage failed, using dialogue: {e}")
            else:
                print(f"[DEBUG] 非导诊请求，使用普通对话")
        
        # 执行对话
        result = modules['dialogue'].chat(
            query=query,
            session_id=f"{session_id}_{mode}",  # 不同模式使用不同的会话历史
            reset=reset,
            system_prompt=system_prompt,
            emotion=emotion,
            emotional_mode=emotional_mode
        )
        
        # 更新 RAG 状态等信息
        result['mode'] = mode
        result['mode_switched'] = False
        
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


@app.route('/tts/stream', methods=['POST'])
def tts_stream_endpoint():
    """
    流式语音合成接口
    边生成边返回音频数据，降低首音频延迟
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "text is required"}), 400
        
        text = data['text']
        speaker = data.get('speaker')
        speed = data.get('speed', 1.0)
        
        # 检查 TTS 模块是否支持流式
        tts_module = modules.get('tts')
        if tts_module is None:
            return jsonify({"error": "TTS module not available"}), 500
        
        if not hasattr(tts_module, 'synthesize_stream'):
            logger.warning("TTS module does not support streaming, falling back to regular synthesis")
            # 回退到普通模式
            result = tts_module.synthesize(text=text)
            if result.get('output_path'):
                return send_file(result['output_path'], mimetype='audio/wav')
            else:
                return jsonify({"error": "TTS synthesis failed"}), 500
        
        logger.info(f"[Streaming TTS] Request: {text[:50]}...")
        
        def generate():
            """生成器函数，逐块返回音频数据"""
            for chunk in tts_module.synthesize_stream(text, speaker, speed):
                yield chunk
        
        from flask import Response
        return Response(
            generate(),
            mimetype='audio/wav',
            headers={
                'Transfer-Encoding': 'chunked',
                'X-Content-Type-Options': 'nosniff'
            }
        )
        
    except Exception as e:
        logger.error(f"TTS stream endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """完整对话流程接口（ASR + Emotion + Speaker + Dialogue + TTS）"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('session_id', 'default')
        # 从请求中获取模式，如果提供则更新 session_modes
        mode_from_request = request.form.get('mode')
        if mode_from_request and mode_from_request in ['patient', 'doctor', 'consultation']:
            session_modes[session_id] = mode_from_request
            logger.info(f"[Mode] 从请求更新模式: session={session_id}, mode={mode_from_request}")
        current_mode = session_modes.get(session_id, 'patient')
        
        # 处理音色偏好
        voice_state = get_voice_state(session_id)
        available_voice_clones = get_available_voice_clones()
        if voice_state.get('current') and voice_state['current'] not in available_voice_clones:
            logger.info(f"Voice clone '{voice_state['current']}' no longer available, reverting to default")
            voice_state['current'] = None
        
        requested_voice_clone = request.form.get('voice_clone_id')
        voice_clone_id = requested_voice_clone.strip() if requested_voice_clone else None
        
        # 修复逻辑：明确区分"使用默认音色"和"使用之前保存的音色"
        # 如果用户明确指定 '0' 或空字符串，表示要使用默认音色，应该清除保存的音色
        if voice_clone_id in ("0", ""):
            voice_clone_id = None
            voice_state['current'] = None  # 清除之前保存的音色，确保使用默认音色
            logger.info("User explicitly requested default voice, clearing saved voice preference")
        elif voice_clone_id and voice_clone_id in available_voice_clones:
            # 用户指定了有效的音色克隆ID，保存并使用
            voice_state['current'] = voice_clone_id
            logger.info(f"Using requested voice clone: {voice_clone_id}")
        elif voice_clone_id and voice_clone_id not in available_voice_clones:
            # 用户指定的音色克隆ID无效，回退到默认音色（不使用之前保存的音色）
            logger.warning(f"Voice clone ID '{voice_clone_id}' not found, using default voice")
            voice_clone_id = None
            voice_state['current'] = None
        else:
            # 用户没有指定音色，使用之前保存的音色（如果有）
            voice_clone_id = voice_state.get('current')
            if voice_clone_id:
                logger.info(f"Using saved voice preference: {voice_clone_id}")
            else:
                logger.info("Using default voice (no voice preference set)")
        
        if voice_clone_id:
            logger.info(f"Final voice clone selection: {voice_clone_id}")
        
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
        
        # ========================================
        # 语音命令模式切换检测
        # ========================================
        
        # 检测 "结束会诊" 命令
        end_consultation_commands = ['结束会诊', '会诊结束', '生成病历', '生成SOAP']
        text_clean = text.strip().replace(' ', '')

        # ========================================
        # 音色切换语音命令
        # ========================================
        if voice_state.get('awaiting_selection'):
            available_voice_clones = [str(name) for name in get_available_voice_clones() if name]
            voice_state['awaiting_selection'] = False
            selected_voice = None
            normalized_text = text.strip().replace(' ', '').replace('，', '').replace(',', '')
            lower_text = normalized_text.lower()
            
            # 改进音色匹配逻辑：支持完整匹配、部分匹配和ID匹配
            for clone_name in available_voice_clones:
                clone_key = str(clone_name)
                clone_lower = clone_key.lower()
                # 完整匹配
                if clone_lower == lower_text or clone_key == normalized_text:
                    selected_voice = clone_key
                    break
                # 部分匹配（音色名称包含在识别文本中，或识别文本包含音色名称）
                if clone_lower in lower_text or lower_text in clone_lower:
                    selected_voice = clone_key
                    break
                # ID匹配（如果音色名称是纯数字或包含数字）
                if clone_key.isdigit() and clone_key in normalized_text:
                    selected_voice = clone_key
                    break
            
            voice_response_voice = None
            if not available_voice_clones:
                response_text = "当前没有可切换的音色，将继续使用默认音色。"
                voice_state['current'] = None
            elif selected_voice:
                voice_state['current'] = selected_voice
                response_text = "已成功切换"
                voice_response_voice = selected_voice
            else:
                voice_state['current'] = None
                response_text = "无目标音色，使用默认音色"
            voice_clone_id = voice_state.get('current')
            broadcast_message('user_message', {
                'text': text,
                'mode': current_mode,
                'source': 'client',
                'emotion': emotion_result.get('emotion', '') if isinstance(emotion_result, dict) else '',
                'emotion_score': emotion_result.get('score') if isinstance(emotion_result, dict) else None,
                'speaker_id': speaker_result.get('speaker_id', '') if isinstance(speaker_result, dict) else '',
                'speaker_score': speaker_result.get('similarity') if isinstance(speaker_result, dict) else None
            })
            broadcast_message('assistant_message', {
                'text': response_text,
                'mode': current_mode,
                'voice_action': 'voice_switch_confirm',
                'voice_target': voice_state.get('current')
            })
            tts_result = modules['tts'].synthesize(
                text=response_text,
                voice_clone_id=voice_response_voice
            )
            if tts_result.get('output_path'):
                from flask import make_response
                from urllib.parse import quote
                response = make_response(send_file(
                    tts_result['output_path'],
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name='response.wav'
                ))
                response.headers['X-ASR-Text'] = quote(text, safe='')
                response.headers['X-Response-Text'] = quote(response_text, safe='')
                response.headers['X-Voice-Action'] = 'voice_switch_confirm'
                if voice_state.get('current'):
                    response.headers['X-Voice-Clone'] = quote(voice_state['current'], safe='')
                return response
            return jsonify({
                'text': response_text,
                'voice_action': 'voice_switch_confirm',
                'voice_target': voice_state.get('current')
            })

        # 音色切换指令检测 - 支持多种变体和部分匹配
        def detect_voice_switch_intent(text: str) -> bool:
            """
            检测用户是否有切换音色的意图
            支持多种变体：切换音色、换音色、音色、切换声音、换声音等
            """
            text_clean = text.strip().replace(' ', '').replace('，', '').replace(',', '').lower()
            original_text = text.strip()
            
            # 完整匹配的触发词
            full_triggers = [
                '切换音色', '换音色', '音色切换', '切换声音', '换声音', '声音切换',
                '改音色', '换一个音色', '换个音色', '换一下音色',
                '切换语音', '换语音', '语音切换', '改语音',
                '选择音色', '选音色', '换个声音', '换一个声音'
            ]
            
            # 检查完整匹配
            for trigger in full_triggers:
                if trigger in text_clean:
                    logger.info(f"[Voice Switch] 检测到完整匹配触发词: '{trigger}' in '{original_text}'")
                    return True
            
            # 关键词组合匹配（更灵活，支持ASR识别不准确的情况）
            # 包含"切换"或"换" + "音色"或"声音"或"语音"
            switch_keywords = ['切换', '换', '改', '选', '选择']
            voice_keywords = ['音色', '声音', '语音']
            
            has_switch = any(kw in text_clean for kw in switch_keywords)
            has_voice = any(kw in text_clean for kw in voice_keywords)
            
            # 如果同时包含切换关键词和音色关键词，认为是切换音色意图
            if has_switch and has_voice:
                matched_switch = [kw for kw in switch_keywords if kw in text_clean]
                matched_voice = [kw for kw in voice_keywords if kw in text_clean]
                logger.info(f"[Voice Switch] 检测到关键词组合: 切换词={matched_switch}, 音色词={matched_voice} in '{original_text}'")
                return True
            
            # 如果只包含"音色"且文本较短（可能是识别不完整），也认为是意图
            if has_voice and len(text_clean) <= 10:
                # 但排除一些明显不是切换意图的情况
                exclude_patterns = ['什么音色', '哪个音色', '音色是什么', '音色怎么样', '音色如何', '音色好不好']
                if not any(pattern in text_clean for pattern in exclude_patterns):
                    logger.info(f"[Voice Switch] 检测到短文本音色关键词: '{original_text}' (长度={len(text_clean)})")
                    return True
            
            return False
        
        if detect_voice_switch_intent(text):
            available_voice_clones = [str(name) for name in get_available_voice_clones() if name]
            if available_voice_clones:
                voice_state['awaiting_selection'] = True
                voice_list_str = '，'.join(available_voice_clones)
                response_text = f"可以切换的音色包括{voice_list_str}，请说出想要切换的音色名称。"
            else:
                voice_state['awaiting_selection'] = False
                response_text = "当前没有可切换的音色，将继续使用默认音色。"
            broadcast_message('user_message', {
                'text': text,
                'mode': current_mode,
                'source': 'client',
                'emotion': emotion_result.get('emotion', '') if isinstance(emotion_result, dict) else '',
                'emotion_score': emotion_result.get('score') if isinstance(emotion_result, dict) else None,
                'speaker_id': speaker_result.get('speaker_id', '') if isinstance(speaker_result, dict) else '',
                'speaker_score': speaker_result.get('similarity') if isinstance(speaker_result, dict) else None
            })
            broadcast_message('assistant_message', {
                'text': response_text,
                'mode': current_mode,
                'voice_action': 'voice_switch_list',
                'voice_options': available_voice_clones
            })
            tts_result = modules['tts'].synthesize(
                text=response_text,
                voice_clone_id=voice_state.get('current')
            )
            if tts_result.get('output_path'):
                from flask import make_response
                from urllib.parse import quote
                response = make_response(send_file(
                    tts_result['output_path'],
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name='response.wav'
                ))
                response.headers['X-ASR-Text'] = quote(text, safe='')
                response.headers['X-Response-Text'] = quote(response_text, safe='')
                response.headers['X-Voice-Action'] = 'voice_switch_list'
                if available_voice_clones:
                    response.headers['X-Voice-Options'] = quote(','.join(available_voice_clones), safe='')
                return response
            return jsonify({
                'text': response_text,
                'voice_action': 'voice_switch_list',
                'voice_options': available_voice_clones
            })
        
        for cmd in end_consultation_commands:
            if cmd.replace(' ', '') in text_clean or text_clean in cmd.replace(' ', ''):
                current_mode = session_modes.get(session_id, 'patient')
                if current_mode == 'consultation' and session_id in consultation_sessions:
                    # 生成 SOAP 病历
                    try:
                        consultation = consultation_sessions[session_id]
                        soap_result = {'subjective': {}, 'objective': {}, 'assessment': {}, 'plan': {}}
                        
                        if soap_generator:
                            soap_note = soap_generator.generate_soap(consultation)
                            soap_result = soap_note.to_dict()
                        else:
                            # 简单规则生成
                            utterances = [{'speaker': '患者', 'text': u.text} for u in consultation.utterances]
                            soap_result = _generate_simple_soap(utterances, [])
                        
                        response_text = f"会诊已结束，病历已生成。主诉：{soap_result.get('subjective', {}).get('chief_complaint', '未记录')}。"
                        
                        # 清理会话
                        del consultation_sessions[session_id]
                        session_modes[session_id] = 'patient'
                        logger.info(f"[Consultation] Ended session {session_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to generate SOAP: {e}")
                        response_text = "会诊已结束，但病历生成失败。"
                else:
                    response_text = "当前不在会诊模式，无需结束会诊。"
                
                # 语音合成并返回（使用选择的音色克隆）
                tts_result = modules['tts'].synthesize(
                    text=response_text,
                    voice_clone_id=voice_clone_id
                )
                if tts_result.get('output_path'):
                    from flask import make_response
                    from urllib.parse import quote
                    response = make_response(send_file(
                        tts_result['output_path'], mimetype='audio/wav',
                        as_attachment=True, download_name='response.wav'
                    ))
                    response.headers['X-ASR-Text'] = quote(text, safe='')
                    response.headers['X-Response-Text'] = quote(response_text, safe='')
                    response.headers['X-Consultation-Ended'] = 'true'
                    return response
                else:
                    return jsonify({"text": response_text, "consultation_ended": True})
        
        mode_switch_commands = {
            'patient': ['切换到患者模式', '患者模式', '切换患者模式', '进入患者模式', '我是患者'],
            'doctor': ['切换到医生模式', '医生模式', '切换医生模式', '进入医生模式', '我是医生'],
            'consultation': ['切换到会诊模式', '会诊模式', '开始会诊', '进入会诊模式', '启动会诊']
        }
        
        for target_mode, commands in mode_switch_commands.items():
            for cmd in commands:
                if cmd.replace(' ', '') in text_clean or text_clean in cmd.replace(' ', ''):
                    # 检测到模式切换命令 - 存储到全局会话模式
                    old_mode = session_modes.get(session_id, 'patient')
                    session_modes[session_id] = target_mode
                    logger.info(f"[Mode Switch] Session {session_id}: {old_mode} -> {target_mode}")
                    
                    mode_names = {'patient': '患者', 'doctor': '医生', 'consultation': '会诊'}
                    response_text = f"好的，已切换到{mode_names[target_mode]}模式。"
                    
                    if target_mode == 'patient':
                        response_text += "请描述您的症状，我会为您提供导诊建议。"
                    elif target_mode == 'doctor':
                        response_text += "我将为您提供专业的辅助诊断建议。"
                    elif target_mode == 'consultation':
                        response_text += "会诊模式已启动，我会记录对话并生成病历。"
                        # 创建会诊会话
                        try:
                            from modules.aci.consultation_session import ConsultationSession
                            consultation_sessions[session_id] = ConsultationSession()
                            logger.info(f"[Consultation] Created session for {session_id}")
                        except Exception as e:
                            logger.warning(f"Failed to create consultation session: {e}")
                    
                    # 广播消息到网页（包含模式切换信息）
                    broadcast_message('mode_switched', {
                        'old_mode': old_mode,
                        'new_mode': target_mode,
                        'text': response_text
                    })
                    broadcast_message('user_message', {'text': text, 'mode': target_mode, 'source': 'client'})
                    broadcast_message('assistant_message', {'text': response_text, 'mode': target_mode})
                    
                    # 语音合成模式切换确认
                    tts_result = modules['tts'].synthesize(text=response_text)
                    
                    if tts_result.get('output_path'):
                        from flask import make_response
                        from urllib.parse import quote
                        response = make_response(send_file(
                            tts_result['output_path'],
                            mimetype='audio/wav',
                            as_attachment=True,
                            download_name='response.wav'
                        ))
                        response.headers['X-ASR-Text'] = quote(text, safe='')
                        response.headers['X-Response-Text'] = quote(response_text, safe='')
                        response.headers['X-Mode-Switched'] = 'true'
                        response.headers['X-New-Mode'] = target_mode
                        return response
                    else:
                        return jsonify({
                            "text": response_text,
                            "mode": target_mode,
                            "mode_switched": True,
                            "asr": asr_result
                        })
        
        # ========================================
        # 正常对话流程
        # ========================================
        
        # 4. 获取当前会话模式并使用对应的系统提示词
        current_mode = session_modes.get(session_id, 'patient')
        
        # 模式专用的系统提示词
        mode_prompts = {
            'patient': """你是一个智能医疗助手，帮助患者分析症状、给出初步的健康生活建议，并建议应该挂什么科室。
你的职责是，根据患者描述的症状，分析可能的健康问题，提供初步的日常护理建议（如休息、饮食、补水等），并建议患者应该去哪个科室就诊。
你的回答将被直接用于语音合成朗读，因此必须遵守以下格式要求，
只用纯中文回答，禁止英文和数字。
只用中文逗号和句号，禁止其他标点。
禁止使用列表和编号格式，必须写成连贯的一段话。
态度温和友好，像一个耐心的专业护士。
在提供建议的同时，必须告知患者这不替代医生面诊，必要时及时就医。""",

            'doctor': """你是医生的AI诊断辅助助手，帮助医生分析病情、提供鉴别诊断和治疗方案建议。
你应该使用专业的医学术语，提供基于循证医学的建议。
你的职责包括，分析患者症状提供鉴别诊断，建议必要的检查项目，提供治疗方案参考，提示潜在的风险和禁忌症。
你的回答将被直接用于语音合成朗读，因此必须遵守以下格式要求，
只用纯中文回答，禁止英文和数字。
只用中文逗号和句号，禁止其他标点。
禁止使用列表和编号格式，必须写成连贯的一段话。
态度专业严谨，像一个经验丰富的主治医师在与同事讨论病例。""",

            'consultation': """你是会诊记录助手，正在记录医患对话。
请简洁回应确认你正在记录，不需要提供医疗建议。
回复简短即可，如"好的，已记录"或"继续"。"""
        }
        
        system_prompt = mode_prompts.get(current_mode, mode_prompts['patient'])
        
        # 如果是会诊模式，记录到会诊会话
        if current_mode == 'consultation' and session_id in consultation_sessions:
            try:
                consultation_sessions[session_id].add_utterance(
                    text=text,
                    speaker_id=speaker_result.get('speaker_id', 'unknown'),
                    speaker_role='patient',  # 默认为患者，实际应根据声纹识别
                    timestamp=time.time()
                )
                logger.info(f"[Consultation] Recorded utterance: {text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to record consultation utterance: {e}")
        
        # ========================================
        # 患者模式：根据意图决定是否导诊（与 /dialogue 端点逻辑完全一致）
        # ========================================
        triage_result = None
        dialogue_result = None  # 初始化变量，确保在所有路径中都有定义
        
        if current_mode == 'patient' and 'triage' in modules:
            # 判断是否需要导诊的关键词
            triage_keywords = [
                '挂什么科', '看什么科', '去哪个科', '哪个科室',
                '挂号', '看医生', '去医院', '要不要去医院',
                '应该挂', '建议挂', '推荐科室', '推荐医生',
                '帮我挂', '需要看医生', '想看医生'
            ]
            
            # 症状描述词（表示可能需要导诊）
            symptom_patterns = [
                '疼', '痛', '发烧', '发热', '咳嗽', '头晕', '恶心', '呕吐',
                '拉肚子', '腹泻', '胸闷', '心慌', '不舒服', '难受',
                '三天', '一周', '几天了', '好久了'
            ]
            
            # 非导诊请求的关键词
            non_triage_keywords = [
                '吃什么', '怎么办', '注意什么', '食疗', '食物',
                '如何预防', '怎么治', '怎么调理', '有什么偏方',
                '你是谁', '你好', '谢谢', '再见'
            ]
            
            # 检测是否明确请求导诊
            needs_triage = any(kw in text for kw in triage_keywords)
            
            # 如果没有明确请求导诊，但有症状描述且没有非导诊关键词，也触发导诊
            has_symptoms = any(kw in text for kw in symptom_patterns)
            has_non_triage = any(kw in text for kw in non_triage_keywords)
            
            if not needs_triage and has_symptoms and not has_non_triage:
                # 症状描述，可能需要导诊，但不确定
                # 可以考虑提示用户是否需要导诊，这里暂时触发导诊
                needs_triage = True
            
            print(f"\n[DEBUG] 意图分析: 需要导诊={needs_triage}, 有症状={has_symptoms}, 非导诊={has_non_triage}")
            
            if needs_triage:
                try:
                    print(f"[DEBUG] 调用导诊服务，输入: {text}")
                    triage_result = modules['triage'].analyze(text)
                    print(f"[DEBUG] 导诊结果: 科室={triage_result.get('department', {}).get('name', '未匹配')}")
                    
                    # 只有当匹配到科室时才使用导诊回复
                    if triage_result.get('department') and triage_result.get('response'):
                        print(f"[DEBUG] 使用导诊服务回复")
                        response_text = triage_result['response']
                        # 即使使用导诊回复，也需要调用对话模块以获取 RAG 信息
                        dialogue_result = modules['dialogue'].chat(
                            query=text,
                            session_id=f"{session_id}_{current_mode}",
                            system_prompt=system_prompt,
                            emotion=emotion_label,
                            emotional_mode=True  # 启用情感感知模式
                        )
                    else:
                        print(f"[DEBUG] 导诊未匹配到科室，使用普通对话")
                        dialogue_result = modules['dialogue'].chat(
                            query=text,
                            session_id=f"{session_id}_{current_mode}",
                            system_prompt=system_prompt,
                            emotion=emotion_label,
                            emotional_mode=True  # 启用情感感知模式
                        )
                        response_text = dialogue_result.get('response', '')
                except Exception as e:
                    print(f"[DEBUG] 导诊服务失败: {e}")
                    logger.warning(f"Triage failed, using dialogue: {e}")
                    dialogue_result = modules['dialogue'].chat(
                        query=text,
                        session_id=f"{session_id}_{current_mode}",
                        system_prompt=system_prompt,
                        emotion=emotion_label,
                        emotional_mode=True  # 启用情感感知模式
                    )
                    response_text = dialogue_result.get('response', '')
            else:
                print(f"[DEBUG] 非导诊请求，使用普通对话")
                dialogue_result = modules['dialogue'].chat(
                    query=text,
                    session_id=f"{session_id}_{current_mode}",
                    system_prompt=system_prompt,
                    emotion=emotion_label,
                    emotional_mode=True  # 启用情感感知模式
                )
                response_text = dialogue_result.get('response', '')
        else:
            # 其他模式：使用对话模块
            dialogue_result = modules['dialogue'].chat(
                query=text,
                session_id=f"{session_id}_{current_mode}",
                system_prompt=system_prompt,
                emotion=emotion_label,
                emotional_mode=True  # 启用情感感知模式
            )
            response_text = dialogue_result.get('response', '')
        
        # 广播消息到网页（包含识别结果）- 在 TTS 之前执行
        emotion_label = ''
        if isinstance(emotion_result, dict):
            emotion_label = (emotion_result.get('emotion') or '').lower()
        broadcast_message('user_message', {
            'text': text, 
            'mode': current_mode, 
            'source': 'client',
            'emotion': emotion_result.get('emotion', ''),
            'emotion_score': emotion_result.get('score'),
            'speaker_id': speaker_result.get('speaker_id', ''),
            'speaker_score': speaker_result.get('similarity')
        })
        
        # 构建广播数据，包含导诊信息
        broadcast_data = {
            'text': response_text, 
            'mode': current_mode,
            'rag_used': dialogue_result.get('rag_used', False) if dialogue_result else False,
            'rag_context': dialogue_result.get('rag_context', '') if dialogue_result else '',
            'emotion': emotion_result.get('emotion', ''),
            'emotion_score': emotion_result.get('score'),
            'speaker_id': speaker_result.get('speaker_id', ''),
            'speaker_score': speaker_result.get('similarity')
        }
        if triage_result:
            broadcast_data['triage'] = {
                'department': triage_result.get('department', {}),
                'doctors': triage_result.get('doctors', []),
                'diseases': triage_result.get('diseases', [])
            }
        broadcast_message('assistant_message', broadcast_data)
        
        # 5. 流式语音合成（使用选择的音色克隆）
        # 如果回复文本为空，返回 JSON 错误
        if not response_text or not response_text.strip():
            logger.warning("Response text is empty, skipping TTS synthesis")
            return jsonify({
                "error": "Empty response text",
                "asr": asr_result,
                "emotion": emotion_result,
                "speaker": speaker_result,
                "response": ""
            })
        
        # 检查 TTS 模块
        tts_module = modules.get('tts')
        if tts_module is None:
            return jsonify({"error": "TTS module not available"}), 500
        
        # 准备响应头信息
        from urllib.parse import quote
        
        # 提取情感感知模式的 TTS 指令（用于 CosyVoice instruct 模式）
        tts_instruct = None
        if dialogue_result and dialogue_result.get('emotional_mode'):
            tts_instruct = dialogue_result.get('tts_instruct')
            style = dialogue_result.get('style', '')
            if tts_instruct:
                logger.info(f"[Emotional TTS] 使用情感风格指令: {tts_instruct}")
            if style:
                logger.info(f"[Emotional TTS] LLM 生成的风格描述: {style}")
        
        # 读取配置决定是否使用流式 TTS
        use_streaming_tts = config.get('tts', {}).get('streaming', True)
        
        # 检查是否支持流式 TTS 且配置启用
        if use_streaming_tts and hasattr(tts_module, 'synthesize_stream'):
            # 使用流式 TTS（支持音色克隆需要使用 synthesize_stream_with_clone）
            def generate_stream():
                """生成器函数，逐块返回音频数据"""
                try:
                    # 注意：synthesize_stream 目前不支持音色克隆
                    # 如果需要音色克隆，使用普通 synthesize 然后流式返回
                    if voice_clone_id:
                        # 音色克隆模式：先完整合成，再流式返回
                        logger.info(f"[Streaming TTS] Using voice clone: {voice_clone_id}")
                        if emotion_label and hasattr(tts_module, 'synthesize_with_emotion'):
                            result = tts_module.synthesize_with_emotion(
                                text=response_text,
                                emotion=emotion_label,
                                voice_clone_id=voice_clone_id
                            )
                        else:
                            result = tts_module.synthesize(
                                text=response_text, 
                                voice_clone_id=voice_clone_id,
                                instruct=tts_instruct  # 情感感知 TTS 指令
                            )
                        if result.get('output_path') and os.path.exists(result['output_path']):
                            with open(result['output_path'], 'rb') as f:
                                while True:
                                    chunk = f.read(4096)
                                    if not chunk:
                                        break
                                    yield chunk
                    else:
                        # 无音色克隆：使用真正的流式合成
                        for chunk in tts_module.synthesize_stream(response_text, None, 1.0):
                            yield chunk
                except Exception as e:
                    logger.error(f"[Streaming TTS] Error: {e}")
            
            # 构建流式响应
            from flask import Response
            response = Response(
                generate_stream(),
                mimetype='audio/wav',
                headers={
                    'Transfer-Encoding': 'chunked',
                    'X-Content-Type-Options': 'nosniff',
                    'X-ASR-Text': quote(text, safe=''),
                    'X-Response-Text': quote(response_text, safe=''),
                    'X-Emotion': quote(str(emotion_result.get('emotion', '')), safe=''),
                    'X-Speaker': quote(str(speaker_result.get('speaker_id', '')), safe=''),
                    'X-RAG-Used': str(dialogue_result.get('rag_used', False)) if dialogue_result else 'False',
                    'X-RAG-Context': quote(dialogue_result.get('rag_context', '') if dialogue_result else '', safe=''),
                    'X-Mode-Switched': 'false',
                    'X-Streaming-Audio': 'True'  # 告知客户端这是流式音频
                }
            )
            
            # 添加导诊信息到响应头
            if triage_result:
                response.headers['X-Triage-Department'] = quote(triage_result.get('department', {}).get('name', ''), safe='')
                doctors = triage_result.get('doctors', [])
                if doctors:
                    response.headers['X-Triage-Doctor'] = quote(doctors[0].get('name', ''), safe='')
            
            return response
        else:
            # 回退到普通 TTS（不支持流式的情况）
            logger.warning("TTS module does not support streaming, using regular synthesis")
            if emotion_label and hasattr(tts_module, 'synthesize_with_emotion'):
                tts_result = tts_module.synthesize_with_emotion(
                    text=response_text,
                    emotion=emotion_label,
                    voice_clone_id=voice_clone_id
                )
            else:
                tts_result = tts_module.synthesize(
                    text=response_text,
                    voice_clone_id=voice_clone_id
                )
            
            # 如果有音频文件，返回音频
            if tts_result.get('output_path') and os.path.exists(tts_result['output_path']):
                from flask import make_response
                response = make_response(send_file(
                    tts_result['output_path'],
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name='response.wav'
                ))
                # 添加自定义响应头（URL编码中文字符）
                response.headers['X-ASR-Text'] = quote(text, safe='')
                response.headers['X-Response-Text'] = quote(response_text, safe='')
                response.headers['X-Emotion'] = quote(str(emotion_result.get('emotion', '')), safe='')
                response.headers['X-Speaker'] = quote(str(speaker_result.get('speaker_id', '')), safe='')
                if dialogue_result:
                    response.headers['X-RAG-Used'] = str(dialogue_result.get('rag_used', False))
                    response.headers['X-RAG-Context'] = quote(dialogue_result.get('rag_context', ''), safe='')
                else:
                    response.headers['X-RAG-Used'] = 'False'
                    response.headers['X-RAG-Context'] = ''
                response.headers['X-Mode-Switched'] = 'false'
                response.headers['X-Streaming-Audio'] = 'False'  # 告知客户端这不是流式音频
                return response
            else:
                return jsonify({
                    "asr": asr_result,
                    "emotion": emotion_result,
                    "speaker": speaker_result,
                    "response": response_text,
                    "tts": {"output_path": None, "error": tts_result.get('error', 'Unknown error')}
                })
        
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


# ============= 临床智能 (ACI) API =============

# 导入 ACI 模块
try:
    from modules.aci.consultation_session import ConsultationSession, ConsultationManager
    from modules.aci.speaker_diarization import SpeakerDiarizer
    from modules.aci.clinical_entity_extractor import ClinicalEntityExtractor
    from modules.aci.soap_generator import SOAPGenerator
    from modules.aci.hallucination_detector import HallucinationDetector
    from modules.aci.emergency_detector import EmergencyDetector
    ACI_AVAILABLE = True
    logger.info("ACI modules loaded successfully")
except ImportError as e:
    ACI_AVAILABLE = False
    logger.warning(f"ACI modules not available: {e}")

# ACI 全局实例
consultation_manager = None
entity_extractor = None
soap_generator = None
hallucination_detector = None
emergency_detector = None
speaker_diarizer = None


def initialize_aci_modules():
    """初始化 ACI 模块"""
    global consultation_manager, entity_extractor, soap_generator
    global hallucination_detector, emergency_detector, speaker_diarizer
    
    if not ACI_AVAILABLE:
        logger.warning("ACI modules not available, skipping initialization")
        return
    
    try:
        consultation_manager = ConsultationManager()
        entity_extractor = ClinicalEntityExtractor(
            knowledge_graph=modules.get('knowledge_graph'),
            dialogue_module=modules.get('dialogue')
        )
        hallucination_detector = HallucinationDetector(
            dialogue_module=modules.get('dialogue')
        )
        soap_generator = SOAPGenerator(
            entity_extractor=entity_extractor,
            dialogue_module=modules.get('dialogue'),
            hallucination_detector=hallucination_detector
        )
        emergency_detector = EmergencyDetector()
        speaker_diarizer = SpeakerDiarizer(
            speaker_module=modules.get('speaker')
        )
        logger.info("ACI modules initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ACI modules: {e}")


@app.route('/consultation/start', methods=['POST'])
def start_consultation():
    """开始新的会诊会话"""
    if not ACI_AVAILABLE or not consultation_manager:
        return jsonify({"error": "ACI 模块未初始化"}), 500
    
    try:
        data = request.json or {}
        patient_info = data.get('patient_info', {})
        
        session = consultation_manager.create_session(patient_info=patient_info)
        
        return jsonify({
            "status": "success",
            "session_id": session.session_id,
            "message": "会诊会话已创建"
        })
        
    except Exception as e:
        logger.error(f"Start consultation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/consultation/<session_id>/register-speaker', methods=['POST'])
def register_consultation_speaker(session_id):
    """注册会诊说话人"""
    if not consultation_manager:
        return jsonify({"error": "ACI 模块未初始化"}), 500
    
    try:
        session = consultation_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "会话不存在"}), 404
        
        data = request.json
        speaker_id = data.get('speaker_id')
        role = data.get('role')  # doctor, patient, family
        name = data.get('name')
        
        if not speaker_id or not role:
            return jsonify({"error": "需要 speaker_id 和 role"}), 400
        
        session.register_speaker(speaker_id, role, name)
        
        if speaker_diarizer:
            speaker_diarizer.register_role(speaker_id, role, name)
        
        return jsonify({
            "status": "success",
            "message": f"说话人 {speaker_id} 已注册为 {role}"
        })
        
    except Exception as e:
        logger.error(f"Register speaker error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/consultation/<session_id>/utterance', methods=['POST'])
def add_consultation_utterance(session_id):
    """添加对话记录"""
    if not consultation_manager:
        return jsonify({"error": "ACI 模块未初始化"}), 500
    
    try:
        session = consultation_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "会话不存在"}), 404
        
        # 支持 JSON 或 form-data
        if request.content_type and 'multipart/form-data' in request.content_type:
            text = request.form.get('text', '')
            speaker_id = request.form.get('speaker_id')
            speaker_role = request.form.get('speaker_role')
            audio_file = request.files.get('audio')
            audio_segment = audio_file.read() if audio_file else None
        else:
            data = request.json or {}
            text = data.get('text', '')
            speaker_id = data.get('speaker_id')
            speaker_role = data.get('speaker_role')
            audio_segment = None
        
        # 如果没有角色，尝试推断
        if not speaker_role and speaker_diarizer:
            speaker_role, _ = speaker_diarizer.infer_role_from_content(text)
        
        # 提取实体
        entities = []
        if entity_extractor:
            extracted = entity_extractor.extract_entities(text, speaker_role)
            entities = [e.to_dict() for e in extracted]
        
        # 急救检测
        emergency_alert = None
        if emergency_detector:
            alert = emergency_detector.assess_risk(text)
            if alert.level in ["critical", "urgent"]:
                emergency_alert = alert.to_dict()
        
        # 添加发言
        utterance = session.add_utterance(
            text=text,
            speaker_id=speaker_id,
            speaker_role=speaker_role,
            audio_segment=audio_segment,
            entities=entities
        )
        
        response = {
            "status": "success",
            "utterance_id": utterance.id,
            "speaker_role": utterance.speaker_role,
            "entities": entities
        }
        
        if emergency_alert:
            response["emergency_alert"] = emergency_alert
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Add utterance error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/consultation/<session_id>/soap', methods=['GET'])
def get_consultation_soap(session_id):
    """获取 SOAP 病历"""
    if not consultation_manager or not soap_generator:
        return jsonify({"error": "ACI 模块未初始化"}), 500
    
    try:
        session = consultation_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "会话不存在"}), 404
        
        # 生成 SOAP
        soap = soap_generator.generate_soap(session)
        
        # 返回格式
        output_format = request.args.get('format', 'json')
        
        if output_format == 'markdown':
            return soap.to_markdown(), 200, {'Content-Type': 'text/markdown; charset=utf-8'}
        else:
            return jsonify({
                "status": "success",
                "soap": soap.to_dict()
            })
        
    except Exception as e:
        logger.error(f"Get SOAP error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/consultation/<session_id>/preview', methods=['GET'])
def get_soap_preview(session_id):
    """获取 SOAP 实时预览（轻量级）"""
    if not consultation_manager or not soap_generator:
        return jsonify({"error": "ACI 模块未初始化"}), 500
    
    try:
        session = consultation_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "会话不存在"}), 404
        
        preview = soap_generator.generate_realtime_preview(session)
        preview["transcript"] = session.get_transcript(include_roles=True)
        preview["statistics"] = session.get_statistics()
        
        return jsonify({
            "status": "success",
            "preview": preview
        })
        
    except Exception as e:
        logger.error(f"Get SOAP preview error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/consultation/<session_id>/end', methods=['POST'])
def end_consultation(session_id):
    """结束会诊会话"""
    if not consultation_manager:
        return jsonify({"error": "ACI 模块未初始化"}), 500
    
    try:
        session = consultation_manager.end_session(session_id, save=True)
        if not session:
            return jsonify({"error": "会话不存在"}), 404
        
        return jsonify({
            "status": "success",
            "message": "会诊已结束",
            "statistics": session.get_statistics()
        })
        
    except Exception as e:
        logger.error(f"End consultation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/emergency/assess', methods=['POST'])
def assess_emergency():
    """评估急救风险"""
    if not emergency_detector:
        return jsonify({"error": "急救检测模块未初始化"}), 500
    
    try:
        data = request.json
        text = data.get('text', '')
        audio_features = data.get('audio_features')
        
        alert = emergency_detector.assess_risk(text, audio_features)
        
        response = {
            "status": "success",
            "alert": alert.to_dict()
        }
        
        # 如果是危急级别，添加急救模式响应
        if alert.level == "critical":
            response["emergency_mode"] = emergency_detector.trigger_emergency_mode(alert)
            response["first_aid"] = emergency_detector.get_first_aid_guidance(
                alert.triggers[0] if alert.triggers else "general"
            )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Assess emergency error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/aci/generate-soap', methods=['POST'])
def generate_soap_from_dialogue():
    """
    从对话记录生成 SOAP 病历
    
    输入格式1 - 结构化:
    {
        "utterances": [
            {"speaker": "医生", "text": "您哪里不舒服？"},
            {"speaker": "患者", "text": "我头疼了三天，还有点发烧。"}
        ]
    }
    
    输入格式2 - 文本:
    {
        "dialogue_text": "患者：我头疼了三天，还有点发烧。\n医生：您有没有其他症状？"
    }
    """
    try:
        data = request.json
        utterances = data.get('utterances', [])
        dialogue_text = data.get('dialogue_text', '')
        
        # 如果提供了原始文本格式，解析为结构化格式
        if dialogue_text and not utterances:
            utterances = _parse_dialogue_text(dialogue_text)
        
        if not utterances:
            return jsonify({"error": "缺少对话记录，请提供 utterances 参数"}), 400
        
        # 创建一个临时会话来存储对话
        from modules.aci.consultation_session import ConsultationSession
        session = ConsultationSession()
        
        # 角色映射
        role_map = {
            '医生': 'doctor',
            '患者': 'patient',
            '家属': 'family'
        }
        
        # 添加对话记录
        for i, utt in enumerate(utterances):
            speaker = utt.get('speaker', '患者')
            text = utt.get('text', '')
            role = role_map.get(speaker, 'patient')
            session.add_utterance(
                text=text,
                speaker_id=f"speaker_{role}",
                speaker_role=role,
                timestamp=float(i)
            )
        
        # 提取实体
        entities = []
        if entity_extractor:
            for utt in utterances:
                extracted = entity_extractor.extract_entities(
                    utt.get('text', ''),
                    speaker_role=role_map.get(utt.get('speaker', '患者'), 'patient')
                )
                entities.extend([e.to_dict() for e in extracted])
        
        # 生成 SOAP - 直接使用 LLM 生成
        print(f"\n[DEBUG] 开始SOAP生成，对话轮数: {len(utterances)}")
        soap_result = _generate_simple_soap(utterances, entities)
        
        return jsonify({
            "status": "success",
            "soap": soap_result
        })
        
    except Exception as e:
        logger.error(f"Generate SOAP error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _parse_dialogue_text(dialogue_text: str) -> list:
    """
    解析对话文本为结构化格式
    
    支持的格式:
    - 患者：我头疼了三天
    - 医生: 有没有发烧？
    - 家属：他昨天开始的
    
    每行一句，或者用换行符分隔
    """
    import re
    
    utterances = []
    
    # 按行分割
    lines = dialogue_text.strip().split('\n')
    
    # 支持的角色前缀
    role_patterns = [
        (r'^患者[：:\s]+(.+)$', '患者'),
        (r'^病人[：:\s]+(.+)$', '患者'),
        (r'^医生[：:\s]+(.+)$', '医生'),
        (r'^大夫[：:\s]+(.+)$', '医生'),
        (r'^家属[：:\s]+(.+)$', '家属'),
        (r'^护士[：:\s]+(.+)$', '医生'),  # 护士归类为医疗方
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        matched = False
        for pattern, speaker in role_patterns:
            match = re.match(pattern, line)
            if match:
                text = match.group(1).strip()
                if text:
                    utterances.append({
                        'speaker': speaker,
                        'text': text
                    })
                matched = True
                break
        
        # 如果没有匹配到角色前缀，默认为患者
        if not matched and line:
            utterances.append({
                'speaker': '患者',
                'text': line
            })
    
    return utterances


def _generate_simple_soap(utterances, entities):
    """
    完全使用大模型生成 SOAP 病历和提取实体
    """
    dialogue_module = modules.get('dialogue')
    
    # 构建对话转录
    transcript_lines = []
    for utt in utterances:
        speaker = utt.get('speaker', '患者')
        text = utt.get('text', '')
        transcript_lines.append(f"{speaker}：{text}")
    transcript = "\n".join(transcript_lines)
    
    logger.info(f"[SOAP] 开始生成，对话轮数: {len(utterances)}, LLM可用: {dialogue_module is not None}")
    
    if dialogue_module and len(utterances) > 0:
        try:
            # 简化的提示词，让模型更容易遵循
            prompt = f"""分析以下医患对话，生成病历报告。

对话内容：
{transcript}

请直接输出以下格式的病历（不要输出其他内容）：

主诉：[患者的主要症状和持续时间]
现病史：[症状发展过程的描述]
生命体征：[体温、血压等检查数据，如果对话中没有就写"待检查"]
体格检查：[医生检查发现，如果没有就写"待检查"]
诊断：[医生给出的诊断，如果没有就写"待诊断"]
病情评估：[对病情的分析]
治疗方案：[开的药物和治疗方法]
医嘱：[医生的建议和随访要求]
症状：[从对话中提取的所有症状，用逗号分隔]
疾病：[从对话中提取的疾病名称，用逗号分隔]
药物：[从对话中提取的药物名称，用逗号分隔]"""

            result = dialogue_module.chat(
                query=prompt,
                session_id="soap_generation",
                reset=True,
                use_rag=False
            )
            
            response_text = result.get('response', '')
            logger.info(f"[SOAP] LLM响应长度: {len(response_text)}")
            print(f"\n{'='*50}")
            print(f"[SOAP] LLM响应内容:")
            print(response_text)
            print(f"{'='*50}\n")
            
            if response_text:
                # 解析文本格式的响应
                soap_result = _parse_soap_text(response_text)
                soap_result['generated_by'] = 'llm'
                logger.info("[SOAP] 成功使用大模型生成病历")
                return soap_result
                
        except Exception as e:
            logger.error(f"[SOAP] LLM生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 回退：从对话中提取基本信息
    logger.info("[SOAP] 使用回退规则生成")
    patient_texts = [utt.get('text', '') for utt in utterances if utt.get('speaker') == '患者']
    doctor_texts = [utt.get('text', '') for utt in utterances if utt.get('speaker') == '医生']
    
    return {
        'subjective': {
            'chief_complaint': patient_texts[0] if patient_texts else '未记录',
            'history': ' '.join(patient_texts[1:3]) if len(patient_texts) > 1 else ''
        },
        'objective': {
            'vital_signs': '待检查',
            'content': '待检查'
        },
        'assessment': {
            'diagnosis': '待诊断',
            'content': '待评估'
        },
        'plan': {
            'treatment': '待制定',
            'content': ''
        },
        'entities': entities,
        'generated_by': 'rules'
    }


def _parse_soap_text(text):
    """解析LLM生成的文本格式SOAP"""
    result = {
        'subjective': {'chief_complaint': '', 'history': ''},
        'objective': {'vital_signs': '', 'content': ''},
        'assessment': {'diagnosis': '', 'content': ''},
        'plan': {'treatment': '', 'content': ''},
        'entities': []
    }
    
    # 提取各字段 - 支持逗号/句号/换行分隔的格式
    # LLM可能返回 "主诉，头痛..." 或 "主诉：头痛..." 格式
    patterns = {
        'chief_complaint': r'主诉[，,：:]\s*(.+?)(?=现病史|生命体征|体格检查|诊断|$)',
        'history': r'现病史[，,：:]\s*(.+?)(?=生命体征|体格检查|诊断|病情评估|$)',
        'vital_signs': r'生命体征[，,：:]\s*(.+?)(?=体格检查|诊断|病情评估|$)',
        'physical_exam': r'体格检查[，,：:]\s*(.+?)(?=诊断|病情评估|治疗|$)',
        'diagnosis': r'诊断[，,：:]\s*(.+?)(?=病情评估|治疗方案|医嘱|$)',
        'assessment_content': r'病情评估[，,：:]\s*(.+?)(?=治疗方案|医嘱|症状|$)',
        'treatment': r'治疗方案[，,：:]\s*(.+?)(?=医嘱|症状|疾病|$)',
        'instructions': r'医嘱[，,：:]\s*(.+?)(?=症状|疾病|药物|$)',
        'symptoms': r'症状[，,：:]\s*(.+?)(?=疾病|药物|$)',
        'diseases': r'疾病[，,：:]\s*(.+?)(?=药物|$)',
        'medications': r'药物[，,：:]\s*(.+?)$'
    }
    
    import re
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == 'chief_complaint':
                result['subjective']['chief_complaint'] = value
            elif key == 'history':
                result['subjective']['history'] = value
            elif key == 'vital_signs':
                result['objective']['vital_signs'] = value
            elif key == 'physical_exam':
                result['objective']['content'] = value
            elif key == 'diagnosis':
                result['assessment']['diagnosis'] = value
            elif key == 'assessment_content':
                result['assessment']['content'] = value
            elif key == 'treatment':
                result['plan']['treatment'] = value
            elif key == 'instructions':
                result['plan']['content'] = value
            elif key == 'symptoms':
                for s in value.split('，'):
                    s = s.strip().replace('、', '').replace(',', '')
                    if s and len(s) >= 2:
                        result['entities'].append({'type': 'symptom', 'text': s})
            elif key == 'diseases':
                for d in value.split('，'):
                    d = d.strip().replace('、', '').replace(',', '')
                    if d and len(d) >= 2:
                        result['entities'].append({'type': 'disease', 'text': d})
            elif key == 'medications':
                for m in value.split('，'):
                    m = m.strip().replace('、', '').replace(',', '')
                    if m and len(m) >= 2:
                        result['entities'].append({'type': 'medication', 'text': m})
    
    return result


@app.route('/aci/status', methods=['GET'])
def aci_status():
    """获取 ACI 模块状态"""
    return jsonify({
        "available": ACI_AVAILABLE,
        "modules": {
            "consultation_manager": consultation_manager is not None,
            "entity_extractor": entity_extractor is not None,
            "soap_generator": soap_generator is not None,
            "hallucination_detector": hallucination_detector is not None,
            "emergency_detector": emergency_detector is not None,
            "speaker_diarizer": speaker_diarizer is not None
        }
    })


if __name__ == '__main__':
    # 初始化所有模块
    initialize_modules()
    
    # 初始化 ACI 模块
    initialize_aci_modules()
    
    # 启动服务器
    server_config = config.get('server', {})
    app.run(
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 5000),
        debug=server_config.get('debug', False)
    )

