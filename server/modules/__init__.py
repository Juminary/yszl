"""
医疗语音助手 - 模块包

目录结构:
├── core/       - 核心功能 (ASR, TTS, Dialogue, RAG)
├── audio/      - 音频分析 (Emotion, Speaker, Paralinguistic, SoundEvent)
├── medical/    - 医疗功能 (Triage, Diagnosis, Medication, MedicalDict, Intent)
├── aci/        - 临床智能 (Consultation, SOAP, Emergency, etc.)
└── knowledge/  - 知识库 (KnowledgeGraph, Cypher)
"""

# Core modules
from .core.asr import ASRModule
from .core.tts import TTSModule
from .core.dialogue import DialogueModule, SimplDialogueModule
from .core.rag import RAGModule

# Audio modules
from .audio.emotion import EmotionModule
from .audio.speaker import SpeakerModule
from .audio.paralinguistic import ParalinguisticAnalyzer
from .audio.sound_event import SoundEventDetector

# Medical modules
from .medical.triage import TriageModule
from .medical.diagnosis_assistant import DiagnosisAssistant
from .medical.medication import MedicationModule
from .medical.medical_dict import MedicalDictionary
from .medical.intent_classifier import IntentClassifier

# ACI modules (Ambient Clinical Intelligence)
from .aci.consultation_session import ConsultationSession, ConsultationManager
from .aci.speaker_diarization import SpeakerDiarizer
from .aci.clinical_entity_extractor import ClinicalEntityExtractor
from .aci.soap_generator import SOAPGenerator
from .aci.hallucination_detector import HallucinationDetector
from .aci.emergency_detector import EmergencyDetector

# Knowledge modules
from .knowledge.knowledge_graph import KnowledgeGraphModule
from .knowledge.cypher_generator import CypherGenerator

__all__ = [
    # Core
    'ASRModule',
    'TTSModule',
    'DialogueModule',
    'SimplDialogueModule',
    'RAGModule',
    # Audio
    'EmotionModule',
    'SpeakerModule',
    'ParalinguisticAnalyzer',
    'SoundEventDetector',
    # Medical
    'TriageModule',
    'DiagnosisAssistant',
    'MedicationModule',
    'MedicalDictionary',
    'IntentClassifier',
    # ACI
    'ConsultationSession',
    'ConsultationManager',
    'SpeakerDiarizer',
    'ClinicalEntityExtractor',
    'SOAPGenerator',
    'HallucinationDetector',
    'EmergencyDetector',
    # Knowledge
    'KnowledgeGraphModule',
    'CypherGenerator',
]
