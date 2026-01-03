# Medical modules - 医疗功能模块
from .triage import TriageModule
from .diagnosis_assistant import DiagnosisAssistant
from .medication import MedicationModule
from .medical_dict import MedicalDictionary
from .intent_classifier import IntentClassifier

__all__ = [
    'TriageModule',
    'DiagnosisAssistant',
    'MedicationModule',
    'MedicalDictionary',
    'IntentClassifier'
]
