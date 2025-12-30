"""
API测试脚本
用于测试服务器各个接口
"""

import requests
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APITester:
    """API测试类"""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """
        初始化测试器
        
        Args:
            server_url: 服务器地址
        """
        self.server_url = server_url
        logger.info(f"API Tester initialized: {server_url}")
    
    def test_health(self):
        """测试健康检查接口"""
        logger.info("Testing /health endpoint...")
        
        try:
            response = requests.get(f"{self.server_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Health check passed")
                logger.info(f"  Status: {data.get('status')}")
                logger.info(f"  Modules: {data.get('modules')}")
                return True
            else:
                logger.error(f"✗ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Health check error: {e}")
            return False
    
    def test_asr(self, audio_path: str):
        """测试语音识别接口"""
        logger.info(f"Testing /asr endpoint with {audio_path}...")
        
        try:
            if not Path(audio_path).exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{self.server_url}/asr", files=files)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ ASR test passed")
                logger.info(f"  Text: {data.get('text')}")
                logger.info(f"  Language: {data.get('language')}")
                logger.info(f"  Confidence: {data.get('confidence')}")
                return True
            else:
                logger.error(f"✗ ASR test failed: {response.status_code}")
                logger.error(f"  {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ ASR test error: {e}")
            return False
    
    def test_emotion(self, audio_path: str):
        """测试情感识别接口"""
        logger.info(f"Testing /emotion endpoint with {audio_path}...")
        
        try:
            if not Path(audio_path).exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{self.server_url}/emotion", files=files)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Emotion test passed")
                logger.info(f"  Emotion: {data.get('emotion')} ({data.get('emotion_zh')})")
                logger.info(f"  Confidence: {data.get('confidence')}")
                logger.info(f"  Prosody: {data.get('prosody')}")
                return True
            else:
                logger.error(f"✗ Emotion test failed: {response.status_code}")
                logger.error(f"  {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Emotion test error: {e}")
            return False
    
    def test_speaker_register(self, audio_path: str, speaker_id: str):
        """测试声纹注册接口"""
        logger.info(f"Testing /speaker/register with {audio_path}, speaker_id={speaker_id}...")
        
        try:
            if not Path(audio_path).exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                data = {'speaker_id': speaker_id}
                response = requests.post(
                    f"{self.server_url}/speaker/register",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ Speaker registration passed")
                logger.info(f"  Status: {result.get('status')}")
                logger.info(f"  Action: {result.get('action')}")
                logger.info(f"  Samples: {result.get('num_samples')}")
                return True
            else:
                logger.error(f"✗ Speaker registration failed: {response.status_code}")
                logger.error(f"  {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Speaker registration error: {e}")
            return False
    
    def test_speaker_recognize(self, audio_path: str):
        """测试声纹识别接口"""
        logger.info(f"Testing /speaker/recognize with {audio_path}...")
        
        try:
            if not Path(audio_path).exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(f"{self.server_url}/speaker/recognize", files=files)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Speaker recognition passed")
                logger.info(f"  Speaker: {data.get('speaker_id')}")
                logger.info(f"  Similarity: {data.get('similarity')}")
                logger.info(f"  Recognized: {data.get('recognized')}")
                return True
            else:
                logger.error(f"✗ Speaker recognition failed: {response.status_code}")
                logger.error(f"  {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Speaker recognition error: {e}")
            return False
    
    def test_dialogue(self, query: str, session_id: str = "test"):
        """测试对话接口"""
        logger.info(f"Testing /dialogue with query: {query}...")
        
        try:
            data = {
                'query': query,
                'session_id': session_id
            }
            response = requests.post(
                f"{self.server_url}/dialogue",
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ Dialogue test passed")
                logger.info(f"  Query: {result.get('query')}")
                logger.info(f"  Response: {result.get('response')}")
                return True
            else:
                logger.error(f"✗ Dialogue test failed: {response.status_code}")
                logger.error(f"  {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Dialogue test error: {e}")
            return False
    
    def test_tts(self, text: str, output_path: str = "test_tts_output.wav"):
        """测试语音合成接口"""
        logger.info(f"Testing /tts with text: {text}...")
        
        try:
            data = {'text': text}
            response = requests.post(
                f"{self.server_url}/tts",
                json=data
            )
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✓ TTS test passed")
                logger.info(f"  Output saved to: {output_path}")
                return True
            else:
                logger.error(f"✗ TTS test failed: {response.status_code}")
                logger.error(f"  {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ TTS test error: {e}")
            return False
    
    def test_info(self):
        """测试系统信息接口"""
        logger.info("Testing /info endpoint...")
        
        try:
            response = requests.get(f"{self.server_url}/info")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ Info test passed")
                logger.info(f"  Config: {json.dumps(data.get('config', {}), indent=2)}")
                logger.info(f"  Modules: {json.dumps(data.get('modules', {}), indent=2)}")
                return True
            else:
                logger.error(f"✗ Info test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Info test error: {e}")
            return False
    
    def run_all_tests(self, test_audio: str = None):
        """运行所有测试"""
        print("\n" + "="*60)
        print("API 测试套件")
        print("="*60 + "\n")
        
        results = {}
        
        # 1. 健康检查
        results['health'] = self.test_health()
        print()
        
        # 2. 系统信息
        results['info'] = self.test_info()
        print()
        
        # 3. 对话测试
        results['dialogue'] = self.test_dialogue("你好")
        print()
        
        # 4. 语音合成测试
        results['tts'] = self.test_tts("这是一个测试")
        print()
        
        # 如果提供了测试音频，测试其他接口
        if test_audio and Path(test_audio).exists():
            results['asr'] = self.test_asr(test_audio)
            print()
            
            results['emotion'] = self.test_emotion(test_audio)
            print()
            
            results['speaker_register'] = self.test_speaker_register(test_audio, "test_user")
            print()
            
            results['speaker_recognize'] = self.test_speaker_recognize(test_audio)
            print()
        else:
            logger.warning("No test audio provided, skipping audio-related tests")
        
        # 显示结果
        print("\n" + "="*60)
        print("测试结果:")
        print("="*60)
        for test_name, passed in results.items():
            status = "✓" if passed else "✗"
            print(f"{status} {test_name}")
        print("="*60)
        
        total = len(results)
        passed = sum(results.values())
        print(f"\n通过: {passed}/{total}")
        
        return all(results.values())


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API Testing Tool')
    parser.add_argument('--server', type=str, default='http://localhost:5000',
                       help='服务器地址')
    parser.add_argument('--audio', type=str, help='测试音频文件路径')
    
    args = parser.parse_args()
    
    tester = APITester(server_url=args.server)
    tester.run_all_tests(test_audio=args.audio)


if __name__ == "__main__":
    main()
