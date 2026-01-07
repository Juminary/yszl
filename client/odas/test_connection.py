import sys
import os
import time
import logging

# 将父目录加入路径以导入 acoustic_frontend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acoustic_frontend.odas_client import ODASClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    # 创建客户端 (Server 模式)
    client = ODASClient(sst_port=9000)
    
    logger.info("Starting ODAS Client (Waiting for connection on port 9000)...")
    client.start()
    
    logger.info("You can now start ODAS using: ./start_odas.sh start")
    
    try:
        while True:  # 持续运行直到手动停止
            if client.is_connected():
                sources = client.get_tracked_sources()
                if sources:
                    print(f"\rDetected {len(sources)} sources. Primary DOA: {sources[0].azimuth:.1f}°    ", end="")
                else:
                    print("\rConnected, but no active sources...    ", end="")
            else:
                print("\rWaiting for ODAS to connect...          ", end="")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        client.stop()

if __name__ == "__main__":
    test_connection()
