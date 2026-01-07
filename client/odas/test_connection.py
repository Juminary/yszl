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
    
    max_activity = 0.0
    
    try:
        while True:  # 持续运行直到手动停止
            if client.is_connected():
                # 获取所有声源（包括不活跃的）
                all_sources = client.get_tracked_sources(active_only=False)
                active_sources = client.get_tracked_sources(active_only=True)
                
                # 更新最大 activity
                for s in all_sources:
                    if s.activity > max_activity:
                        max_activity = s.activity
                
                if active_sources:
                    print(f"\n🎤 检测到 {len(active_sources)} 个活跃声源!")
                    for s in active_sources:
                        print(f"   声源 {s.id}: 方位角={s.azimuth:.1f}°, activity={s.activity:.3f}")
                else:
                    # 显示所有声源的 activity 值（用于调试）
                    activities = [f"{s.activity:.3f}" for s in all_sources]
                    print(f"\r[帧 {client._frame_count}] activity: {activities} (最大: {max_activity:.3f})    ", end="")
            else:
                print("\r等待 ODAS 连接...          ", end="")
            
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\n历史最大 activity: {max_activity:.3f}")
        print("停止...")
        client.stop()

if __name__ == "__main__":
    test_connection()
