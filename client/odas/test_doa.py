#!/usr/bin/env python3
"""
DOA 功能快速测试脚本

使用方法:
1. 确保 ODAS 已安装: ./start_odas.sh install
2. 运行测试: python test_doa.py

脚本会:
1. 设置麦克风增益
2. 启动 ODAS 客户端（监听）
3. 启动 ODAS 进程
4. 实时显示声源方向
"""

import sys
import os
import time
import logging
import subprocess
from pathlib import Path

# 添加父目录以导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from acoustic_frontend.odas_client import ODASClient, TrackedSource

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_card_id() -> int:
    """检测 ReSpeaker 声卡 ID"""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'seeed' in line.lower():
                import re
                match = re.search(r'card (\d+)', line)
                if match:
                    return int(match.group(1))
    except Exception:
        pass
    return 3  # 默认


def set_mic_gain(card_id: int, adc_gain: int = 8, digital_vol: int = 160):
    """设置麦克风增益"""
    print(f"设置麦克风增益: ADC={adc_gain}, Digital={digital_vol}")
    for i in range(1, 9):
        subprocess.run(
            ['amixer', '-c', str(card_id), 'cset', f'name=ADC{i} PGA gain', str(adc_gain)],
            capture_output=True
        )
        subprocess.run(
            ['amixer', '-c', str(card_id), 'cset', f'name=CH{i} digital volume', str(digital_vol)],
            capture_output=True
        )
    print("✅ 增益设置完成")


def create_runtime_config(card_id: int) -> str:
    """创建运行时配置"""
    config_path = Path(__file__).parent / "respeaker_6mic.cfg"
    runtime_path = '/tmp/odas_runtime.cfg'
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    import re
    content = re.sub(r'card = \d+;', f'card = {card_id};', content)
    
    with open(runtime_path, 'w') as f:
        f.write(content)
    
    return runtime_path


def main():
    print("=" * 60)
    print("DOA 功能快速测试")
    print("=" * 60)
    
    # 1. 检测声卡
    card_id = detect_card_id()
    print(f"检测到声卡 ID: {card_id}")
    
    # 2. 设置增益
    set_mic_gain(card_id)
    
    # 3. 启动 ODAS 客户端（先启动，作为服务器等待连接）
    print("\n启动 ODAS 客户端...")
    client = ODASClient(sst_port=9000, sss_port=9001)
    
    def on_source_active(source: TrackedSource):
        print(f"\n🎤 声源激活: ID={source.id}, 方位角={source.azimuth:.1f}°, activity={source.activity:.3f}")
    
    def on_doa_update(source: TrackedSource):
        print(f"🎯 DOA更新: {source.azimuth:.1f}°")
    
    client.on("on_source_active", on_source_active)
    client.on("on_doa_update", on_doa_update)
    
    client.start()
    print("✅ ODAS 客户端已启动 (等待 ODAS 连接...)")
    
    time.sleep(0.5)
    
    # 4. 启动 ODAS 进程
    print("\n启动 ODAS 进程...")
    odas_binary = Path(__file__).parent / "odas_build" / "build" / "bin" / "odaslive"
    
    if not odas_binary.exists():
        print(f"❌ ODAS 未安装: {odas_binary}")
        print("请运行: ./start_odas.sh install")
        client.stop()
        return
    
    # 杀掉旧进程
    subprocess.run(['pkill', '-f', 'odaslive'], capture_output=True)
    time.sleep(0.5)
    
    # 创建运行时配置
    runtime_config = create_runtime_config(card_id)
    
    # 启动 ODAS
    log_file = open('/tmp/odas.log', 'w')
    odas_process = subprocess.Popen(
        [str(odas_binary), '-c', runtime_config],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    
    time.sleep(1.5)
    
    if odas_process.poll() is not None:
        print(f"❌ ODAS 启动失败，请检查 /tmp/odas.log")
        client.stop()
        return
    
    print(f"✅ ODAS 已启动 (PID: {odas_process.pid})")
    
    # 5. 等待连接
    print("\n等待 ODAS 连接...")
    for _ in range(10):
        if client.is_connected():
            print("✅ ODAS 已连接!")
            break
        time.sleep(0.5)
    else:
        print("⚠️ 等待连接超时，继续运行...")
    
    # 6. 实时显示
    print("\n" + "=" * 60)
    print("实时声源定位 (按 Ctrl+C 退出)")
    print("=" * 60)
    print("对着麦克风说话或拍手，观察方位角变化")
    print()
    
    try:
        frame = 0
        max_activity = [0.0] * 4
        
        while True:
            frame += 1
            all_sources = client.get_tracked_sources(active_only=False)
            active_sources = client.get_tracked_sources(active_only=True)
            
            # 更新最大 activity
            for i, s in enumerate(all_sources):
                if i < len(max_activity) and s.activity > max_activity[i]:
                    max_activity[i] = s.activity
            
            # 显示
            if active_sources:
                print(f"\n🎤 检测到 {len(active_sources)} 个活跃声源:")
                for s in active_sources:
                    print(f"   声源 {s.id}: 方位角={s.azimuth:.1f}°, activity={s.activity:.3f}")
            else:
                activities = [f"{s.activity:.3f}" for s in all_sources]
                print(f"\r[帧 {frame}] activity: {activities} (最大: {[f'{m:.3f}' for m in max_activity]})    ", end="", flush=True)
            
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print(f"\n\n历史最大 activity: {max(max_activity):.3f}")
    finally:
        print("停止...")
        odas_process.terminate()
        odas_process.wait(timeout=3)
        client.stop()
        print("完成!")


if __name__ == "__main__":
    main()


