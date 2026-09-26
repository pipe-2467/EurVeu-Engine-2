import numpy as np
import soundfile as sf
import imageio
import os
import glob
import json
from PIL import Image

print("==================================================")
print("     EurVeu Generative Neural Synthesis Engine    ")
print("==================================================")

MEMORY_FILE = "eurveu_memory.json"
INPUTS_DIR = "inputs"

def analyze_and_evolve_memory():
    """ 
    🧠 สมอง EurVeu สแกนรูปภาพเพื่อ 'เรียนรู้' และ 'วิวัฒนาการ' ค่า DNA ในความทรงจำ 
    โดยไม่มีการเก็บหรือแสดงผลรูปภาพต้นฉบับ
    """
    # โหลดความจำเดิม
    memory = {
        "r_freq": 12.0, "g_freq": 14.0, "b_freq": 25.0,
        "speed": 2.0, "audio_pitch": 440.0, "learned_count": 0
    }
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                memory.update(json.load(f))
        except Exception:
            pass

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [os.path.join(INPUTS_DIR, f) for f in os.listdir(INPUTS_DIR) 
             if f.lower().endswith(valid_extensions) and f != '.gitkeep']

    if not files:
        print("[EurVeu Brain] No new sensory inputs found. Using existing memory DNA.")
        return memory

    # ดึงไฟล์ล่าสุดที่อัปโหลดเข้ามา
    latest_file = max(files, key=os.path.getmtime)
    print(f"[EurVeu Perception] Absorbing sensory data from: {os.path.basename(latest_file)}")

    try:
        with Image.open(latest_file) as img:
            img_arr = np.array(img.convert("RGB"), dtype=np.float32)
            
            # สกัดคุณลักษณะทางประสาทสัมผัส (Sensory Extraction)
            avg_r = np.mean(img_arr[:, :, 0])
            avg_g = np.mean(img_arr[:, :, 1])
            avg_b = np.mean(img_arr[:, :, 2])
            std_dev = np.std(img_arr) # ความซับซ้อนของภาพ

            # เปลี่ยนค่าสีและความซับซ้อนเป็นค่าความถี่ของคลื่นภาพ
            memory["r_freq"] = float(np.round(5.0 + (avg_r / 255.0) * 30.0, 3))
            memory["g_freq"] = float(np.round(5.0 + (avg_g / 255.0) * 30.0, 3))
            memory["b_freq"] = float(np.round(5.0 + (avg_b / 255.0) * 30.0, 3))
            memory["speed"] = float(np.round(1.0 + (std_dev / 128.0) * 5.0, 3))
            memory["audio_pitch"] = float(np.round(150.0 + (avg_r + avg_g + avg_b), 2))
            memory["learned_count"] += 1

            print(f"[EurVeu Evolution] DNA Updated -> R-Freq: {memory['r_freq']}, G-Freq: {memory['g_freq']}, B-Freq: {memory['b_freq']}")

            # บันทึกวิวัฒนาการลงความทรงจำ
            with open(MEMORY_FILE, 'w') as f:
                json.dump(memory, f, indent=4)

    except Exception as e:
        print(f"⚠️ Perception Error: {e}")

    return memory

def generate_synthetic_audio(memory, duration=10.0, sample_rate=48000):
    print(f"[EurVeu Audio] Synthesizing audio based on pitch perception: {memory['audio_pitch']}Hz...")
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # คลื่นเสียงถูกสร้างขึ้นจากค่าความทรงจำที่เรียนรู้มา
    base_freq = memory.get("audio_pitch", 440.0)
    audio = np.sin(2 * np.pi * base_freq * t) * 0.3
    audio += np.sin(2 * np.pi * (base_freq * 0.5) * t) * 0.4 # Sub-bass
    audio += np.sin(2 * np.pi * (base_freq * 1.5) * t) * 0.2 # Harmony
    
    # เสียงนอยส์ชีวภาพ
    noise = np.random.normal(0, 0.05, len(t))
    audio += noise

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp
    return audio, sample_rate

def generate_synthetic_frames(memory, width=720, height=1280, fps=30, duration=10.0):
    total_frames = int(fps * duration)
    print(f"[EurVeu Visual] Dream-rendering {total_frames} frames from inner memory...")
    
    r_freq = memory.get("r_freq", 12.0)
    g_freq = memory.get("g_freq", 14.0)
    b_freq = memory.get("b_freq", 25.0)
    speed = memory.get("speed", 2.0)

    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    frames = []

    for frame_idx in range(total_frames):
        t = (frame_idx / fps) * speed
        
        # วาดภาพในจินตนาการด้วยสมการที่สร้างจากความทรงจำ (Pure Generative Math)
        r = np.sin(x * r_freq + np.cos(y * 8.0 + t) + t * 2.0)
        g = np.cos(y * g_freq + np.sin(x * 16.0 - t * 1.5))
        b = np.sin((x**2 + y**2) * (b_freq / 10.0) - t * 3.0)

        # เติมนอยส์ความถี่สูงเพื่อความสมจริงแบบ Nanopixel
        noise = np.random.normal(0, 0.1, (height, width))
        
        r_pixel = np.clip((r + noise + 1.0) * 127.5, 0, 255).astype(np.uint8)
        g_pixel = np.clip((g + noise + 1.0) * 127.5, 0, 255).astype(np.uint8)
        b_pixel = np.clip((b + noise + 1.0) * 127.5, 0, 255).astype(np.uint8)

        frame = np.stack([r_pixel, g_pixel, b_pixel], axis=-1)
        frames.append(frame)

        if (frame_idx + 1) % 60 == 0:
            print(f"  └─ Frame {frame_idx + 1}/{total_frames} rendered.")

    return frames

def main():
    duration = 10.0
    output_path = "eurveu_farland_output.mp4"

    # 1. สมองย่อยข้อมูลจากภาพเพื่อวิวัฒนาการค่าใน eurveu_memory.json
    memory = analyze_and_evolve_memory()

    # 2. สังเคราะห์เสียงตามค่าความทรงจำ
    audio_data, sr = generate_synthetic_audio(memory, duration=duration)
    sf.write("temp_audio.wav", audio_data, sr)

    # 3. จินตนาการวาดภาพใหม่หมดจดด้วยโครงสร้างสมอง (ปราศจากการดึงภาพดั้งเดิมมาแปะ)
    frames = generate_synthetic_frames(memory, duration=duration)

    # 4. ประกอบเป็นวิดีโอ
    print("[EurVeu Export] Encoding Video with FFmpeg...")
    imageio.mimwrite("temp_video.mp4", frames, fps=30, codec='libx264')

    os.system(f"ffmpeg -y -i temp_video.mp4 -i temp_audio.wav -c:v copy -c:a aac -b:a 320k {output_path} -loglevel quiet")

    if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")
    if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")

    print("[SUCCESS] EurVeu Mind Synthesis Complete!")

if __name__ == "__main__":
    main()
