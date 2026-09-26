import numpy as np
import soundfile as sf
import imageio
import os
import glob
from PIL import Image
from eurveu_brain import EurVeuBrain

print("==================================================")
print("     EurVeu Autonomous Cloud Render Engine v2     ")
print("==================================================")

def load_latest_input_image(inputs_dir="inputs", target_size=(720, 1280)):
    """ ดึงภาพหรือเฟรมวิดีโอล่าสุดจากโฟลเดอร์ inputs/ เพื่อนำมาสร้างวิดีโอ """
    width, height = target_size
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    files = glob.glob(os.path.join(inputs_dir, "*"))
    image_files = [f for f in files if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("[EurVeu Input] No user image found. Generating pure synthetic base.")
        return None

    # เรียงลำดับตามเวลาอัปโหลดล่าสุด (Most Recent File)
    latest_file = max(image_files, key=os.path.getmtime)
    print(f"[EurVeu Input] Ingesting latest input: {os.path.basename(latest_file)}")
    
    try:
        with Image.open(latest_file) as img:
            img = img.convert("RGB")
            img = img.resize((width, height))
            return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Error loading image {latest_file}: {e}")
        return None

def generate_audio(brain_vectors, duration=10.0, sample_rate=48000):
    num_layers = int(brain_vectors.get("audio_density", 20000))
    print(f"[EurVeu Audio] Synthesizing {num_layers:,} frequency waves...")
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)

    freqs = np.random.uniform(20.0, 22000.0, num_layers)
    phases = np.random.uniform(0, 2 * np.pi, num_layers)
    amps = np.random.exponential(0.05, num_layers)

    for i in range(min(num_layers, 5000)): # ออพติไมซ์เพื่อความเร็ว
        audio += amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp
    return audio, sample_rate

def generate_frames(brain_vectors, input_image=None, width=720, height=1280, fps=30, duration=10.0):
    total_frames = int(fps * duration)
    print(f"[EurVeu Visual] Rendering {total_frames} Nanopixel frames ({width}x{height})...")
    
    # ดึงค่าพารามิเตอร์ที่วิวัฒนาการมาจากสมอง EurVeu
    r_freq = brain_vectors.get("r_freq", 12.0)
    g_freq = brain_vectors.get("g_freq", 14.0)
    b_freq = brain_vectors.get("b_freq", 25.0)

    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    frames = []

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        
        # สมการสังเคราะห์ Farland Noise ที่ปรับค่าตามการเรียนรู้ของสมอง
        r_wave = np.sin(x * r_freq + np.cos(y * 8.0 + t) + t * 2.0)
        g_wave = np.cos(y * g_freq + np.sin(x * 16.0 - t * 1.5))
        b_wave = np.sin((x**2 + y**2) * b_freq - t * 3.0)

        # แปลงคลื่นให้อยู่ในย่าน [0, 255]
        r_synth = (r_wave + 1.0) * 127.5
        g_synth = (g_wave + 1.0) * 127.5
        b_synth = (b_wave + 1.0) * 127.5
        
        synth_frame = np.stack([r_synth, g_synth, b_synth], axis=-1)

        # หากมีรูปภาพจากผู้ใช้ ให้นำภาพมาผสม (Blend) กับ Farland Art
        if input_image is not None:
            blend_factor = 0.5 + 0.3 * np.sin(t * 2.0) # ทำให้ภาพกระพริบวูบวาบตามจังหวะ
            final_frame = input_image * blend_factor + synth_frame * (1.0 - blend_factor)
        else:
            final_frame = synth_frame

        # เติมนอยส์ Nanopixel
        noise = np.random.normal(0, 8.0, (height, width, 3))
        final_frame = np.clip(final_frame + noise, 0, 255).astype(np.uint8)

        frames.append(final_frame)

        if (frame_idx + 1) % 60 == 0:
            print(f"  └─ Frame {frame_idx + 1}/{total_frames} done.")

    return frames

def main():
    duration = 10.0
    output_path = "eurveu_farland_output.mp4"

    # 1. ให้สมอง EurVeu สแกนอ่านไฟล์ใน inputs/
    brain = EurVeuBrain(memory_file="eurveu_memory.json", inputs_dir="inputs")
    brain.scan_and_understand_inputs()
    brain_vectors = brain.memory.get("learned_vectors", {})

    # 2. อ่านไฟล์ภาพอัปโหลดล่าสุด
    input_image = load_latest_input_image(inputs_dir="inputs", target_size=(720, 1280))

    # 3. สร้างเสียง
    audio_data, sr = generate_audio(brain_vectors, duration=duration)
    sf.write("temp_audio.wav", audio_data, sr)

    # 4. สร้างวิดีโอ
    frames = generate_frames(brain_vectors, input_image=input_image, duration=duration)

    # 5. ประกอบไฟล์เป็นวิดีโอ MP4
    print("[EurVeu Export] Encoding Video with FFmpeg...")
    imageio.mimwrite("temp_video.mp4", frames, fps=30, codec='libx264')

    os.system(f"ffmpeg -y -i temp_video.mp4 -i temp_audio.wav -c:v copy -c:a aac -b:a 320k {output_path} -loglevel quiet")

    # เคลียร์ไฟล์ชั่วคราว
    if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")
    if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")

    print("[SUCCESS] EurVeu Farland Render Complete!")

if __name__ == "__main__":
    main()
