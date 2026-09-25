import numpy as np
import soundfile as sf
import imageio
import os

print("==================================================")
print("     EurVeu Autonomous Cloud Render Engine        ")
print("==================================================")

def generate_audio(duration=10.0, sample_rate=48000, num_layers=20000):
    print(f"[EurVeu Audio] Synthesizing {num_layers:,} frequency waves...")
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)

    # สุ่มย่านความถี่อย่างอิสระไร้กฎเกณฑ์ (Sub-bass ถึง Ultra-high frequency)
    freqs = np.random.uniform(20.0, 22000.0, num_layers)
    phases = np.random.uniform(0, 2 * np.pi, num_layers)
    amps = np.random.exponential(0.05, num_layers)

    for i in range(num_layers):
        audio += amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])

    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp
    return audio, sample_rate

def generate_frames(width=720, height=1280, fps=30, duration=10.0):
    total_frames = int(fps * duration)
    print(f"[EurVeu Visual] Rendering {total_frames} Nanopixel frames ({width}x{height})...")
    
    y, x = np.ogrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    frames = []

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        
        # สมการสังเคราะห์ Farland Noise & Structural Distortion
        r = np.sin(x * 12.0 + np.cos(y * 8.0 + t) + t * 2.0)
        g = np.cos(y * 14.0 + np.sin(x * 16.0 - t * 1.5))
        b = np.sin((x**2 + y**2) * 25.0 - t * 3.0)

        # เพิ่มพิกเซลนอยส์ความถี่สูงเพื่อความสมจริงแบบ Nanopixel
        noise = np.random.normal(0, 0.12, (height, width))
        r = np.clip((r + noise + 1.0) * 127.5, 0, 255).astype(np.uint8)
        g = np.clip((g + noise + 1.0) * 127.5, 0, 255).astype(np.uint8)
        b = np.clip((b + noise + 1.0) * 127.5, 0, 255).astype(np.uint8)

        frame = np.stack([r, g, b], axis=-1)
        frames.append(frame)

        if (frame_idx + 1) % 60 == 0:
            print(f"  └─ Frame {frame_idx + 1}/{total_frames} done.")

    return frames

def main():
    duration = 10.0  # กำหนดความยาวของวิดีโอ (วินาที)
    output_path = "eurveu_farland_output.mp4"

    # 1. สร้างไฟล์เสียง
    audio_data, sr = generate_audio(duration=duration)
    sf.write("temp_audio.wav", audio_data, sr)

    # 2. สร้างภาพแต่ละเฟรม
    frames = generate_frames(duration=duration)

    # 3. ประกอบรวมวิดีโอและเสียงด้วย FFmpeg
    print("[EurVeu Export] Encoding Video with FFmpeg...")
    imageio.mimwrite("temp_video.mp4", frames, fps=30, codec='libx264')

    os.system(f"ffmpeg -y -i temp_video.mp4 -i temp_audio.wav -c:v copy -c:a aac -b:a 320k {output_path} -loglevel quiet")

    # ลบไฟล์ชั่วคราวหลังประมวลผลเสร็จ
    if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")
    if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")

    print("[SUCCESS] EurVeu Farland Render Complete!")

if __name__ == "__main__":
    main()
