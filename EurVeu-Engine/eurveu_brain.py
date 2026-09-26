import numpy as np
import json
import os
import glob
from PIL import Image

class EurVeuBrain:
    def __init__(self, memory_file="eurveu_memory.json", inputs_dir="inputs"):
        self.memory_file = memory_file
        self.inputs_dir = inputs_dir
        os.makedirs(self.inputs_dir, exist_ok=True)
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "generation": 0,
            "curiosity_index": 1.0,
            "processed_files_hash": [],
            "visual_entropy_history": [],
            "learned_vectors": {
                "r_freq": 12.0,
                "g_freq": 14.0,
                "b_freq": 25.0,
                "audio_density": 20000
            }
        }

    def compute_image_hash(self, img_path):
        """ คำนวณลายนิ้วมือไฟล์ พร้อมย่อขนาดเพื่อป้องกัน RAM เต็ม """
        try:
            # ใช้ Try-Except เพื่อให้ไฟล์ที่เสียไม่ทำให้โปรแกรมค้าง
            with Image.open(img_path) as img:
                img_small = img.resize((16, 16)).convert('L')
                pixels = np.array(img_small)
                avg = pixels.mean()
                diff = pixels > avg
                return hash(tuple(diff.flatten()))
        except Exception as e:
            print(f"  ⚠️ Warning: Cannot process {os.path.basename(img_path)} - {e}")
            return None

    def scan_and_understand_inputs(self):
        print(f"[EurVeu Brain] Fast Scanning '{self.inputs_dir}'...")
        
        # ค้นหาไฟล์ภาพและวิดีโอ
        all_files = glob.glob(f"{self.inputs_dir}/*")
        valid_sources = []

        for file_path in all_files:
            if file_path.endswith('.gitkeep'):
                continue
                
            file_name = os.path.basename(file_path)
            
            # ตรวจสอบภาพ/วิดีโอ ป้องกันโปรแกรมล้ม
            try:
                file_hash = self.compute_image_hash(file_path)
                
                # หากสแกนไฟล์ภาพไม่ได้ ให้สร้าง Hash จากชื่อไฟล์และขนาดไฟล์แทน (สำหรับวิดีโอ)
                if file_hash is None:
                    file_size = os.path.getsize(file_path)
                    file_hash = hash((file_name, file_size))

                if file_hash in self.memory["processed_files_hash"]:
                    print(f"  ├─ Already in memory: {file_name}")
                    continue

                print(f"  ├─ Successfully Perceived: {file_name}")
                self.memory["processed_files_hash"].append(file_hash)
                valid_sources.append(file_path)

            except Exception as err:
                print(f"  ❌ Skip corrupted file: {file_name} ({err})")
                continue

        if valid_sources:
            print(f"[EurVeu Brain] Evolution updated with {len(valid_sources)} new sources!")
            self.memory["generation"] += 1
            vectors = self.memory["learned_vectors"]
            vectors["r_freq"] += len(valid_sources) * 0.5
            vectors["g_freq"] += len(valid_sources) * 0.8
            vectors["b_freq"] += len(valid_sources) * 1.0

        self.save_memory()
        return valid_sources

    def save_memory(self):
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=4)
