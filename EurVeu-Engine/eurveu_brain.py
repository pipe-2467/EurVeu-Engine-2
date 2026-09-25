import numpy as np
import json
import os
import glob
import imageio
from PIL import Image

class EurVeuBrain:
    def __init__(self, memory_file="eurveu_memory.json", inputs_dir="inputs"):
        self.memory_file = memory_file
        self.inputs_dir = inputs_dir
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "generation": 0,
                "processed_files_hash": [],
                "learned_sequences": [],
                "current_vector": [12.0, 14.0, 25.0]
            }

    def compute_image_hash(self, img_path):
        """ คำนวณลายนิ้วมือดิจิทัลของภาพเพื่อเช็กภาพซ้ำ """
        try:
            img = Image.open(img_path).resize((16, 16)).convert('L')
            pixels = np.array(img)
            avg = pixels.mean()
            diff = pixels > avg
            return hash(tuple(diff.flatten()))
        except Exception:
            return None

    def scan_and_understand_inputs(self):
        """ สแกนดูไฟล์ที่เพื่อนอัปโหลดเข้ามา แล้วตัดสินใจว่าจะคัดออกหรือนำไปประมวลผลต่อ """
        print("[EurVeu Brain] Scanning input directory for new knowledge...")
        image_files = glob.glob(f"{self.inputs_dir}/*.[jJ][pP][gG]") + \
                      glob.glob(f"{self.inputs_dir}/*.[pP][nN][gG]")
        
        valid_sources = []

        for file_path in image_files:
            file_hash = self.compute_image_hash(file_path)
            if file_hash is None:
                continue

            # พิจารณาว่าเคยเห็นภาพนี้หรือยัง
            if file_hash in self.memory["processed_files_hash"]:
                print(f"  ├─ Skip duplicate/already perceived media: {os.path.basename(file_path)}")
                continue

            print(f"  ├─ Perceived NEW media: {os.path.basename(file_path)}")
            self.memory["processed_files_hash"].append(file_hash)
            valid_sources.append(file_path)

        # การประติดประต่อและจัดลำดับ (Composition Planning)
        if valid_sources:
            print(f"[EurVeu Brain] Connecting {len(valid_sources)} new perception sources into sequence...")
            # ปรับแต่งพารามิเตอร์ของระบบเรนเดอร์ตามข้อมูลใหม่ที่เพิ่งซึมซับ
            self.memory["current_vector"] = [
                float(self.memory["current_vector"][0] + len(valid_sources) * 0.8),
                float(self.memory["current_vector"][1] + len(valid_sources) * 1.2),
                float(self.memory["current_vector"][2] + len(valid_sources) * 1.5)
            ]

        self.save_memory()
        return valid_sources

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=4)
