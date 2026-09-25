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
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
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
        """ คำนวณลายนิ้วมือดิจิทัลของภาพเพื่อคัดกรองภาพซ้ำ """
        try:
            img = Image.open(img_path).resize((16, 16)).convert('L')
            pixels = np.array(img)
            avg = pixels.mean()
            diff = pixels > avg
            return hash(tuple(diff.flatten()))
        except Exception:
            return None

    def scan_and_understand_inputs(self):
        """ สแกนดูภาพ/วิดีโอ พิจารณาภาพซ้ำ และประติดประต่อสร้างชุดข้อมูลใหม่ """
        print(f"[EurVeu Brain] Scanning '{self.inputs_dir}' for new perceptions...")
        
        image_files = (
            glob.glob(f"{self.inputs_dir}/*.[jJ][pP][gG]") +
            glob.glob(f"{self.inputs_dir}/*.[pP][nN][gG]") +
            glob.glob(f"{self.inputs_dir}/*.[jJ][pP][eE][gG]")
        )
        
        valid_sources = []

        for file_path in image_files:
            file_hash = self.compute_image_hash(file_path)
            if file_hash is None:
                continue

            # พิจารณาภาพซ้ำ
            if file_hash in self.memory["processed_files_hash"]:
                print(f"  ├─ Skip duplicate media: {os.path.basename(file_path)}")
                continue

            print(f"  ├─ Perceived NEW media: {os.path.basename(file_path)}")
            self.memory["processed_files_hash"].append(file_hash)
            valid_sources.append(file_path)

        # การประติดประต่อเวกเตอร์การเรียนรู้ (Self-Evolution Vectoring)
        if valid_sources:
            print(f"[EurVeu Brain] Evolving memory vectors based on {len(valid_sources)} new sources...")
            self.memory["generation"] += 1
            vectors = self.memory["learned_vectors"]
            vectors["r_freq"] += len(valid_sources) * 0.8
            vectors["g_freq"] += len(valid_sources) * 1.2
            vectors["b_freq"] += len(valid_sources) * 1.5

        self.save_memory()
        return valid_sources

    def save_memory(self):
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=4)
