"""
================================================================================
                    EurVeu Deep Neural Brain Engine v4.0
================================================================================
Architecture: Deep Multi-Scale Vision Encoder + Cross-Attention + Dynamic Latent 
              Space Quantization & Persistent Experiential Memory Bank
File: EurVeu-Engine/eurveu_brain.py
Description: Full-scale sensory ingestion, spatial coordinate mapping, multi-head
              attention feature extraction, and persistent long-term memory evolution.
================================================================================
"""

import os
import sys
import json
import time
import math
import logging
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageStat, ImageFilter

# ------------------------------------------------------------------------------
# 0. Global Logging Setup & Hardware Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [EurVeu-Brain] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EurVeuBrain")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# 1. Advanced Multi-Head Self-Attention Layer
# ------------------------------------------------------------------------------
class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-Head Spatial Self-Attention for learning global spatial context
    and structural feature dependencies across spatial feature maps.
    """
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadSpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj   = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        # Projections
        q = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2) # [B, heads, N, head_dim]
        k = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, N)                       # [B, heads, head_dim, N]
        v = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2) # [B, heads, N, head_dim]

        # Scaled Dot-Product Attention
        energy = torch.matmul(q, k) / math.sqrt(self.head_dim) # [B, heads, N, N]
        attention = F.softmax(energy, dim=-1)

        out = torch.matmul(attention, v) # [B, heads, N, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, C, H, W)
        out = self.out_proj(out)

        return self.gamma * out + x


# ------------------------------------------------------------------------------
# 2. Residual Neural Backbone Block
# ------------------------------------------------------------------------------
class ResidualEncoderBlock(nn.Module):
    """
    Deep Residual Conv Block with Group Normalization and GELU activation.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2  = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += res
        return self.act2(out)


# ------------------------------------------------------------------------------
# 3. Custom Multi-Scale Deep Vision Encoder Architecture
# ------------------------------------------------------------------------------
class DeepPerceptionEncoder(nn.Module):
    """
    Multi-Scale Deep Convolutional Vision Encoder for extracting latent representations
    from visual sensory inputs.
    """
    def __init__(self, feature_dim=512):
        super(DeepPerceptionEncoder, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # 128x128 -> 64x64
            nn.GroupNorm(8, 64),
            nn.GELU()
        )

        self.stage1 = ResidualEncoderBlock(64, 128, stride=2)   # -> 32x32
        self.attn1  = MultiHeadSpatialAttention(128, num_heads=4)

        self.stage2 = ResidualEncoderBlock(128, 256, stride=2)  # -> 16x16
        self.attn2  = MultiHeadSpatialAttention(256, num_heads=8)

        self.stage3 = ResidualEncoderBlock(256, 512, stride=2)  # -> 8x8
        self.attn3  = MultiHeadSpatialAttention(512, num_heads=8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection_head = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.attn1(x)
        x = self.stage2(x)
        x = self.attn2(x)
        x = self.stage3(x)
        x = self.attn3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        out_features = self.projection_head(x)
        return out_features


# ------------------------------------------------------------------------------
# 4. Latent Space Mapping Engine & Feeling Metrics Extractor
# ------------------------------------------------------------------------------
class LatentSpaceCoordinateMapper:
    """
    Translates physical image parameters and deep feature vectors into 
    precise coordinates within the 128D Latent Vector Space.
    """
    def __init__(self, feature_dim=512, latent_dim=128):
        self.latent_dim = latent_dim
        self.linear_projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        ).to(DEVICE)

    def extract_sensory_metrics(self, pil_image, image_tensor):
        """Extracts statistical and aesthetic metrics from sensory inputs."""
        stat = ImageStat.Stat(pil_image)
        mean_color = stat.mean
        std_color = stat.stddev

        brightness = float(np.mean(mean_color) / 255.0)
        
        # Color warmth calculation (Red vs Blue ratio)
        r_val = mean_color[0] if len(mean_color) > 0 else 128
        b_val = mean_color[2] if len(mean_color) > 2 else 128
        warmth = float((r_val - b_val) / 255.0)

        contrast = float(np.mean(std_color) / 128.0)

        # Image Entropy calculation
        histogram = pil_image.histogram()
        total_pixels = sum(histogram)
        entropy = 0.0
        for count in histogram:
            if count > 0:
                p = count / total_pixels
                entropy -= p * math.log2(p)

        # Sharpness assessment using edge detection
        edges = pil_image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        sharpness = float(np.mean(edge_stat.mean) / 255.0)

        return {
            "brightness": float(np.round(brightness, 6)),
            "warmth": float(np.round(warmth, 6)),
            "contrast": float(np.round(contrast, 6)),
            "entropy": float(np.round(entropy, 6)),
            "sharpness": float(np.round(sharpness, 6))
        }

    def compute_spatial_coordinate(self, pil_image, image_tensor, feature_vector):
        """Calculates exact 128-dimensional spatial vector in Latent Space."""
        metrics = self.extract_sensory_metrics(pil_image, image_tensor)

        # Generate deterministic seed based on sensory metrics and SHA256 hash
        img_bytes = pil_image.tobytes()
        hash_digest = hashlib.sha256(img_bytes).hexdigest()
        seed_int = int(hash_digest[:8], 16) % (2**31)

        torch.manual_seed(seed_int)
        stochastic_spatial_base = torch.randn(1, self.latent_dim).to(DEVICE)

        with torch.no_grad():
            feature_spatial_projection = self.linear_projector(feature_vector)

        # Blend stochastic coordinate base with deep neural feature projection
        combined_coordinate = 0.4 * stochastic_spatial_base + 0.6 * feature_spatial_projection

        # Apply warmth and entropy modulation vectors
        modulation = torch.zeros(1, self.latent_dim).to(DEVICE)
        modulation[0, 0::2] = metrics["warmth"] * 0.2
        modulation[0, 1::2] = metrics["entropy"] * 0.15

        final_spatial_coordinate = combined_coordinate + modulation

        return final_spatial_coordinate, metrics, hash_digest


# ------------------------------------------------------------------------------
# 5. Long-Term Experiential Neural Memory Bank
# ------------------------------------------------------------------------------
class ExperientialMemoryBank:
    """
    Persistent JSON-backed long-term memory system storing evolution cycles,
    learned sensory coordinates, and environmental experience logs.
    """
    def __init__(self, memory_filepath="eurveu_memory.json"):
        self.memory_filepath = memory_filepath
        self.memory_state = self._load_or_initialize()

    def _load_or_initialize(self):
        if os.path.exists(self.memory_filepath):
            try:
                with open(self.memory_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded existing Neural Memory Bank (Evolution Cycle: {data.get('evolution_cycle', 0)})")
                    return data
            except Exception as e:
                logger.warning(f"Failed to read memory file ({e}). Initializing new Neural Memory Bank.")

        return {
            "engine_name": "EurVeu Neural Engine",
            "version": "4.0.0-Ultimate",
            "evolution_cycle": 0,
            "total_perceptions": 0,
            "memory_log": [],
            "spatial_trajectories": [],
            "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": None
        }

    def persist(self):
        self.memory_state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.memory_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.memory_state, f, indent=4, ensure_ascii=False)
            logger.info("Experiential Memory Bank saved to disk successfully.")
        except Exception as e:
            logger.error(f"Error persisting memory bank: {e}")

    def record_experience(self, filename, metrics, spatial_vector, hash_digest):
        self.memory_state["evolution_cycle"] += 1
        self.memory_state["total_perceptions"] += 1

        vector_preview = spatial_vector.squeeze().detach().cpu().numpy()[:12].tolist()
        vector_preview = [float(np.round(x, 6)) for x in vector_preview]

        log_entry = {
            "cycle": self.memory_state["evolution_cycle"],
            "input_file": filename,
            "sha256": hash_digest,
            "sensory_metrics": metrics,
            "latent_coordinate_sample": vector_preview,
            "timestamp": time.time()
        }

        self.memory_state["memory_log"].append(log_entry)
        if len(self.memory_state["memory_log"]) > 150:
            self.memory_state["memory_log"].pop(0)

        self.persist()


# ------------------------------------------------------------------------------
# 6. Main EurVeu Brain Subsystem Controller
# ------------------------------------------------------------------------------
class EurVeuBrain:
    """
    Central Executive Manager for EurVeu Brain. Coordinates perception pipeline,
    neural feature extraction, spatial coordinate calculation, and memory persistence.
    """
    def __init__(self, inputs_dir="inputs", memory_filepath="eurveu_memory.json", latent_dim=128):
        self.inputs_dir = inputs_dir
        self.latent_dim = latent_dim
        
        logger.info(f"Initializing EurVeu Brain Subsystems on Device: {DEVICE}")

        self.encoder = DeepPerceptionEncoder(feature_dim=512).to(DEVICE)
        self.encoder.eval()

        self.mapper = LatentSpaceCoordinateMapper(feature_dim=512, latent_dim=latent_dim)
        self.memory = ExperientialMemoryBank(memory_filepath=memory_filepath)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def perceive_and_evolve(self):
        """
        Executes sensory perception loop over available image inputs, computes spatial
        coordinates in 128D Latent Space, and records experience into memory.
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        matched_files = []

        if os.path.exists(self.inputs_dir):
            for file_entry in os.listdir(self.inputs_dir):
                if file_entry.lower().endswith(valid_exts) and file_entry != '.gitkeep':
                    matched_files.append(os.path.join(self.inputs_dir, file_entry))

        if not matched_files:
            logger.info("No visual sensory input found. Spawning origin coordinate vector.")
            default_coordinate = torch.randn(1, self.latent_dim).to(DEVICE)
            default_metrics = {
                "brightness": 0.5,
                "warmth": 0.0,
                "contrast": 0.1,
                "entropy": 3.0,
                "sharpness": 0.1
            }
            return default_coordinate, default_metrics

        # Select latest input file
        latest_file = max(matched_files, key=os.path.getmtime)
        filename = os.path.basename(latest_file)
        logger.info(f"Processing sensory visual input: {filename}")

        try:
            pil_image = Image.open(latest_file).convert("RGB")
            tensor_image = self.transform(pil_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feature_vector = self.encoder(tensor_image)
                spatial_coordinate, metrics, hash_digest = self.mapper.compute_spatial_coordinate(
                    pil_image, tensor_image, feature_vector
                )

            # Record experience into neural memory bank
            self.memory.record_experience(filename, metrics, spatial_coordinate, hash_digest)

            logger.info(f"Perception Cycle {self.memory.memory_state['evolution_cycle']} Complete.")
            logger.info(f"Metrics -> Warmth: {metrics['warmth']} | Contrast: {metrics['contrast']} | Entropy: {metrics['entropy']}")

            return spatial_coordinate, metrics

        except Exception as e:
            logger.error(f"Error encountered during perception pass: {e}")
            fallback_coord = torch.randn(1, self.latent_dim).to(DEVICE)
            fallback_metrics = {
                "brightness": 0.5,
                "warmth": 0.0,
                "contrast": 0.1,
                "entropy": 3.0,
                "sharpness": 0.1
            }
            return fallback_coord, fallback_metrics


if __name__ == "__main__":
    logger.info("Executing Standalone EurVeu Brain Pipeline Test...")
    brain_instance = EurVeuBrain()
    coord, metrics = brain_instance.perceive_and_evolve()
    print("EurVeu Brain Test Complete. Output Coordinate Shape:", coord.shape)
