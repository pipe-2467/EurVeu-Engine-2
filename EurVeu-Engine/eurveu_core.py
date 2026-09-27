"""
================================)
                    EurVeu Custom Deep Neural Core Engine v4.0
================================================================================
Architecture: Multi-Stage Transposed Conv Generative Decoder + Self-Attention + 
              Spherical Latent Traversal (SLERP) + Spatial Harmonic Audio Synthesizer
File: EurVeu-Engine/eurveu_core.py
Description: Full-scale video frame generation, spatial audio synthesis, 
              and high-bitrate FFmpeg video compilation pipeline.
================================================================================
"""

import os
import sys
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import imageio
from PIL import Image
from eurveu_brain import EurVeuBrain

# ------------------------------------------------------------------------------
# 0. Global Logging Setup & Hardware Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [EurVeu-Core] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EurVeuCore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# 1. Spatial Self-Attention for Neural Decoder
# ------------------------------------------------------------------------------
class DecoderSpatialAttention(nn.Module):
    """
    Spatial Attention Layer for Generative Neural Decoder to maintain structural
    coherence across feature dimensions during image synthesis.
    """
    def __init__(self, in_channels):
        super(DecoderSpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        q = self.query(x).view(batch_size, -1, N).permute(0, 2, 1) # [B, N, C']
        k = self.key(x).view(batch_size, -1, N)                    # [B, C', N]
        energy = torch.bmm(q, k)                                   # [B, N, N]
        attention = F.softmax(energy, dim=-1)

        v = self.value(x).view(batch_size, -1, N)                  # [B, C, N]
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return self.gamma * out + x


# ------------------------------------------------------------------------------
# 2. Residual Bottleneck Block for Generative Decoder
# ------------------------------------------------------------------------------
class GenerativeResidualBlock(nn.Module):
    """
    Residual Bottleneck Block with Group Normalization and GELU activation
    for deep spatial feature generation.
    """
    def __init__(self, channels):
        super(GenerativeResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.act(out)


# ------------------------------------------------------------------------------
# 3. Custom Multi-Stage Deep Generative Neural Decoder Network
# ------------------------------------------------------------------------------
class DeepGenerativeNeuralDecoder(nn.Module):
    """
    Deep Transposed Convolutional Neural Network with Residual Blocks and 
    Self-Attention for decoding 128D Latent Coordinates into high-resolution frames.
    """
    def __init__(self, latent_dim=128):
        super(DeepGenerativeNeuralDecoder, self).__init__()

        self.latent_expansion = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256 * 18 * 32),
            nn.GELU()
        )

        self.upsample_stage1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), # -> 36x64
            nn.GroupNorm(8, 128),
            nn.GELU(),
            GenerativeResidualBlock(128)
        )
        self.attn1 = DecoderSpatialAttention(128)

        self.upsample_stage2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # -> 72x128
            nn.GroupNorm(8, 64),
            nn.GELU(),
            GenerativeResidualBlock(64)
        )
        self.attn2 = DecoderSpatialAttention(64)

        self.upsample_stage3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),   # -> 144x256
            nn.GroupNorm(8, 32),
            nn.GELU()
        )

        self.upsample_stage4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),   # -> 288x512
            nn.GroupNorm(8, 16),
            nn.GELU()
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid() # Normalize pixel tensor values to range [0.0, 1.0]
        )

    def forward(self, z):
        x = self.latent_expansion(z)
        x = x.view(-1, 256, 18, 32)

        x = self.upsample_stage1(x)
        x = self.attn1(x)

        x = self.upsample_stage2(x)
        x = self.attn2(x)

        x = self.upsample_stage3(x)
        x = self.upsample_stage4(x)

        generated_frame = self.output_layer(x)
        return generated_frame


# ------------------------------------------------------------------------------
# 4. Spherical Latent Traversal Engine (SLERP & Harmonic Orbital Motion)
# ------------------------------------------------------------------------------
class SphericalLatentTraversalEngine:
    """
    Computes smooth trajectories through 128D Latent Vector Space using Spherical
    Linear Interpolation (SLERP), orbital wave motion, and sensory metrics.
    """
    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim

    def slerp(self, val, low, high):
        """Spherical Linear Interpolation between two N-dimensional vectors."""
        low_norm = low / torch.norm(low, dim=-1, keepdim=True)
        high_norm = high / torch.norm(high, dim=-1, keepdim=True)

        dot = torch.clamp(torch.sum(low_norm * high_norm, dim=-1, keepdim=True), -0.9995, 0.9995)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        if so.item() == 0:
            return (1.0 - val) * low + val * high

        scale0 = torch.sin((1.0 - val) * omega) / so
        scale1 = torch.sin(val * omega) / so

        return scale0 * low + scale1 * high

    def compute_trajectory_step(self, start_vector, frame_idx, total_frames, metrics):
        t = (frame_idx / total_frames) * 2.0 * math.pi
        warmth = metrics.get("warmth", 0.0)
        entropy = metrics.get("entropy", 3.0)
        contrast = metrics.get("contrast", 0.1)

        # Generate target secondary destination vector
        torch.manual_seed(frame_idx + 2000)
        target_destination_vector = torch.randn_like(start_vector).to(DEVICE)

        # Compute SLERP interpolation base
        slerp_progress = (math.sin(t * 0.5 - math.pi / 2.0) + 1.0) / 2.0
        interpolated_base = self.slerp(slerp_progress, start_vector, target_destination_vector)

        # Harmonic orbital wave offsets
        radial_orbital_wave = math.sin(t * 2.5) * (0.15 + contrast * 0.1)
        angular_orbital_wave = math.cos(t * 1.8) * (0.12 + warmth * 0.2)

        # Stochastic noise layer scaled by entropy
        stochastic_noise = torch.randn_like(start_vector).to(DEVICE) * (0.01 + entropy * 0.002)

        final_step_coordinate = interpolated_base + radial_orbital_wave + angular_orbital_wave + stochastic_noise
        return final_step_coordinate


# ------------------------------------------------------------------------------
# 5. Spatial Harmonic Audio Synthesizer
# ------------------------------------------------------------------------------
class SpatialHarmonicAudioSynthesizer:
    """
    Synthesizes multi-layered ambient spatial audio tracks based on sensory metrics.
    Integrates carrier harmonics, binaural beats, and sub-bass resonance.
    """
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    def synthesize_track(self, metrics, duration=10.0):
        logger.info("Synthesizing Multi-Layered Spatial Harmonic Audio Track...")
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)

        warmth = metrics.get("warmth", 0.0)
        contrast = metrics.get("contrast", 0.1)
        entropy = metrics.get("entropy", 3.0)

        # Fundamental resonant carrier frequency (OM / Cosmic Resonance base)
        base_frequency = 136.10 + warmth * 60.0

        carrier_wave = np.sin(2 * np.pi * base_frequency * t) * 0.35
        harmonic_layer1 = np.sin(2 * np.pi * (base_frequency * 1.5) * t) * 0.18
        harmonic_layer2 = np.sin(2 * np.pi * (base_frequency * 2.0) * t) * 0.08
        sub_bass_layer = np.sin(2 * np.pi * (base_frequency * 0.5) * t) * 0.40

        # Binaural Theta Wave (7.83 Hz Schumann Resonance offset)
        binaural_theta = np.sin(2 * np.pi * (base_frequency + 7.83) * t) * 0.15

        # Ambient stochastic noise floor
        noise_amplitude = 0.003 + contrast * 0.01 + entropy * 0.001
        ambient_noise = np.random.normal(0, noise_amplitude, len(t))

        # Composite signal
        audio_composite = carrier_wave + harmonic_layer1 + harmonic_layer2 + sub_bass_layer + binaural_theta + ambient_noise

        # Amplitude Normalization & Smooth Fade In/Out
        max_amp = np.max(np.abs(audio_composite))
        if max_amp > 0:
            audio_composite = audio_composite / max_amp

        fade_samples = int(self.sample_rate * 0.5) # 0.5s fade
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)

        audio_composite[:fade_samples] *= fade_in
        audio_composite[-fade_samples:] *= fade_out

        return audio_composite, self.sample_rate


# ------------------------------------------------------------------------------
# 6. Main EurVeu Core Execution Pipeline
# ------------------------------------------------------------------------------
def execute_eurveu_core_pipeline():
    duration = 10.0
    fps = 30
    total_frames = int(fps * duration)
    output_filename = "eurveu_farland_output.mp4"
    latent_dim = 128

    logger.info(f"Initializing EurVeu Neural Core Render Subsystem on Device: {DEVICE}")

    # Step 1: Perceive sensory input via Neural Brain Engine
    brain = EurVeuBrain(inputs_dir="inputs", memory_filepath="eurveu_memory.json", latent_dim=latent_dim)
    start_spatial_coordinate, sensory_metrics = brain.perceive_and_evolve()

    # Step 2: Initialize Generative Neural Decoder & Traversal Engine
    decoder = DeepGenerativeNeuralDecoder(latent_dim=latent_dim).to(DEVICE)
    decoder.eval()

    traversal_engine = SphericalLatentTraversalEngine(latent_dim=latent_dim)

    # Step 3: Render Video Frames via Deep Neural Forward Pass
    rendered_frames = []
    logger.info(f"Rendering {total_frames} Spatial Frames via Neural Forward Pass...")

    render_start_time = time.time()
    with torch.no_grad():
        for frame_idx in range(total_frames):
            current_step_coord = traversal_engine.compute_trajectory_step(
                start_spatial_coordinate, frame_idx, total_frames, sensory_metrics
            )

            # Forward Pass through Deep Decoder
            generated_tensor = decoder(current_step_coord) # [1, 3, 288, 512]

            frame_np = generated_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_uint8 = (frame_np * 255.0).clip(0, 255).astype(np.uint8)

            # Resize frame to target vertical video resolution (720x1280)
            pil_frame = Image.fromarray(frame_uint8).resize((720, 1280), Image.BILINEAR)
            rendered_frames.append(np.array(pil_frame))

            if (frame_idx + 1) % 60 == 0 or (frame_idx + 1) == total_frames:
                logger.info(f"  └─ Generated Neural Frame {frame_idx + 1}/{total_frames}")

    elapsed_render_time = time.time() - render_start_time
    logger.info(f"Visual Frame Rendering Complete in {elapsed_render_time:.2f} seconds.")

    # Step 4: Synthesize Audio Track
    audio_synthesizer = SpatialHarmonicAudioSynthesizer(sample_rate=48000)
    audio_data, sr = audio_synthesizer.synthesize_track(sensory_metrics, duration=duration)

    temp_audio_filepath = "temp_audio.wav"
    temp_video_filepath = "temp_video.mp4"

    sf.write(temp_audio_filepath, audio_data, sr)

    # Step 5: Encode Final MP4 Video via High-Bitrate FFmpeg Pipeline
    logger.info("Encoding High-Bitrate Final MP4 Video Output via FFmpeg...")
    imageio.mimwrite(temp_video_filepath, rendered_frames, fps=fps, codec='libx264')

    ffmpeg_command = (
        f"ffmpeg -y -i {temp_video_filepath} -i {temp_audio_filepath} "
        f"-c:v copy -c:a aac -b:a 320k {output_filename} -loglevel quiet"
    )
    os.system(ffmpeg_command)

    # Clean temporary rendering artifacts
    if os.path.exists(temp_audio_filepath):
        os.remove(temp_audio_filepath)
    if os.path.exists(temp_video_filepath):
        os.remove(temp_video_filepath)

    logger.info(f"[SUCCESS] EurVeu Deep Learning Synthesis Complete! Output File: {output_filename}")


if __name__ == "__main__":
    execute_eurveu_core_pipeline()
