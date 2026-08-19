# Smart Glasses Integrating Emotion Recognition for Autistic Individuals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg )](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue )](https://www.python.org/ )
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white )](https://pytorch.org/ )
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange )](https://huggingface.co/ )
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi )](https://fastapi.tiangolo.com/ )

An end-to-end multimodal AI assistive platform designed for smart glasses (Vuzix Blade 2) to empower individuals with Autism Spectrum Disorder (ASD) by delivering real-time, non-intrusive emotional intelligence cues during social interactions.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture & Modalities](#️-system-architecture--modalities)
  - [1. Vision Modality (Facial Emotion Recognition)](#1-vision-modality-facial-emotion-recognition)
  - [2. Audio Modality (Speech Emotion Recognition)](#2-audio-modality-speech-emotion-recognition)
  - [3. Text Modality (Context Emotion Recognition)](#3-text-modality-context-emotion-recognition)
  - [4. Multimodal Fusion Module](#4-multimodal-fusion-module)
- [Model Performance Benchmarks](#-model-performance-benchmarks)
- [Hardware & Mobile Integration](#-hardware--mobile-integration)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [How to Run](#-how-to-run)
  - [1. Evaluation & Benchmarking](#1-evaluation--benchmarking)
  - [2. Model Training](#2-model-training)
  - [3. Backend API Deployment](#3-backend-api-deployment)
  - [4. Interactive Demo (Google Colab)](#4-interactive-demo-google-colab)
- [Challenges & Current Limitations](#️-challenges--current-limitations)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## Project Overview

Individuals with Autism Spectrum Disorder (ASD) frequently encounter challenges interpreting subtle emotional cues—such as changing facial expressions, voice inflection, and verbal context—during live conversations. These difficulties can lead to heightened social anxiety, miscommunication, and fatigue.

This platform bridges that gap by transforming smart wearable technology into an assistive real-time social feedback loop:

1. **Sensory Capture:** Smart glasses continuously record video frames and audio streams.
2. **Cloud AI Processing:** Inputs are streamed to a high-throughput FastAPI inference server deploying fine-tuned Vision, Speech, and Natural Language AI models.
3. **Multimodal Fusion:** Predictions across all three channels are weighted and combined into a single, high-confidence output.
4. **HUD Visual Feedback:** Real-time emotion notifications are transmitted back and projected subtly onto the smart glasses HUD (Heads-Up Display).

---

## Key Features

- **Multi-Sensor Data Fusion:** Combines visual, acoustic, and linguistic signals for significantly higher prediction robustness than single-modality systems.
- **Ultra-Low Latency Pipeline:** Uses lightweight, fine-tuned transformer architectures served via FastAPI and ngrok tunnels for real-time responsiveness.
- **Wearable HUD Interface:** Native Android application designed for Vuzix Blade 2 to display clean, non-distracting visual indicators.
- **Robustness Against Ambient Noise:** Audio processing includes noise injection training, FFmpeg scrubbing, and Librosa gain normalization for real-world environmental clarity.

---

## System Architecture & Modalities

The system classifies emotional cues across human interactions into four core target categories: **Neutral**, **Happy**, **Angry**, and **Sad**.

```

┌─────────────────┐        HTTPS / Audio & Video Stream       ┌────────────────────────┐
│  Vuzix Glasses  │ ────────────────────────────────────────> │ FastAPI Cloud Server   │
│ (Camera & Mic)  │ <──────────────────────────────────────── │  [ViT + Wav2Vec + BERT]│
└─────────────────┘              JSON Emotion Payload         └────────────────────────┘

```

### 1. Vision Modality (Facial Emotion Recognition)

- **Architecture:** Fine-tuned **Vision Transformer (ViT)** (`dima806/facial_emotions_image_detection`).
- **Dataset:** Facial Emotion Recognition Dataset (~12,000 cropped RGB expressions).
- **Optimization:** Extensive data augmentations (random rotations, color jitter, horizontal flipping) coupled with sequence-level fine-tuning of the classification head.

### 2. Audio Modality (Speech Emotion Recognition)

- **Architecture:** Fine-tuned **Wav2Vec 2.0** (`superb/wav2vec2-base-superb-er`).
- **Dataset:** RAVDESS Speech Audio Dataset (1,440 validated audio clips).
- **Optimization:** Artificial background noise injection during training to eliminate silence-induced overconfidence, duration padding/truncation, and acoustic volume normalization.

### 3. Text Modality (Context Emotion Recognition)

- **Architecture:** **DistilRoBERTa / DistilBERT** Sequence Classifier.
- **Transcription Pipeline:** OpenAI **Whisper** model for automated speech-to-text (STT) parsing.
- **Optimization:** Class-weight balancing, custom label alignment, and contextual nuance mapping.

### 4. Multimodal Fusion Module

- **Fusion Strategy:** Equal-weighted confidence ensemble across active inference outputs:

  $$\text{Score}_{\text{fused}} = \frac{1}{3} \times \left( P_{\text{vision}} + P_{\text{audio}} + P_{\text{text}} \right)$$

- **Output Format:** JSON dictionary containing top-predicted emotion label, combined confidence percentage, and individual model probabilities.

---

## Model Performance Benchmarks

Our fine-tuning pipeline demonstrates significant accuracy improvements over standard pre-trained baselines across all modalities:

| Modality | Base Model Architecture | Baseline Accuracy | Fine-Tuned Accuracy | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Vision (Facial Expressions)** | Vision Transformer (ViT) | 64.40% | **85.25%** | +20.85% |
| **Audio (Speech Tone)** | Wav2Vec 2.0 | 38.40% | **96.82%** | +58.42% |
| **Text (Transcribed Context)** | DistilRoBERTa | 88.36% | **96.73%** | +8.37% |

---

## Hardware & Mobile Integration

- **Hardware Device:** Vuzix Blade 2 Smart Glasses (Android OS).
- **Mobile Stack:** Kotlin, Android NDK/SDK, OkHttp3 for networking.
- **Transmission Protocol:** Secure HTTPS streaming through ngrok tunnels to Colab/Cloud GPU endpoints.
- **User Workflow:**
  1. The wearer initiates tracking via a physical touchpad tap on the glasses arm.
  2. The frame buffer and audio segment are packed and transmitted.
  3. The server runs inference and returns the JSON emotion payload within milliseconds.
  4. An intuitive, color-coded icon/text alert renders in the user's peripheral view field.

---

## Getting Started

### Prerequisites

Ensure you have the following software and drivers installed:

- **Python:** v3.10 or higher
- **PyTorch:** v2.0+ with CUDA/GPU support enabled
- **System Tools:** FFmpeg (required for audio extraction and conversion)
- **Cloud Platform (Optional):** Google Colab Pro / AWS GPU Instance for inference hosting

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/shathahsaleem/multimodal-emotion-recognition.git](https://github.com/shathahsaleem/multimodal-emotion-recognition.git )
   cd multimodal-emotion-recognition
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install opendatasets transformers torch scikit-learn pandas pillow torchvision librosa soundfile accelerate fastapi uvicorn
   ```

---

## How to Run

### 1. Evaluation & Benchmarking

Run baseline testing across pre-trained model instances prior to custom fine-tuning:

```bash
python scripts/benchmark.py --modality all
```

### 2. Model Training

To execute fine-tuning on custom vision, audio, or text datasets:

```bash
# Train Vision Transformer (ViT)
python train.py --modality vision --epochs 10 --batch-size 32

# Train Wav2Vec2 Audio Model
python train.py --modality audio --epochs 15 --batch-size 16
```

### 3. Backend API Deployment

Launch the FastAPI cloud inference service with ngrok exposure:

```bash
python server/main.py
```

### 4. Interactive Demo (Google Colab)

Open `notebooks/interactive_demo.ipynb` in Google Colab to test webcam frames and live microphone audio streams interactively through your browser.

---

## Challenges & Current Limitations

- **Single-Speaker Target Scope:** Optimized primarily for one-on-one conversations; multi-person speaker diarization is not currently integrated.
- **Discrete Emotion Categories:** Confined to four core emotional states (Neutral, Happy, Angry, Sad); complex/mixed emotions (e.g., sarcasm, anxiety) are mapped to the nearest classification bucket.
- **Hardware Triggering:** Currently relies on manual touchpad activation rather than fully continuous background monitoring to conserve battery life.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE ) file for complete details.

---

## Acknowledgments

- **Vuzix Corporation** for wearable SDK support and hardware access.
- **Hugging Face** for providing open-source transformer model baselines.
- **RAVDESS & Kaggle Datasets** for open audio-visual training resources.
