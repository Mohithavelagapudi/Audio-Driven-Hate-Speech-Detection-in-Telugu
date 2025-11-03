# 🎧 Audio-Driven Hate Speech Detection in Telugu

<p align="center"> <img src="https://img.shields.io/badge/Language-Telugu-blueviolet?style=for-the-badge"/> <img src="https://img.shields.io/badge/Domain-Hate_Speech_Detection-red?style=for-the-badge"/> <img src="https://img.shields.io/badge/Modality-Multimodal_(Audio+Text)-orange?style=for-the-badge"/> <img src="https://img.shields.io/badge/Dataset_Size-2_Hours-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/Accuracy-91%25_(Audio)_%7C_89%25_(Text)-brightgreen?style=for-the-badge"/> <img src="https://img.shields.io/badge/F1_Score-0.89-lightgrey?style=for-the-badge"/> 
  
**Low-resource multimodal hate speech detection leveraging acoustic and textual representations for robust moderation in Telugu.**

----

### 🚀 Overview

While hate speech detection has progressed rapidly for English, Telugu — with over 83 million speakers — still lacks annotated resources.
This project introduces the first multimodal Telugu hate speech dataset and a suite of audio-, text-, and fusion-based models for comprehensive detection.

### 🧠 Core Highlights

- 🗂️ First Telugu hate-speech dataset (2 hours of annotated audio–text pairs).

- 🔊 Multimodal pipeline integrating acoustic and textual cues.

- ⚙️ Evaluated OpenSMILE, Wav2Vec2, LaBSE, and XLM-R baselines.

- 🎯 Achieved 91 % accuracy (audio) and 89 % (text); fusion improved robustness.

----

## 🧩 Abstract

This study fills a critical resource gap in Telugu hate-speech detection.
A manually annotated 2-hour multimodal dataset was curated from YouTube.
Acoustic (OpenSMILE + SVM) and textual (LaBSE) models achieved 91 % and 89 % accuracy, respectively.
Fusion approaches highlight the complementary role of vocal prosody and linguistic cues.

----

## 🎯 Problem Statement

| Challenge                | Description                                                                   |
| ------------------------ | ----------------------------------------------------------------------------- |
| 🗣️ **Low-Resource Gap** | Telugu lacks labeled corpora and pretrained models for hate-speech detection. |
| 🔊 **Modality Gap**      | Text-only systems ignore vocal signals (tone, sarcasm, aggression).           |

💡 Goal: Develop a multimodal framework combining speech and text for richer, context-aware classification.

----

### 📊 Dataset: DravLangGuard

| Attribute                     | Description                            |
| ----------------------------- | -------------------------------------- |
| **Source**                    | YouTube (≥ 50 K subscribers)           |
| **Annotators**                | 3 native Telugu postgraduates          |
| **Classes**                   | Hate / Non-Hate (4 sub-types for Hate) |
| **Inter-Annotator Agreement** | 0.79 (Cohen’s κ)                       |

### Dataset Composition

The dataset is balanced between hate and non-hate content, with a detailed breakdown of hate speech categories.

| Class | Sub-Class | Short Label | No. of Samples | Total Duration (min) |
| :---- | :-------------------- | :---------: | :------------: | :------------------: |
| **Hate (H)** | Gender | G | 111 | 15.75 |
| | Religion | R | 82 | 15.49 |
| | Political / Nationality | P | 68 | 14.90 |
| | Personal Defamation | C | 133 | 14.90 |
| **Non-Hate (NH)**| Non-Hate | N | 208 | 60.00 |

----

### 🧰 Preprocessing Pipeline

🎵 Audio

- Resampled (16 kHz / 48 kHz), loudness-normalized, duration-filtered.

- Extracted OpenSMILE ComParE 2016 LLDs → statistical aggregates.

- Augmentation: time-shift, stretch, Gaussian noise (class balancing).

✍️ Text

- Tokenization & contextual embeddings (LaBSE, mBERT, XLM-R).

- Sequence truncation (128–512 tokens).

- Handles transliterated Telugu effectively via LaBSE multilingual alignment.

----

## 🧮 Methodology

### 🔹 Unimodal Pipelines

| Modality     | Methods                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------- |
| 🧾 **Text**  | TF-IDF + XGBoost / SVM · Transformer fine-tuning (mBERT, XLM-R, LaBSE)                            |
| 🔊 **Audio** | OpenSMILE + SVM / RF / XGBoost / MLP · Wav2Vec2 (XLS-R, Indic) · AST · LSTM/1D-CNN sequence heads |


<p align="center">
  <img src="HateSpeech_images/image (45).png" alt="" width="1000"/>
</p>


### 🔸 Multimodal Fusion Strategies

| Strategy | Description | Implemented Models |
|----------|-------------|--------------------|
| Early Fusion (Feature-Level) | Concatenate / project audio & text embeddings into shared latent space before joint classification. | OpenSMILE + LaBSE (Attention), Wav2Vec2 + XLM-R, CLAP (joint audio-text encoders) |
| Late Fusion (Decision-Level) | Independent modality-specific classifiers; logits combined (averaging / voting). | OpenSMILE + LaBSE, Wav2Vec2 + XLM-R |
| Intermediate / Cross-Attention | Audio & text projected; multi-head attention exchanges contextual cues before pooling. | Custom cross-attention head (OpenSMILE ↔ LaBSE) |

### Model Components
- Projection Layers: LayerNorm + Linear → common_dim (256–512) + ReLU.
- Cross-Attention: MultiHeadAttention (n_heads=2–8) on modality token pair sequence.
- Classifier: Dropout (p=0.3–0.4) + Linear stack → softmax.
- Optimization: AdamW (lr 1e-5–2e-4), weight decay 0.01, stratified 80/20 splits.
- Metrics: Accuracy, Macro F1, Confusion Matrix, Class-wise Precision/Recall.

----

## 📈 Results Summary

### Text Based Model Performance

<p align="center">
  <img src="HateSpeech_images/image (47).png" alt="" width="1000"/>
</p>

### Speech-Based Model Performance

<p align="center">
  <img src="HateSpeech_images/image (46).png" alt="" width="1000"/>
</p>

### Multimodals (Early-Fusion) Performance

<p align="center">
  <img src="HateSpeech_images/image (48).png" alt="" width="1000"/>
</p>

### Multimodals (Late-Fusion) Performance

<p align="center">
  <img src="HateSpeech_images/image (49).png" alt="" width="1000"/>
</p>

----

## Key Insights
- Acoustic prosody (energy, voicing, MFCC dynamics) effectively disambiguates implicit or sarcastic hate where lexical tokens alone are ambiguous.
- Lightweight classical models on engineered audio features can outperform large pretrained transformers in constrained low-resource settings (data scarcity + overfitting risk).
- Cross-attention fusion provides richer inter-modality interaction but incurs higher computational overhead; late fusion remains robust when one modality degrades.
- LaBSE’s multilingual alignment aids transliterated Telugu tokens versus vanilla mBERT, improving binary discrimination.
- Multimodal gains are modest in balanced settings—suggesting future improvements via temporal alignment (utterance-level segmentation) & noise-robust ASR augmentation.

----

### 🧠 Tech Stack

| Category          | Tools / Models                         |
| ----------------- | -------------------------------------- |
| **Audio**         | OpenSMILE · Wav2Vec2 · AST . CLAP          |
| **Text**          | LaBSE · XLM-R · mBERT . TF-IDF               |
| **ML / DL**       | SVM · XGBoost · PyTorch · Transformers . RandomForest |
| **Evaluation**    | Accuracy · Macro F1 · Confusion Matrix |
| **Visualization** | Matplotlib · Seaborn                   |
| **Audio Processing**| Librosa . Torchaudio                 |


----


## 🙏 Citation

If you use this dataset or code in your research, please cite the original paper:

```bibtex
@inproceedings{kumar2024audio,
  title={Audio Driven Detection of Hate Speech in Telugu: Toward Ethical and Secure CPS},
  author={Kumar M, Santhosh and Ravula P, Sai and Teja M, Prasanna and Surya J, Ajay and V, Mohitha and Lal G, Jyothish},
  year={2024}
}
```

