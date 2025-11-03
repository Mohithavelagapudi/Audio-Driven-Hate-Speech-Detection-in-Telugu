# 🎧 Audio-Driven Hate Speech Detection in Telugu

**Low-resource multimodal hate speech detection leveraging acoustic and textual representations for robust moderation in Telugu.**
----

## 🧩 Abstract

Despite rapid advances in hate speech detection for English, Telugu — a low-resource language with ~83M speakers — lacks any publicly available annotated dataset for this task.
This project bridges that gap by building the first manually annotated multimodal Telugu hate speech dataset comprising 2 hours of YouTube audio–text pairs.
We systematically evaluate speech-based, text-based, and fusion-based models. Results show that OpenSMILE + SVM achieves 91% accuracy (F1 = 0.89) for audio, while LaBSE yields 89% accuracy (F1 = 0.88) for text.
Multimodal fusion further enhances performance, demonstrating the complementary power of acoustic and textual cues in identifying hate speech.

----

## 🎯 Problem Statement

1. Low-Resource Language Gap: Telugu lacks high-quality datasets and pretrained models for hate speech detection.

2. Modality Gap: Text-only models miss vocal cues such as tone, sarcasm, and aggression, which are critical for accurate classification.

This work develops a robust multimodal framework combining both audio and text signals to detect hate speech more effectively.
----

📊 The Dataset

The dataset was created as part of the **DravLangGuard** initiative and is meticulously annotated for hate speech.

### Data Collection & Annotation

*   **Source**: Audio clips were gathered from YouTube channels with over 50,000 subscribers to ensure realistic, in-the-wild data.
*   **Annotation**: Three native Telugu speakers with postgraduate degrees performed the annotation. They first classified content into **Hate** and **Non-Hate**. The hate speech was further categorized into four subclasses based on YouTube's hate speech policy.
*   **Reliability**: The inter-annotator agreement was measured at **~0.79 using Cohen's Kappa**, indicating a high degree of reliability.

### Dataset Statistics

The dataset is balanced between hate and non-hate content, with a detailed breakdown of hate speech categories.

| Class | Sub-Class | Short Label | No. of Samples | Total Duration (min) |
| :---- | :-------------------- | :---------: | :------------: | :------------------: |
| **Hate (H)** | Gender | G | 111 | 15.75 |
| | Religion | R | 82 | 15.49 |
| | Political / Nationality | P | 68 | 14.90 |
| | Personal Defamation | C | 133 | 14.90 |
| **Non-Hate (NH)**| Non-Hate | N | 208 | 60.00 |

----

### Preprocessing
- Audio: Resampled to 16 kHz (speech models) / 48 kHz (CLAP), loudness normalization, duration filtering (outlier analysis via ±2σ & percentile banding).
- Features (Audio): OpenSMILE ComParE 2016 Low-Level Descriptors → statistical aggregation (max, range, kurtosis, skewness, trend coefficients).
- Features (Text): Tokenization & contextual embeddings (LaBSE / mBERT / XLM-RoBERTa), truncation 128–512 tokens.
- Augmentation (Audio): Time-shift, time-stretch, Gaussian noise (class balancing).

----

## Methodology
### Unimodal Pipelines
1. Text: Classical TF-IDF + XGBoost / SVM; Transformer fine-tuning (mBERT, XLM-RoBERTa, LaBSE).
2. Audio: Wav2Vec2 (XLS-R), Indic Wav2Vec2 (emotion pretraining), Audio Spectrogram Transformer (AST), OpenSMILE + classical classifiers (SVM, RF, XGBoost, MLP), sequence heads (LSTM / 1D-CNN).



### Multimodal Fusion Strategies
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

### Speech-Based Model Performance

### Multimodals (Early-Fusion) Performance

### Multimodals (Late-Fusion) Performance

----

## Key Insights
- Acoustic prosody (energy, voicing, MFCC dynamics) effectively disambiguates implicit or sarcastic hate where lexical tokens alone are ambiguous.
- Lightweight classical models on engineered audio features can outperform large pretrained transformers in constrained low-resource settings (data scarcity + overfitting risk).
- Cross-attention fusion provides richer inter-modality interaction but incurs higher computational overhead; late fusion remains robust when one modality degrades.
- LaBSE’s multilingual alignment aids transliterated Telugu tokens versus vanilla mBERT, improving binary discrimination.
- Multimodal gains are modest in balanced settings—suggesting future improvements via temporal alignment (utterance-level segmentation) & noise-robust ASR augmentation.

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

