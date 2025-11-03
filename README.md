# Audio-Driven Hate Speech Detection in Telugu

**Low-resource multimodal hate speech detection leveraging acoustic and textual representations for robust moderation in Telugu.**

## Abstract
Hate speech proliferation on social platforms poses escalating safety and socio-psychological risks. Existing detection systems are skewed toward high-resource languages (e.g., English), leaving ~83M Telugu speakers underserved. Further, text-only pipelines miss prosodic and paralinguistic cues critical for identifying implicit, sarcastic, or transliterated hate. This project introduces a manually curated and annotated multimodal (audio + transcript) Telugu hate speech dataset totaling ~2 hours of speech (balanced: 1h Hate / 1h Non-Hate) with four hate subcategories (Religion, Political/National Affiliation, Gender, Personal Defamation). We benchmark classical ML, feature-based representations (OpenSMILE ComParE 2016), and state-of-the-art transformer encoders (LaBSE, mBERT, XLM-RoBERTa, Wav2Vec2, CLAP) across binary and multi-class tasks. Results show acoustic features coupled with a lightweight SVM outperform end-to-end audio transformers for binary classification (91% accuracy, 0.89 F1), while LaBSE yields strongest text-only performance (89% accuracy, 0.88 F1). Multimodal late fusion (OpenSMILE + LaBSE) sustains high robustness (89% accuracy, 0.87 F1) and demonstrates complementarity of vocal and lexical signals, establishing reproducible baselines for low-resource hate speech research.

## Problem Statement
Platforms lack effective moderation tools for low-resource languages and for spoken content where aggression, intent, and implicit targeting are conveyed through tone, pitch, and rhythm. Pure text pipelines underperform on noisy transcripts, code-mixed utterances, and transliteration artifacts. We address both the Low-Resource Gap (Telugu) and the Modality Gap (speech vs text) through systematic representation learning and fusion strategies.

## Dataset Overview
| Attribute | Hate | Non-Hate | Total |
|-----------|------|----------|-------|
| Duration (approx.) | 1 h | 1 h | 2 h |
| Utterances | 82 (Rel) / 68 (Pol) / 111 (Gen) / 132 (Defam) | 208 | 601 |
| Unique Speakers | 36 / 22 / 56 / 46 (per hate class) | 72 | — |
| Unique Words (Hate) | 3903 | — | — |
| Unique Words (Non-Hate) | — | 3772 | — |
| Vocabulary (Combined) | — | — | 6975 |

Labels (Multi-class): Religion (R), Political (P), Gender (G), Personal Defamation (C), Non-Hate (NH).

### Collection & Annotation
- Source: YouTube public content (manual curation & segmentation).
- Policy Basis: YouTube Hate Speech guidelines.
- Transcription: Human (manual), preserving code-mixed & transliterated tokens.
- Balance Strategy: Controlled sampling across subcategories.

### Preprocessing
- Audio: Resampled to 16 kHz (speech models) / 48 kHz (CLAP), loudness normalization, duration filtering (outlier analysis via ±2σ & percentile banding).
- Features (Audio): OpenSMILE ComParE 2016 Low-Level Descriptors → statistical aggregation (max, range, kurtosis, skewness, trend coefficients).
- Features (Text): Tokenization & contextual embeddings (LaBSE / mBERT / XLM-RoBERTa), truncation 128–512 tokens.
- Augmentation (Audio): Time-shift, time-stretch, Gaussian noise (class balancing).

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

### Methodology Diagram
(Add your pipeline image here)
```md
![Methodology Pipeline](./assets/methodology_pipeline.png)
```
> Place the proposed pipeline image at `assets/methodology_pipeline.png`.

## Results Summary
### Text Modality (Binary)
| Model | Accuracy | F1 |
|-------|----------|----|
| TF-IDF + XGBoost | Baseline (≈ lower than mBERT) | — |
| mBERT (fine-tuned) | 0.83 | 0.81 |
| LaBSE (fine-tuned) | **0.89** | **0.88** |

### Text Modality (Multi-class)
| Model | Accuracy | F1 |
|-------|----------|----|
| XLM-RoBERTa | 0.80 | 0.81 |
| LaBSE | 0.76 | — |

### Audio Modality (Binary)
| Model | Accuracy | F1 |
|-------|----------|----|
| OpenSMILE + SVM | **0.91** | **0.89** |
| Wav2Vec2 (XLS-R) + LSTM | < 0.90 | — |
| CLAP (audio only) | Balanced | — |

### Multimodal (Binary)
| Fusion | Accuracy | F1 |
|--------|----------|----|
| Late (OpenSMILE + LaBSE) | **0.89** | 0.87 |
| Early (CLAP joint) | Competitive | — |

### Multimodal (Multi-class)
| Fusion | Accuracy | F1 |
|--------|----------|----|
| Late (OpenSMILE + LaBSE) | 0.72 | 0.72 |
| Early (CLAP joint) | Balanced | — |

> Note: Some F1 values omitted where not explicitly computed in initial logs; update after reproducible runs.

## Key Insights
- Acoustic prosody (energy, voicing, MFCC dynamics) effectively disambiguates implicit or sarcastic hate where lexical tokens alone are ambiguous.
- Lightweight classical models on engineered audio features can outperform large pretrained transformers in constrained low-resource settings (data scarcity + overfitting risk).
- Cross-attention fusion provides richer inter-modality interaction but incurs higher computational overhead; late fusion remains robust when one modality degrades.
- LaBSE’s multilingual alignment aids transliterated Telugu tokens versus vanilla mBERT, improving binary discrimination.
- Multimodal gains are modest in balanced settings—suggesting future improvements via temporal alignment (utterance-level segmentation) & noise-robust ASR augmentation.

## Repository Structure
```
multi-modal/     # Fusion experiments (CLAP, Early/Late Fusion architectures)
unimodal/        # Individual audio or text modality notebooks
Final_Project.pdf
FINAL PPT.pptx
README.md
```

## Quick Start
```bash
# Clone repository
git clone https://github.com/<your-org>/telugu-audio-hate-speech.git
cd telugu-audio-hate-speech

# Create environment (example)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
```python
# Example: Load LaBSE for text classification
from transformers import AutoTokenizer, AutoModel
import torch
model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
text = "Sample Telugu hate speech utterance"
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
with torch.no_grad():
    emb = model(**inputs).pooler_output
print(emb.shape)
```

## Reproducibility
- Random seeds fixed (42) for splits.
- Stratified train/test (80/20).
- Provide saved feature matrices (e.g., OpenSMILE statistics, Wav2Vec2 pooled embeddings) for rapid benchmarking.
- Suggest future extension: k-fold cross-validation + class imbalance handling (focal loss / reweighting).

## Future Work
- Extend to code-mixed Telugu-English joint modeling.
- Incorporate ASR + confidence gating for noisy transcripts.
- Temporal fusion (frame-level alignment of prosodic events with token spans).
- Contrastive multimodal pretraining on unlabeled regional speech.

## Ethical Considerations
- Focus on research transparency; dataset derived from publicly accessible media.
- Recommend anonymization of speaker identities in any release.
- Avoid deploying without human moderation loop & bias auditing.

## Citation
```
@inproceedings{your2025teluguhate,
  title={Audio-Driven Multimodal Hate Speech Detection in Telugu},
  author={Your Name},
  year={2025},
  note={Manually curated multimodal dataset; baseline acoustic + transformer fusion models}
}
```

## License
Specify license (e.g., Apache 2.0) before public release.

---
> Update placeholders (image path, repo URL, missing metrics) once finalized.

