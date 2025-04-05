# Part A 

## Question 1: Speech Enhancement and Speaker Verification

### Introduction and Background

Multi-speaker environments pose unique challenges, where overlapping speech complicates both enhancement and speaker verification. Our work builds a robust system that:
- Enhances audio quality in noisy conditions.
- Verifies speaker identity with high accuracy.

This task is critical for applications in security systems, voice assistants, and automated transcription services.

### Data Preparation

We utilized two major datasets:
- **VoxCeleb1**: Contains clean speech samples used for initial evaluation.
- **VoxCeleb2**: Offers a larger, diverse speaker set for fine-tuning and extended testing.

The audio processing pipeline involves:
- Resampling audio files to a uniform sampling rate.
- Adjusting signal lengths through padding or trimming.
- Creating training and validation splits with unique speaker IDs.
- Extracting features in batches using a pre-trained feature extractor to standardize inputs.

### Model Architecture and Components

Our system leverages a pre-trained **WavLM** backbone for robust speech representation. Key components include:

- **Speaker Encoder**: 
  - Uses frozen feature extraction layers to preserve learned patterns.
  - An attentive pooling layer aggregates frame-level features into robust speaker embeddings.
  
- **ArcFace Layer**: 
  - Introduces a margin-based loss to improve class separation, critical for enhancing verification accuracy.
  
- **Low-Rank Adaptation (LoRA)**:
  - Applied to transformer layers to fine-tune a limited number of parameters, reducing computational cost while adapting to the VoxCeleb2 dataset.

Model statistics highlight the efficiency:
- **Trainable Parameters**: 221,184
- **Total Parameters**: 90,006,896
- **Trainable %**: 0.2457%

### Speaker Verification Evaluation

Evaluation is conducted on VoxCeleb1 trial pairs using the following metrics:
- **Speaker Identification Accuracy**
- **Equal Error Rate (EER)**
- **True Accept Rate (TAR) at low False Accept Rates (FAR)**
- **Minimum Detection Cost Function (MinDCF)**
- **Area Under the ROC Curve (AUC)**

### Results and Analysis

| Metric                        | Pre-trained | Fine-tuned | Improvement     |
|-------------------------------|-------------|------------|-----------------|
| Speaker Identification (%)    | 56.94       | 86.11      | +29.17          |
| EER (%)                       | 37.81       | 20.02      | -17.79          |
| TAR@1% FAR (%)                | 6.26        | 27.15      | +20.89          |
| TAR@0.1% FAR (%)              | 1.29        | 11.21      | +9.92           |
| AUC                           | 0.67        | 0.8846     | +0.2146         |

*Key observations:*
- Significant improvement in speaker identification accuracy.
- A reduction in EER indicates fewer verification errors.
- Enhanced TAR values demonstrate robust performance under stringent conditions.

### Visualization and Analysis

Visual aids include:
- **ROC and DET Curves**: Indicate improved TPR and lower error rates for the fine-tuned model.
- **KDE Plots**: Show clearer separation between genuine and impostor score distributions.
- **t-SNE Embeddings**: Highlight distinct clustering for the fine-tuned model, demonstrating enhanced discriminative power.

### Conclusion

The fine-tuning process using LoRA and ArcFace loss markedly improved the model’s performance:
- **Speaker Identification Accuracy** increased from 56.94% to 86.11%.
- **Verification metrics** such as EER and TAR show substantial gains.

---

## Question 2: Speaker Separation and Speech Enhancement Experiments

### Multi-Speaker Dataset Creation

To simulate realistic conditions, a multi-speaker dataset was created by mixing utterances from two different speakers:
- **Training Scenario**: First 50 identities (sorted in ascending order) from VoxCeleb2.
- **Testing Scenario**: Next 50 identities.
- Audio files were mixed to create overlapping speech signals.

### Speaker Separation and Speech Enhancement

A pre-trained **SepFormer** model was used for separation:
- **Process**: The model separates the mixed signals into individual speaker outputs.
- **Enhancement**: Subsequent evaluation is performed on the separated outputs to improve audio quality.

The separation quality is assessed using:
- **Signal to Interference Ratio (SIR)**
- **Signal to Artefacts Ratio (SAR)**
- **Signal to Distortion Ratio (SDR)**
- **Perceptual Evaluation of Speech Quality (PESQ)**

### Speaker Identification on Enhanced Speech

The separated signals are fed into both the pre-trained and fine-tuned speaker verification models:
- **Evaluation Metric**: Rank-1 identification accuracy.
- This measures the percentage of signals correctly identified after separation.

### Results and Analysis

**Separation Metrics:**

| Metric              | Value  |
|---------------------|--------|
| Average SDR (dB)    | 5.32   |
| Average SIR (dB)    | 18.68  |
| Average SAR (dB)    | 4.92   |
| Average PESQ        | 1.23   |

**Speaker Identification Accuracy:**

| Model               | Accuracy (%) | Improvement   |
|---------------------|--------------|---------------|
| Pre-trained Model   | 12.50        |               |
| Fine-tuned Model    | 16.80        | +4.30         |

*Observations:*
- The separation process, while effective in reducing interference, still leaves artifacts and distortions.
- Fine-tuning leads to improved identification, albeit with modest gains due to the complexity of overlapping speech.

### Visualization and Analysis

- **Confusion Matrices**: Display misclassifications with fine-tuned models showing higher diagonal concentration.
- **Spectrograms**: Confirm that despite improvements, artifacts and overlapping frequency components persist.

---

## Question 3: Integrated Pipeline for Speaker Identification and Speech Enhancement

### Overview

This experiment develops an integrated pipeline that performs both:
- **Speech Separation and Enhancement**: Using SepFormer followed by a convolution-based enhancement module.
- **Speaker Identification**: Via a fine-tuned WavLM-based model with LoRA adaptation.

### Model Architecture and Components

**Joint Pipeline Components:**

1. **Pre-trained SepFormer**: Separates overlapping speech signals.
2. **Enhancement Module**:
   - A convolutional network that reduces noise and interference.
3. **Speaker Verification Module**:
   - A WavLM-based model with LoRA adaptation and attention-based aggregation.
   - Integrates multiple loss functions (reconstruction, embedding similarity, spectral loss) for joint optimization.

This integration ensures that separation is optimized for downstream speaker identification.

### Evaluation Metrics

The joint system is evaluated on:
- **Separation Metrics**: SDR, SIR, SAR, and PESQ.
- **Speaker Identification Metric**: Rank-1 accuracy for both pre-trained and fine-tuned verification models.

### Results and Analysis

**Joint Pipeline Separation Performance:**

| Metric       | Value   |
|--------------|---------|
| SDR (dB)     | 7.83    |
| SIR (dB)     | 21.35   |
| SAR (dB)     | 5.32    |
| PESQ         | 1.22    |

**Speaker Identification Performance:**

| Model              | Accuracy | Improvement  |
|--------------------|----------|--------------|
| Pre-trained Model  | 0.14     |              |
| Fine-tuned Model   | 0.15     | +0.1         |

*Key points:*
- The joint pipeline shows improved SDR and SIR, indicating better suppression of interference.
- The overall speaker identification improvement is modest, suggesting that while separation quality has improved, further fine-tuning is required.

### Comparison with Previous Results

- The joint pipeline yields higher SDR and SIR compared to the earlier separation experiments.
- Speaker identification accuracy shows only a slight gain over the pre-trained model, indicating room for improvement in the integrated approach.

### Conclusion and Future Work

The integrated pipeline successfully combines speech separation and speaker verification:
- **Improvements**: Enhanced separation quality and modest gains in speaker identification.
- **Challenges**: Residual artifacts (reflected in PESQ and SAR) and modest identification improvements point to the need for additional refinement.
- **Future Directions**:
  - Explore more advanced separation architectures.
  - Incorporate multi-task learning strategies to optimize both separation and identification simultaneously.
  - Use adversarial training to further reduce artifacts.
  - Expand the dataset to include more varied acoustic environments.

---

## References

1. Nagrani, A., Chung, J. S., & Zisserman, A. (2017). *VoxCeleb: a large-scale speaker identification dataset.*
2. Chung, J. S., Nagrani, A., & Zisserman, A. (2018). *VoxCeleb2: Deep speaker recognition.*
3. Chen, N., et al. (2021). *WavLM: Large-Scale Pre-trained Models for Speech.* Microsoft Research.
4. Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.*
5. Subakan, C., et al. (2021). *SepFormer: Transformer based separation for speech separation.* [GitHub Repository](https://github.com/speechbrain/sepformer-whamr)
6. Liu, L., et al. *Parameter-Efficient Fine-Tuning (PEFT) Library.* [GitHub Repository](https://github.com/huggingface/peft)
7. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A next-generation hyperparameter optimization framework.*
8. Rafii, Z., et al. *mir_eval: A library for music and audio evaluation.* [GitHub Repository](https://github.com/craffel/mir_eval)
9. ITU-T Recommendation P.862. (2001). *Perceptual evaluation of speech quality (PESQ): An objective method for end-to-end speech quality assessment.*
10. Paszke, A., et al. *Torchaudio: An audio library for PyTorch.* [PyTorch](https://pytorch.org/audio/)
11. Wolf, T., et al. (2020). *Transformers: State-of-the-art Natural Language Processing.* [GitHub Repository](https://github.com/huggingface/transformers)
12. Palanisamy, R., et al. *SpeechBrain: An open-source and all-in-one speech toolkit.* [SpeechBrain](https://speechbrain.github.io/)

---

This README provides a comprehensive overview of our methodology, experiments, and the results obtained in the multi-speaker speech enhancement and verification tasks. For further details, please refer to the full report in the [M23CSA001_Ques1.pdf](https://github.com/bahetiaditi/M23CSA001_Speech_Assignment2.git).

Feel free to explore the code, replicate the experiments, and contribute to the project!
