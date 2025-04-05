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

# Part B
# MFCC Feature Extraction and Comparative Analysis of Indian Languages

Language identification from speech is a critical task with applications ranging from automated transcription systems to multilingual virtual assistants. In this project, we analyze the acoustic signatures of Hindi, Tamil, and Bengali using Mel-Frequency Cepstral Coefficients (MFCCs) and evaluate the discriminative power of these features both visually and statistically. Furthermore, we develop classification models to accurately identify the spoken language from short audio segments.

---

## Dataset Description

The dataset used in this project is sourced from the [Audio Dataset with 10 Indian Languages on Kaggle](https://www.kaggle.com/datasets/balakrishcodes/audio-dataset-for-language-identification). For this assignment, a representative subset of three languages (Hindi, Tamil, and Bengali) is selected. Key preprocessing steps include:
- Truncating each audio file to a fixed duration of 3 seconds.
- Converting audio to mono-channel.
- Processing a maximum of 20 audio files per language to maintain class balance.

---

## MFCC Feature Extraction and Analysis (Task A)

### MFCC Background

MFCCs approximate the human auditory system’s response by capturing the short-term power spectrum of audio signals on a mel scale. They are widely used in speech processing and language recognition for their ability to capture timbre and phonetic characteristics.

### Extraction Process

- **Library Used:** LibROSA
- **Features Extracted:** 13 MFCC coefficients, along with first-order (delta) and second-order (delta-delta) derivatives.
- **Processing Steps:** Load the audio at its native sampling rate, normalize, and truncate to 3 seconds.
- **Output:** Structured feature vectors stored with accompanying metadata such as sample rate and file path.

### MFCC Spectrogram Visualization

MFCC spectrograms were generated for sample audio clips from Hindi, Tamil, and Bengali. Visual analysis reveals:
- **Hindi:** Dynamic fluctuations in lower coefficients, indicating varied phonetic transitions.
- **Bengali:** Smoother transitions with more stable mid-range coefficients.
- **Tamil:** Dense lower-frequency energy bands, possibly reflecting retroflex consonants and longer vowel durations.

These visual patterns indicate that MFCCs capture language-specific acoustic characteristics.

### Statistical Analysis of MFCCs

For each language:
- **Mean and Standard Deviation:** Computed for all 13 MFCC coefficients.
- **Observations:** 
  - Hindi shows more variability in the first few coefficients.
  - Tamil exhibits consistent patterns with lower variance.
  - Bengali maintains a balanced spectral profile.
  
Pairwise t-tests were performed on each MFCC coefficient between language pairs. The resulting p-values (with significance threshold set at 0.05) indicate that Hindi differs significantly from the other two languages, while Tamil and Bengali share some similarities.

### Dimensionality Reduction using PCA

Principal Component Analysis (PCA) was applied to the aggregated MFCC feature vectors (mean per sample). The first two principal components explain approximately 46% of the variance, with a scatter plot revealing:
- Distinct clusters for Hindi, Tamil, and Bengali.
- Hindi forms a clearly separate cluster, whereas Tamil and Bengali show some overlap.

### Summary of Significant Differences

A quantitative summary shows:
- **Hindi vs Tamil:** 100% of MFCC coefficients are significantly different.
- **Hindi vs Bengali:** Approximately 92% of coefficients are significantly different.
- **Tamil vs Bengali:** Around 85% show significant differences.

These statistics reinforce the observation that Hindi is acoustically more distinct compared to Tamil and Bengali.

---

## Language Classification using MFCCs (Task B)

### Formulation of Classification Task

The goal is to build a supervised machine learning system that predicts the spoken language (Hindi, Tamil, or Bengali) from the extracted MFCC features. This is formulated as a multiclass classification problem.

### Feature Engineering Strategy

For each audio sample:
- **Features:** 13 MFCCs along with their delta and delta-delta coefficients.
- **Aggregation:** Compute mean and standard deviation across time, resulting in a fixed-length vector of size 78 (13 × 6).

### Train-Test Split and Label Encoding

- **Split:** Stratified sampling with 80% training and 20% testing.
- **Label Encoding:** Convert language labels into numerical format using `LabelEncoder` from scikit-learn.

### Classification Models

Three classifiers were trained:
  
#### Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Preprocessing:** Standardization and PCA (retaining 95% variance)
- **Tuning:** Hyperparameters such as C and gamma were tuned using RandomizedSearchCV.

#### Random Forest (RF)
- **Hyperparameters:** Number of estimators, maximum depth, and minimum samples split.
- **Note:** Tree-based models are less sensitive to feature scaling.

#### Neural Network (MLP)
- **Architecture:** One or two hidden layers with ReLU or tanh activation functions.
- **Training:** Early stopping was implemented, with hyperparameter tuning over hidden layer sizes and learning rates.

### Hyperparameter Tuning Methodology

All models were tuned using `RandomizedSearchCV` with 3-fold cross-validation on a subsample of the training data. The best hyperparameters were then used to retrain the model on the full training set.

### Evaluation Metrics and Results

Models were evaluated based on:
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrices:** For class-wise prediction quality

**Best Model:**  
- **SVM Classifier**
- **Test Accuracy:** 99.62%
- **Macro-Averaged F1-Score:** 0.99

The SVM classifier achieved near-perfect classification performance with minimal misclassifications across the three languages.
