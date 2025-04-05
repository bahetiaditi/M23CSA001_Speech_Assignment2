import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json  
# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# Device configuration
COMPUTING_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {COMPUTING_DEVICE} device")

# Data directories and output paths
VOXCELEB2_DIR = "/home/m23csa001/Data/vox2_test_aac-001/aac"
VOXCELEB1_DIR = "/home/m23csa001/Data/vox1-20250328T141257Z-001/vox1/vox1_test_wav/wav"
VOXCELEB1_TRIALS = "/home/m23csa001/Data/trials.txt"
RESULTS_DIR = "./results_optuna"
MODEL_DIR = os.path.join(RESULTS_DIR, "saved_models")
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Default hyperparameters
DEFAULT_CONFIG = {
    "batch_size": 8,
    "learning_rate": 3e-5,
    "num_epochs": 10,
    "max_audio_duration": 6,  # seconds
    "min_audio_duration": 3,  # seconds
    "sampling_rate": 16000,
    "embedding_dimension": 256,
    "margin": 0.2,
    "scale_factor": 16.0,
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05
}

class AudioDataset(Dataset):
    def __init__(self, audio_dir, speaker_ids, feature_extractor, is_val=False, max_samples=None):
        self.audio_dir = audio_dir
        self.speaker_ids = speaker_ids
        self.feature_extractor = feature_extractor
        self.is_val = is_val
        self.samples = []
        speaker_count = {s: 0 for s in speaker_ids}
        
        for speaker in tqdm(speaker_ids, desc=f"{'Validation' if is_val else 'Training'} dataset preparation"):
            speaker_path = os.path.join(audio_dir, speaker)
            if not os.path.exists(speaker_path):
                continue
            for root, _, files in os.walk(speaker_path):
                for file in files:
                    if file.endswith(".m4a"):
                        if max_samples and speaker_count[speaker] >= max_samples:
                            continue
                        audio_path = os.path.join(root, file)
                        label = val_speaker_to_id.get(speaker, -1) if is_val else train_speaker_to_id[speaker]
                        self.samples.append({
                            "audio_path": audio_path,
                            "label": label,
                            "speaker": speaker
                        })
                        speaker_count[speaker] += 1
        print(f"Dataset created with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_signal = self.load_audio(sample["audio_path"])
        if audio_signal is None:
            fallback_idx = random.randint(0, len(self) - 1)
            return self[fallback_idx]
        return {
            "audio_signal": audio_signal,
            "label": sample["label"],
            "speaker": sample["speaker"],
            "audio_path": sample["audio_path"]
        }
    
    def load_audio(self, file_path):
        try:
            signal, sr = torchaudio.load(file_path)
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            if sr != DEFAULT_CONFIG["sampling_rate"]:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=DEFAULT_CONFIG["sampling_rate"])
                signal = resampler(signal)
            
            if signal.shape[1] < DEFAULT_CONFIG["min_audio_duration"] * DEFAULT_CONFIG["sampling_rate"]:
                padding = DEFAULT_CONFIG["min_audio_duration"] * DEFAULT_CONFIG["sampling_rate"] - signal.shape[1]
                signal = F.pad(signal, (0, padding))
            
            if signal.shape[1] > DEFAULT_CONFIG["max_audio_duration"] * DEFAULT_CONFIG["sampling_rate"]:
                signal = signal[:, :int(DEFAULT_CONFIG["max_audio_duration"] * DEFAULT_CONFIG["sampling_rate"])]
                
            return signal.squeeze(0)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

def batch_processor(feature_extractor):
    def process_batch(batch):
        valid_items = [x for x in batch if x["audio_signal"] is not None]
        if not valid_items:
            return None
        
        audio_signals = [x["audio_signal"].numpy() for x in valid_items]
        labels = torch.tensor([x["label"] for x in valid_items])
        speakers = [x["speaker"] for x in valid_items]
        paths = [x["audio_path"] for x in valid_items]
        
        inputs = feature_extractor(
            audio_signals,
            sampling_rate=DEFAULT_CONFIG["sampling_rate"],
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        return {
            "input_features": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
            "speaker": speakers,
            "audio_path": paths
        }
    return process_batch

class AttentivePooling(nn.Module):
    def __init__(self, input_dim=1024, output_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.PReLU(),
            nn.Linear(input_dim, output_dim)
        )
        # Initialize weights
        for layer in [self.attention[0], self.attention[2], 
                     self.projector[0], self.projector[3]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, features):
        attention_weights = self.attention(features)
        context_vector = torch.sum(attention_weights * features, dim=1)
        return self.projector(context_vector)

class ArcFaceLayer(nn.Module):
    def __init__(self, embed_dim, num_classes, scale=DEFAULT_CONFIG["scale_factor"], margin=DEFAULT_CONFIG["margin"]):
        super(ArcFaceLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        cosine_similarity = F.linear(normalized_embeddings, normalized_weights)
        cosine_similarity = torch.clamp(cosine_similarity, -1.0 + 1e-7, 1.0 - 1e-7)
        
        theta = torch.acos(cosine_similarity)
        target_mask = torch.zeros_like(cosine_similarity)
        target_mask.scatter_(1, labels.view(-1, 1), 1)
        
        theta_with_margin = theta + self.margin * target_mask
        modified_cosine = torch.cos(theta_with_margin)
        
        final_logits = self.scale * (target_mask * modified_cosine + (1 - target_mask) * cosine_similarity)
        return F.cross_entropy(final_logits, labels)

class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers, pretrained_model="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm_encoder = WavLMModel.from_pretrained(pretrained_model)
        # Freeze feature extraction layers
        for param in self.wavlm_encoder.feature_extractor.parameters():
            param.requires_grad = False
            
        self.pooling = AttentivePooling(
            input_dim=self.wavlm_encoder.config.hidden_size,
            output_dim=DEFAULT_CONFIG["embedding_dimension"]
        )
        self.arc_face = ArcFaceLayer(
            DEFAULT_CONFIG["embedding_dimension"], 
            num_speakers
        )
    
    def forward(self, input_features, attention_mask=None, labels=None):
        encoder_outputs = self.wavlm_encoder(
            input_values=input_features, 
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state
        speaker_embeddings = self.pooling(hidden_states)
        normalized_embeddings = F.normalize(speaker_embeddings, p=2, dim=1)
        
        if labels is not None:
            loss = self.arc_face(normalized_embeddings, labels)
            return normalized_embeddings, loss
        return normalized_embeddings

def configure_lora(model, rank, alpha, dropout):
    current_device = next(model.parameters()).device
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["k_proj", "v_proj", "q_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    model.wavlm_encoder.encoder = PeftModel(model.wavlm_encoder.encoder, lora_config)
    model.wavlm_encoder.encoder.print_trainable_parameters()
    
    return model.to(current_device)

def compute_equal_error_rate(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold

def compute_min_dcf(y_true, scores, p_target=0.01, cost_miss=1, cost_fa=1):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    
    norm_cost_miss = cost_miss * p_target
    norm_cost_fa = cost_fa * (1 - p_target)
    
    dcf_values = norm_cost_miss * fnr + norm_cost_fa * fpr
    
    min_dcf_idx = np.argmin(dcf_values)
    min_dcf = dcf_values[min_dcf_idx]
    optimal_threshold = thresholds[min_dcf_idx]
    
    return min_dcf, optimal_threshold

def compute_tar_at_far(y_true, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = np.argmin(np.abs(fpr - target_far))
    return tpr[idx], thresholds[idx]

def visualize_roc_curve(y_true, baseline_scores, improved_scores, save_path):
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true, baseline_scores)
    fpr_improved, tpr_improved, _ = roc_curve(y_true, improved_scores)
    
    auc_baseline = roc_auc_score(y_true, baseline_scores)
    auc_improved = roc_auc_score(y_true, improved_scores)
    
    eer_baseline, _ = compute_equal_error_rate(y_true, baseline_scores)
    eer_improved, _ = compute_equal_error_rate(y_true, improved_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_baseline, tpr_baseline, label=f'Baseline (AUC={auc_baseline:.4f}, EER={eer_baseline:.4f})')
    plt.plot(fpr_improved, tpr_improved, label=f'Improved (AUC={auc_improved:.4f}, EER={eer_improved:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")

def visualize_det_curve(y_true, baseline_scores, improved_scores, save_path):
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true, baseline_scores)
    fpr_improved, tpr_improved, _ = roc_curve(y_true, improved_scores)
    
    fnr_baseline = 1 - tpr_baseline
    fnr_improved = 1 - tpr_improved
    
    plt.figure(figsize=(10, 8))
    plt.loglog(fpr_baseline, fnr_baseline, label='Baseline')
    plt.loglog(fpr_improved, fnr_improved, label='Improved')
    plt.grid(True, alpha=0.3)
    plt.xlabel('False Alarm Rate (log scale)')
    plt.ylabel('Miss Rate (log scale)')
    plt.title('DET Curve Comparison')
    plt.legend()
    plt.savefig(save_path)
    print(f"DET curve saved to {save_path}")

def visualize_score_distribution(y_true, scores, save_path, title="Score Distribution"):
    plt.figure(figsize=(10, 6))
    
    genuine_scores = scores[y_true == 1]
    impostor_scores = scores[y_true == 0]
    
    sns.kdeplot(genuine_scores, label="Genuine (Same Speaker)", shade=True)
    sns.kdeplot(impostor_scores, label="Impostor (Different Speaker)", shade=True)
    
    plt.title(title)
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path)
    print(f"Score distribution saved to {save_path}")

def visualize_embeddings(embeddings, labels, save_path, method='tsne', n_components=2):
    if embeddings.shape[0] > 1000:
        indices = np.random.choice(embeddings.shape[0], 1000, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = [labels[i] for i in indices]
    else:
        embeddings_subset = embeddings
        labels_subset = labels
    
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=RANDOM_SEED)
        title = "t-SNE Visualization of Speaker Embeddings"
    else:  # PCA
        reducer = PCA(n_components=n_components)
        title = "PCA Visualization of Speaker Embeddings"
    
    reduced_embeddings = reducer.fit_transform(embeddings_subset)
    
    plt.figure(figsize=(12, 10))
    unique_labels = list(set(labels_subset))
    
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, lbl in enumerate(labels_subset) if lbl == label]
        plt.scatter(
            reduced_embeddings[indices, 0], 
            reduced_embeddings[indices, 1], 
            color=cmap(i), 
            label=f"Speaker {label}", 
            alpha=0.7, 
            s=50
        )
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Embedding visualization saved to {save_path}")

def train_epoch(model, data_loader, optimizer, epoch, logger):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    grad_scaler = torch.cuda.amp.GradScaler()
    
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
            
        inputs = batch["input_features"].to(COMPUTING_DEVICE)
        masks = batch["attention_mask"].to(COMPUTING_DEVICE)
        targets = batch["labels"].to(COMPUTING_DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            embeddings, loss = model(inputs, attention_mask=masks, labels=targets)
        
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        epoch_loss += loss.item()
        
        # Calculate identification accuracy using cosine similarity
        with torch.no_grad():
            sim_matrix = F.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
            )
            
            sim_matrix.fill_diagonal_(-float('inf'))
            predictions = sim_matrix.argmax(dim=1)
            predicted_labels = targets[predictions]
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct_predictions / total_samples if total_samples > 0 else 0
        })
        
        if batch_idx % 10 == 0:
            global_step = epoch * len(data_loader) + batch_idx
            logger.add_scalar('train/loss', loss.item(), global_step)
            logger.add_scalar('train/acc', 100. * correct_predictions / total_samples if total_samples > 0 else 0, global_step)
    
    avg_loss = epoch_loss / len(data_loader)
    avg_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_acc

def evaluate_identification(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    collected_embeddings = []
    collected_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating identification"):
            if batch is None:
                continue
                
            inputs = batch["input_features"].to(COMPUTING_DEVICE)
            masks = batch["attention_mask"].to(COMPUTING_DEVICE)
            targets = batch["labels"].to(COMPUTING_DEVICE)
            
            embeddings = model(inputs, attention_mask=masks)
            
            collected_embeddings.append(embeddings)
            collected_labels.append(targets)
    
    if not collected_embeddings:
        return 0.0, None, None, None
        
    collected_embeddings = torch.cat(collected_embeddings, dim=0)
    collected_labels = torch.cat(collected_labels, dim=0)
    
    sim_matrix = F.cosine_similarity(
        collected_embeddings.unsqueeze(1), collected_embeddings.unsqueeze(0), dim=2
    )
    sim_matrix.fill_diagonal_(-float('inf'))
    predictions = sim_matrix.argmax(dim=1)
    predicted_labels = collected_labels[predictions]
    
    correct = (predicted_labels == collected_labels).sum().item()
    total = collected_labels.size(0)
    
    cm = confusion_matrix(collected_labels.cpu().numpy(), predicted_labels.cpu().numpy())
    
    return 100. * correct / total, collected_embeddings.cpu().numpy(), collected_labels.cpu().numpy(), cm

def process_verification_pairs(model, trial_file, audio_dir, feature_extractor):
    model.eval()
    
    trial_pairs = []
    with open(trial_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                label = int(parts[0])
                path1 = os.path.join(audio_dir, parts[1])
                path2 = os.path.join(audio_dir, parts[2])
                trial_pairs.append((label, path1, path2))
    
    print(f"Loaded {len(trial_pairs)} verification trials")
    
    embedding_cache = {}
    similarity_scores = []
    ground_truth = []
    
    for i, (label, path1, path2) in enumerate(tqdm(trial_pairs, desc="Processing verification trials")):
        for audio_path in [path1, path2]:
            if audio_path not in embedding_cache:
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    if sr != DEFAULT_CONFIG["sampling_rate"]:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=DEFAULT_CONFIG["sampling_rate"])
                        waveform = resampler(waveform)
                    
                    if waveform.shape[1] < DEFAULT_CONFIG["min_audio_duration"] * DEFAULT_CONFIG["sampling_rate"]:
                        padding = DEFAULT_CONFIG["min_audio_duration"] * DEFAULT_CONFIG["sampling_rate"] - waveform.shape[1]
                        waveform = F.pad(waveform, (0, padding))
                    if waveform.shape[1] > DEFAULT_CONFIG["max_audio_duration"] * DEFAULT_CONFIG["sampling_rate"]:
                        waveform = waveform[:, :int(DEFAULT_CONFIG["max_audio_duration"] * DEFAULT_CONFIG["sampling_rate"])]
                    
                    inputs = feature_extractor(
                        waveform.squeeze().numpy(),
                        sampling_rate=DEFAULT_CONFIG["sampling_rate"],
                        return_tensors="pt",
                        padding=True,
                        return_attention_mask=True
                    )
                    
                    with torch.no_grad():
                        embedding = model(
                            inputs.input_values.to(COMPUTING_DEVICE),
                            attention_mask=inputs.attention_mask.to(COMPUTING_DEVICE)
                        )
                    embedding_cache[audio_path] = embedding.cpu()
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    embedding_cache[audio_path] = None
        
        if embedding_cache[path1] is None or embedding_cache[path2] is None:
            continue
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            embedding_cache[path1],
            embedding_cache[path2],
            dim=1
        ).item()
        
        similarity_scores.append(similarity)
        ground_truth.append(label)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{len(trial_pairs)} trials")
    
    return np.array(similarity_scores), np.array(ground_truth)

def compute_verification_metrics(scores, labels):
    eer, eer_threshold = compute_equal_error_rate(labels, scores)
    
    tar_at_1_far, threshold_1_far = compute_tar_at_far(labels, scores, 0.01)  # TAR@1%FAR
    tar_at_01_far, threshold_01_far = compute_tar_at_far(labels, scores, 0.001)  # TAR@0.1%FAR
    
    min_dcf, dcf_threshold = compute_min_dcf(labels, scores)
    
    auc = roc_auc_score(labels, scores)
    
    return {
        "EER (%)": eer * 100,
        "EER Threshold": eer_threshold,
        "TAR@1%FAR (%)": tar_at_1_far * 100,
        "TAR@0.1%FAR (%)": tar_at_01_far * 100,
        "MinDCF": min_dcf,
        "DCF Threshold": dcf_threshold,
        "AUC": auc
    }

def objective(trial):
    config = {
        "batch_size": trial.suggest_int("batch_size", 4, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "margin": trial.suggest_float("margin", 0.1, 0.5),
        "scale_factor": trial.suggest_float("scale_factor", 10.0, 30.0),
        "lora_rank": trial.suggest_int("lora_rank", 2, 8),
        "lora_alpha": trial.suggest_int("lora_alpha", 4, 16),
        "lora_dropout": trial.suggest_float("lora_dropout", 0.01, 0.1)
    }
    
    for key, value in config.items():
        DEFAULT_CONFIG[key] = value
    
    max_speakers_for_optuna = 20
    max_samples_per_speaker = 20
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    optuna_train_speakers = train_speaker_ids[:max_speakers_for_optuna]
    optuna_val_speakers = val_speaker_ids[:5]  # Just use 5 validation speakers
    
    train_data = AudioDataset(
        VOXCELEB2_DIR, 
        optuna_train_speakers, 
        feature_extractor, 
        max_samples=max_samples_per_speaker
    )
    val_data = AudioDataset(
        VOXCELEB2_DIR, 
        optuna_val_speakers, 
        feature_extractor, 
        is_val=True, 
        max_samples=max_samples_per_speaker
    )
    
    batch_processor_fn = batch_processor(feature_extractor)
    train_loader = DataLoader(
        train_data, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=2, 
        collate_fn=batch_processor_fn
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=2, 
        collate_fn=batch_processor_fn
    )
    
    num_speakers = len(optuna_train_speakers)
    model = SpeakerEncoder(num_speakers)
    
    # Apply LoRA configuration
    model = configure_lora(model, config["lora_rank"], config["lora_alpha"], config["lora_dropout"])
    model = model.to(COMPUTING_DEVICE)
    
    # Set ArcFace parameters
    model.arc_face.scale = config["scale_factor"]
    model.arc_face.margin = config["margin"]
    
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    
    trial_dir = os.path.join(RESULTS_DIR, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=trial_dir)
    
    num_optuna_epochs = 5  
    
    for epoch in range(num_optuna_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch, logger)
        
        val_acc, _, _, _ = evaluate_identification(model, val_loader)
        
        trial.report(val_acc, epoch)
        logger.add_scalar('trial/val_acc', val_acc, epoch)
        logger.add_scalar('trial/train_loss', train_loss, epoch)
        logger.add_scalar('trial/train_acc', train_acc, epoch)
        
        if trial.should_prune():
            logger.close()
            raise optuna.exceptions.TrialPruned()
    
    logger.close()
    return val_acc  

def main():
    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Set global variables for speaker IDs
    global train_speaker_ids, val_speaker_ids
    global train_speaker_to_id, val_speaker_to_id
    
    # Get speaker IDs and create mappings
    train_speaker_ids = []
    for speaker_dir in sorted(glob.glob(os.path.join(VOXCELEB2_DIR, "id*")))[:100]:  # Limit to 100 speakers
        speaker_id = os.path.basename(speaker_dir)
        train_speaker_ids.append(speaker_id)
    
    val_speaker_ids = []
    for speaker_dir in sorted(glob.glob(os.path.join(VOXCELEB2_DIR, "id*")))[100:120]:  # Use 20 different speakers for validation
        speaker_id = os.path.basename(speaker_dir)
        val_speaker_ids.append(speaker_id)
    
    train_speaker_to_id = {speaker: idx for idx, speaker in enumerate(train_speaker_ids)}
    val_speaker_to_id = {speaker: idx for idx, speaker in enumerate(val_speaker_ids)}
    
    print(f"Training with {len(train_speaker_ids)} speakers")
    print(f"Validation with {len(val_speaker_ids)} speakers")
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    # Run hyperparameter optimization with Optuna
    if os.path.exists(os.path.join(RESULTS_DIR, "best_params.json")):
        with open(os.path.join(RESULTS_DIR, "best_params.json"), "r") as f:
            best_params = json.load(f)
        print("Loaded best parameters from previous optimization")
    else:
        print("Starting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=10)  
        
        # Save best parameters
        best_params = study.best_params
        with open(os.path.join(RESULTS_DIR, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)
        
        # Print optimization results
        print("Hyperparameter optimization completed.")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.2f}%")
        print("Best hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    
    # Update config with best parameters
    for key, value in best_params.items():
        DEFAULT_CONFIG[key] = value
    
    # Create full datasets for training
    train_dataset = AudioDataset(
        VOXCELEB2_DIR, 
        train_speaker_ids, 
        feature_extractor
    )
    val_dataset = AudioDataset(
        VOXCELEB2_DIR, 
        val_speaker_ids, 
        feature_extractor, 
        is_val=True
    )
    
    # Create data loaders
    batch_processor_fn = batch_processor(feature_extractor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=DEFAULT_CONFIG["batch_size"], 
        shuffle=True,
        num_workers=4, 
        collate_fn=batch_processor_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=DEFAULT_CONFIG["batch_size"], 
        shuffle=False,
        num_workers=4, 
        collate_fn=batch_processor_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = SpeakerEncoder(len(train_speaker_ids))
    model = configure_lora(model, DEFAULT_CONFIG["lora_rank"], DEFAULT_CONFIG["lora_alpha"], DEFAULT_CONFIG["lora_dropout"])
    model = model.to(COMPUTING_DEVICE)
    
    # Set ArcFace parameters
    model.arc_face.scale = DEFAULT_CONFIG["scale_factor"]
    model.arc_face.margin = DEFAULT_CONFIG["margin"]
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(RESULTS_DIR, "final_training"))
    
    # First evaluate the pre-trained model on VoxCeleb1 for verification
    print("\nEvaluating pre-trained model on VoxCeleb1 trials...")
    pretrained_scores, pretrained_labels = process_verification_pairs(
        model, VOXCELEB1_TRIALS, VOXCELEB1_DIR, feature_extractor
    )
    
    # Compute pre-trained metrics
    pretrained_metrics = compute_verification_metrics(pretrained_scores, pretrained_labels)
    
    # Visualize pre-trained score distribution
    visualize_score_distribution(
        pretrained_labels,
        pretrained_scores,
        os.path.join(VISUALIZATION_DIR, "pretrained_score_dist.png"),
        title="Pre-trained Model Score Distribution"
    )
    
    # Evaluate identification accuracy
    pretrained_id_acc, pretrained_embeddings, pretrained_id_labels, _ = evaluate_identification(model, val_loader)
    
    print("\nPre-trained Model Performance:")
    print(f"Speaker Identification Accuracy: {pretrained_id_acc:.2f}%")
    for metric, value in pretrained_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize embeddings
    visualize_embeddings(
        pretrained_embeddings,
        pretrained_id_labels,
        os.path.join(VISUALIZATION_DIR, "pretrained_embeddings.png"),
        method='tsne'
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=DEFAULT_CONFIG["learning_rate"],
        weight_decay=0.01
    )
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DEFAULT_CONFIG["num_epochs"])
    
    # Train the model
    best_val_acc = 0.0
    print("\nStarting model training...")
    
    for epoch in range(DEFAULT_CONFIG["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{DEFAULT_CONFIG['num_epochs']}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch, writer)
        
        # Evaluate on validation set
        val_acc, val_embeddings, val_labels, conf_matrix = evaluate_identification(model, val_loader)
        
        # Log metrics
        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/train_acc', train_acc, epoch)
        writer.add_scalar('epoch/val_acc', val_acc, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': DEFAULT_CONFIG
            }, model_path)
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
    
    # Load the best model for final evaluation
    best_model_path = os.path.join(MODEL_DIR, "best_model.pt")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1} with validation accuracy {checkpoint['val_acc']:.2f}%")
    
    # Evaluate fine-tuned model on VoxCeleb1 trials
    print("\nEvaluating fine-tuned model on VoxCeleb1 trials...")
    finetuned_scores, finetuned_labels = process_verification_pairs(
        model, VOXCELEB1_TRIALS, VOXCELEB1_DIR, feature_extractor
    )
    
    # Compute fine-tuned metrics
    finetuned_metrics = compute_verification_metrics(finetuned_scores, finetuned_labels)
    
    # Visualize fine-tuned score distribution
    visualize_score_distribution(
        finetuned_labels,
        finetuned_scores,
        os.path.join(VISUALIZATION_DIR, "finetuned_score_dist.png"),
        title="Fine-tuned Model Score Distribution"
    )
    
    # Evaluate identification accuracy
    finetuned_id_acc, finetuned_embeddings, finetuned_id_labels, _ = evaluate_identification(model, val_loader)
    
    # Visualize ROC curve comparison
    visualize_roc_curve(
        pretrained_labels,
        pretrained_scores,
        finetuned_scores,
        os.path.join(VISUALIZATION_DIR, "roc_comparison.png")
    )
    
    # Visualize DET curve
    visualize_det_curve(
        pretrained_labels,
        pretrained_scores,
        finetuned_scores,
        os.path.join(VISUALIZATION_DIR, "det_comparison.png")
    )
    
    # Visualize fine-tuned embeddings
    visualize_embeddings(
        finetuned_embeddings,
        finetuned_id_labels,
        os.path.join(VISUALIZATION_DIR, "finetuned_embeddings.png"),
        method='tsne'
    )
    
    # Print performance comparison
    print("\n===== PERFORMANCE COMPARISON =====")
    print(f"{'Metric':<20} {'Pre-trained':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 65)
    
    # Compare identification accuracy
    print(f"{'ID Accuracy (%)':<20} {pretrained_id_acc:<15.2f} {finetuned_id_acc:<15.2f} {finetuned_id_acc - pretrained_id_acc:<+15.2f}")
    
    # Compare verification metrics
    for metric in pretrained_metrics.keys():
        pre_val = pretrained_metrics[metric]
        fine_val = finetuned_metrics[metric]
        
        # For EER, lower is better (improvement is negative)
        if metric == "EER (%)":
            improvement = pre_val - fine_val
        else:
            improvement = fine_val - pre_val
            
        print(f"{metric:<20} {pre_val:<15.4f} {fine_val:<15.4f} {improvement:<+15.4f}")
    
    # Save detailed results to a file
    result_file = os.path.join(RESULTS_DIR, "final_results.txt")
    with open(result_file, "w") as f:
        f.write("===== WavLM Speaker Verification System Results =====\n\n")
        f.write("Configuration:\n")
        for key, value in DEFAULT_CONFIG.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nPre-trained Model Performance:\n")
        f.write(f"Speaker Identification Accuracy: {pretrained_id_acc:.2f}%\n")
        for metric, value in pretrained_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nFine-tuned Model Performance:\n")
        f.write(f"Speaker Identification Accuracy: {finetuned_id_acc:.2f}%\n")
        for metric, value in finetuned_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
            
        f.write("\nImprovement:\n")
        f.write(f"ID Accuracy: {finetuned_id_acc - pretrained_id_acc:+.2f}%\n")
        for metric in pretrained_metrics.keys():
            pre_val = pretrained_metrics[metric]
            fine_val = finetuned_metrics[metric]
            
            if metric == "EER (%)":
                improvement = pre_val - fine_val
            else:
                improvement = fine_val - pre_val
                
            f.write(f"{metric}: {improvement:+.4f}\n")
    
    print(f"\nDetailed results saved to {result_file}")
    writer.close()
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()
