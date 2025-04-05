import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import mir_eval.separation
from pesq import pesq
from speechbrain.inference import SepformerSeparation
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
BASE_DIR = "/home/m23csa001/Data/vox2_test_aac-001/aac"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
EMBEDDING_DIM = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

class EmbeddingExtractor(nn.Module):
    def __init__(self, hidden_size=1024, embed_dim=256):
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.projection_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  
            nn.PReLU(),
            nn.Linear(hidden_size, embed_dim)
        )
        
    def forward(self, features):
        attn_weights = self.attention_layer(features)
        pooled = torch.sum(attn_weights * features, dim=1)
        return self.projection_layer(pooled)

class SpeakerVerifier(nn.Module):
    def __init__(self, pretrained="microsoft/wavlm-base-plus"):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(pretrained)
        hidden_size = self.wavlm.config.hidden_size
        
        # Freeze feature extraction
        for param in self.wavlm.feature_extractor.parameters():
            param.requires_grad = False
            
        self.embedder = EmbeddingExtractor(hidden_size=hidden_size, embed_dim=EMBEDDING_DIM)
    
    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask)
        features = outputs.last_hidden_state
        embeddings = self.embedder(features)
        return F.normalize(embeddings, p=2, dim=1)


class SpeechSeparator:
    def __init__(self):
        self.model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr",
            savedir="pretrained_models/sepformer-whamr",
            run_opts={"device": DEVICE}
        )
        self.resampler_16k_to_8k = torchaudio.transforms.Resample(16000, 8000).to(DEVICE)
        self.resampler_8k_to_16k = torchaudio.transforms.Resample(8000, 16000).to(DEVICE)
        for param in self.model.parameters():
            param.requires_grad = True

    def separate(self, mixture, training=False):
        mixture_8k = self.resampler_16k_to_8k(mixture.to(DEVICE))
        
        if not training:
            with torch.no_grad():
                est_sources = self.model.separate_batch(mixture_8k.unsqueeze(0))
        else:
            est_sources = self.model.separate_batch(mixture_8k.unsqueeze(0))
        
        separated_sources = []
        for i in range(est_sources.shape[-1]):
            source_8k = est_sources[..., i].squeeze(0)
            source_16k = self.resampler_8k_to_16k(source_8k).cpu()
            separated_sources.append(source_16k)
        return separated_sources

class JointSpeakerSeparationIdentification(nn.Module):
    def __init__(self, pretrained_verifier=None):
        super().__init__()
        # Speaker verification model
        if pretrained_verifier:
            self.verifier = pretrained_verifier
        else:
            self.verifier = SpeakerVerifier()
            
        # Speech separator
        self.separator = SpeechSeparator()
        
        self.enhancement = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=8, stride=4, padding=2),
            nn.Tanh()
        )
        
        self._setup_trainable_parameters()
    
    def _setup_trainable_parameters(self):
        for param in self.verifier.wavlm.feature_extractor.parameters():
            param.requires_grad = False
            
        for name, param in self.separator.model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        for param in self.verifier.embedder.parameters():
            param.requires_grad = True
            
        for param in self.enhancement.parameters():
            param.requires_grad = True
        
    def forward(self, mixture, reference_embeddings=None, training=False):
        # Step 1: Separate the mixture using SepFormer
        separated_sources = self.separator.separate(mixture, training=training)
        
        # Step 2: Extract embeddings from separated sources
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        source_embeddings = []
        
        for source in separated_sources:
            inputs = feature_extractor(
                source.detach().numpy(),  
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                embedding = self.verifier(input_values=inputs['input_values'])
                source_embeddings.append(embedding)
                
        if reference_embeddings is not None:
            similarities = []
            assignments = []
            
            for src_emb in source_embeddings:
                scores = []
                for ref_emb in reference_embeddings:
                    score = F.cosine_similarity(src_emb, ref_emb)
                    scores.append(score)
                
                best_match = torch.argmax(torch.tensor(scores))
                assignments.append(best_match.item())
                similarities.append(scores[best_match.item()])
            
            return separated_sources, assignments, similarities
        
        return separated_sources

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def hook_fn(grad):
    if torch.isnan(grad).any():
        return torch.zeros_like(grad)  
    return grad

class AudioDataset(Dataset):
    def __init__(self, data_dir, split="train", sample_rate=16000, max_samples=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.mixtures = []
        
        speaker_dirs = sorted(glob.glob(os.path.join(data_dir, "id*")))
        
        if split == "train":
            speaker_dirs = speaker_dirs[:50]  # Use first 50 speakers for training
        else:
            speaker_dirs = speaker_dirs[50:100]  # Use next 50 speakers for testing
            
        print(f"Creating {split} dataset...")
        self.speakers = [os.path.basename(d) for d in speaker_dirs]
        
        # Create mixture data
        mixtures = self._create_mixtures(num_mixtures=max_samples if max_samples else 1000)
        self.mixtures = mixtures
        
    def _load_audio(self, file_path):
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            return waveform.squeeze(0)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
    def _create_mixtures(self, num_mixtures=100):
        mixtures = []
        valid_speakers = []
        
        for spk in self.speakers:
            spk_files = glob.glob(os.path.join(self.data_dir, spk, "**/*.m4a"), recursive=True)
            if len(spk_files) >= 2:
                valid_speakers.append(spk)
                
        print(f"Found {len(valid_speakers)} valid speakers with audio files")
        
        for _ in range(num_mixtures):
            if len(valid_speakers) < 2:
                continue
                
            spk1, spk2 = random.sample(valid_speakers, 2)
            spk1_files = glob.glob(os.path.join(self.data_dir, spk1, "**/*.m4a"), recursive=True)
            spk2_files = glob.glob(os.path.join(self.data_dir, spk2, "**/*.m4a"), recursive=True)
            
            file1 = random.choice(spk1_files)
            file2 = random.choice(spk2_files)
            
            sig1 = self._load_audio(file1)
            sig2 = self._load_audio(file2)
            
            if sig1 is None or sig2 is None:
                continue
                
            min_len = min(len(sig1), len(sig2))
            if min_len < 8000:  
                continue
                
            sig1 = sig1[:min_len] / (torch.max(torch.abs(sig1[:min_len])) + 1e-5)
            sig2 = sig2[:min_len] / (torch.max(torch.abs(sig2[:min_len])) + 1e-5)
            
            mixture = (sig1 + sig2) / 2
            
            mixtures.append({
                "mixture": mixture,
                "source1": sig1,
                "source2": sig2,
                "speaker1": spk1,
                "speaker2": spk2
            })
            
        print(f"Created {len(mixtures)} valid mixtures")
        return mixtures
    
    def __len__(self):
        return len(self.mixtures)
    
    def __getitem__(self, idx):
        item = self.mixtures[idx]
        return {
            "mixture": item["mixture"],
            "source1": item["source1"],
            "source2": item["source2"],
            "speaker1": item["speaker1"],
            "speaker2": item["speaker2"]
        }


def train_pipeline(model, train_loader, val_loader, num_epochs=5):
    separator_params = list(model.separator.model.parameters())
    verifier_params = list(model.verifier.parameters())
    enhancement_params = list(model.enhancement.parameters())
    
    parameter_groups = [
        {'params': separator_params, 'lr': LEARNING_RATE * 0.01},  
        {'params': verifier_params, 'lr': LEARNING_RATE * 0.001},  
        {'params': enhancement_params, 'lr': LEARNING_RATE * 0.1}       
    ]
    
    optimizer = torch.optim.AdamW(parameter_groups)
    mse_loss = nn.L1Loss()
    cos_loss = nn.CosineEmbeddingLoss()
    
    best_val_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            mixture = batch["mixture"].to(DEVICE)
            source1 = batch["source1"].to(DEVICE)
            source2 = batch["source2"].to(DEVICE)
            speaker1 = batch["speaker1"]
            speaker2 = batch["speaker2"]
            
            ref_embeddings = []
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
            
            for src in [source1[0], source2[0]]:
                inputs = feature_extractor(
                    src.cpu().numpy(),
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True
                ).to(DEVICE)
                
                with torch.set_grad_enabled(False):  
                    embed = model.verifier(input_values=inputs['input_values'])
                    ref_embeddings.append(embed)
            
            separated_sources = model.separator.separate(mixture[0], training=True)
            for i, src in enumerate(separated_sources):
                if check_nan(src, f"separated_source_{i}"):
                    continue
            
            if len(separated_sources) < 2:
                continue
                
            sep1 = separated_sources[0].to(DEVICE)
            sep2 = separated_sources[1].to(DEVICE)
            
            sep_embeddings = []
            for src in [sep1, sep2]:
                inputs = feature_extractor(
                    src.cpu().detach().numpy(),  
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True
                ).to(DEVICE)
                
                embed = model.verifier(input_values=inputs['input_values'])
                sep_embeddings.append(embed)
            
            min_len = min(len(sep1), len(source1[0]))
            source1_trunc = source1[0][:min_len]
            source2_trunc = source2[0][:min_len]
            sep1_trunc = sep1[:min_len]
            sep2_trunc = sep2[:min_len]
            
            loss_direct = mse_loss(sep1_trunc, source1_trunc) + mse_loss(sep2_trunc, source2_trunc)
            loss_perm = mse_loss(sep1_trunc, source2_trunc) + mse_loss(sep2_trunc, source1_trunc)
            
            if loss_direct < loss_perm:
                perm = [0, 1]  
                recon_loss = loss_direct
            else:
                perm = [1, 0]  
                recon_loss = loss_perm
            
            # Calculate embedding similarity loss
            emb_loss = 0
            for i, p in enumerate(perm):
                target = torch.ones(1).to(DEVICE)
                emb_loss += cos_loss(sep_embeddings[i], ref_embeddings[p], target)
            
            # Calculate enhancement loss using spectral features
            enh_loss = 0
            for i, p in enumerate(perm):
                src = source1_trunc if p == 0 else source2_trunc
                sep = sep1_trunc if i == 0 else sep2_trunc
                
                src_spec = torch.stft(src, n_fft=512, hop_length=256, return_complex=False)
                sep_spec = torch.stft(sep, n_fft=512, hop_length=256, return_complex=False)
                
                src_mag = torch.sqrt(src_spec[..., 0]**2 + src_spec[..., 1]**2).unsqueeze(0)
                sep_mag = torch.sqrt(sep_spec[..., 0]**2 + sep_spec[..., 1]**2).unsqueeze(0) 
                enh_loss += F.mse_loss(sep_mag, src_mag)
            
            if epoch < 2:
                total_loss = recon_loss  
            else:
                total_loss = recon_loss + min(epoch/5, 0.5) * emb_loss + min(epoch/5, 0.3) * enh_loss
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                mixture = batch["mixture"].to(DEVICE)
                source1 = batch["source1"].to(DEVICE)
                source2 = batch["source2"].to(DEVICE)
                speaker1 = batch["speaker1"]
                speaker2 = batch["speaker2"]
                
                ref_embeddings = []
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
                for src in [source1[0], source2[0]]:
                    inputs = feature_extractor(
                        src.cpu().detach().numpy(),  
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                        padding=True
                    ).to(DEVICE)
                    embed = model.verifier(input_values=inputs['input_values'])
                    ref_embeddings.append(embed)
                
                separated_sources = model.separator.separate(mixture[0], training=False)
                
                if len(separated_sources) < 2:
                    continue
                
                sep1 = separated_sources[0].to(DEVICE)
                sep2 = separated_sources[1].to(DEVICE)
                
                sep_embeddings = []
                for src in [sep1, sep2]:
                    inputs = feature_extractor(
                        src.cpu().detach().numpy(),  
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                        padding=True
                    ).to(DEVICE)
                    embed = model.verifier(input_values=inputs['input_values'])
                    sep_embeddings.append(embed)
                
                min_len = min(len(sep1), len(source1[0]))
                source1_trunc = source1[0][:min_len]
                source2_trunc = source2[0][:min_len]
                sep1_trunc = sep1[:min_len]
                sep2_trunc = sep2[:min_len]
                
                # Calculate direct and permuted reconstruction losses
                loss_direct = mse_loss(sep1_trunc, source1_trunc) + mse_loss(sep2_trunc, source2_trunc)
                loss_perm = mse_loss(sep1_trunc, source2_trunc) + mse_loss(sep2_trunc, source1_trunc)
                
                if loss_direct < loss_perm:
                    perm = [0, 1]
                    recon_loss = loss_direct
                else:
                    perm = [1, 0]
                    recon_loss = loss_perm
                
                # Calculate embedding similarity loss
                emb_loss = 0
                for i, p in enumerate(perm):
                    target = torch.ones(1).to(DEVICE)
                    emb_loss += cos_loss(sep_embeddings[i], ref_embeddings[p], target)
                
                # Calculate enhancement loss using spectral features
                enh_loss = 0
                for i, p in enumerate(perm):
                    src = source1_trunc if p == 0 else source2_trunc
                    sep = sep1_trunc if i == 0 else sep2_trunc
                    
                    src_spec = torch.stft(src, n_fft=512, hop_length=256, return_complex=False)
                    sep_spec = torch.stft(sep, n_fft=512, hop_length=256, return_complex=False)
                    
                    src_mag = torch.sqrt(src_spec[..., 0]**2 + src_spec[..., 1]**2).unsqueeze(0)
                    sep_mag = torch.sqrt(sep_spec[..., 0]**2 + sep_spec[..., 1]**2).unsqueeze(0) 
                    enh_loss += F.mse_loss(sep_mag, src_mag)
                
                if epoch < 2:
                    total_loss = recon_loss  # Only reconstruction loss initially
                else:
                    total_loss = recon_loss + min(epoch/5, 0.5) * emb_loss + min(epoch/5, 0.3) * enh_loss
                val_loss += total_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_joint_model.pth"))
            print(f"Model saved at epoch {epoch+1}")


def build_speaker_embeddings(model, speakers, data_dir):
    model.eval()
    feature_extractor =  Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    embeddings = {}
    
    for spk in tqdm(speakers, desc="Building speaker embeddings"):
        spk_files = glob.glob(os.path.join(data_dir, spk, "**/*.m4a"), recursive=True)[:3]
        
        if not spk_files:
            continue
            
        spk_embeddings = []
        
        for file in spk_files:
            waveform, sr = torchaudio.load(file)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)
                
            inputs = feature_extractor(
                waveform.numpy(),
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            
            with torch.no_grad():
                embedding = model(input_values=inputs['input_values']).cpu()
                spk_embeddings.append(embedding)
                
        if spk_embeddings:
            embeddings[spk] = torch.mean(torch.stack(spk_embeddings), dim=0)
            
    return embeddings


def evaluate_separation(reference_sources, estimated_sources):
    if torch.is_tensor(reference_sources[0]):
        reference_sources = [src.cpu().numpy() for src in reference_sources]
    if torch.is_tensor(estimated_sources[0]):
        estimated_sources = [src.cpu().numpy() for src in estimated_sources]
        
    min_len = min([len(src) for src in reference_sources + estimated_sources])
    reference_sources = [src[:min_len] for src in reference_sources]
    estimated_sources = [src[:min_len] for src in estimated_sources]
    
    reference_sources = np.stack(reference_sources)
    estimated_sources = np.stack(estimated_sources)
    
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources)
    
    pesq_scores = []
    try:
        for i in range(reference_sources.shape[0]):
            ref = reference_sources[i]
            est = estimated_sources[i]
            
            ref = ref / (np.abs(ref).max() + 1e-8)
            est = est / (np.abs(est).max() + 1e-8)
            
            score = pesq(SAMPLE_RATE, ref, est, 'wb')
            pesq_scores.append(score)
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        pesq_scores = [0.0] * reference_sources.shape[0]
        
    return {
        "SDR": float(np.mean(sdr)),
        "SIR": float(np.mean(sir)),
        "SAR": float(np.mean(sar)),
        "PESQ": float(np.mean(pesq_scores))
    }


def evaluate_identification(model, feature_extractor, embeddings, separated_sources, true_speakers):
    
    model.eval()
    predictions = []
    
    for source in separated_sources:
       
        if torch.is_tensor(source):
            source = source.cpu().numpy()
            
        
        inputs = feature_extractor(
            source,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        
        with torch.no_grad():
            query_embed = model(input_values=inputs['input_values']).cpu()
            
        
        best_score = -1
        best_spk = None
        
        for spk_id, ref_embed in embeddings.items():
            score = torch.cosine_similarity(query_embed, ref_embed, dim=1)
            score = score.item()
            
            if score > best_score:
                best_score = score
                best_spk = spk_id
                
        predictions.append(best_spk)
        
    correct = sum(1 for p in predictions if p in true_speakers)
    accuracy = correct / len(separated_sources) if separated_sources else 0
    
    return {
        "accuracy": accuracy,
        "predictions": predictions
    }


def main():
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize datasets
    print("Initializing datasets...")
    train_dataset = AudioDataset(BASE_DIR, split="train", max_samples=200)
    test_dataset = AudioDataset(BASE_DIR, split="test", max_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize models
    print("Creating models...")
    pretrained_verifier = SpeakerVerifier().to(DEVICE)
    
    # Apply LoRA adaptation for the finetuned model
    finetuned_verifier = SpeakerVerifier().to(DEVICE)
    
    # Configure LoRA for verifier
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["k_proj", "v_proj", "q_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    # Apply LoRA to the verifier model
    finetuned_verifier.wavlm.encoder = PeftModel(finetuned_verifier.wavlm.encoder, lora_config)
    
    
    joint_model = JointSpeakerSeparationIdentification(pretrained_verifier=pretrained_verifier).to(DEVICE)
    
    # Training parameters
    # optimizer = torch.optim.AdamW([
    #     {'params': joint_model.separator.parameters(), 'lr': LEARNING_RATE * 0.1},
    #     {'params': joint_model.verifier.parameters(), 'lr': LEARNING_RATE * 0.01},
    #     {'params': joint_model.enhancement.parameters(), 'lr': LEARNING_RATE}
    # ])
    # ---------------------------------------------------
    
    # Train the joint model
    # print("Training joint model...")
    # train_pipeline(joint_model, train_loader, test_loader, num_epochs=EPOCHS)
    
    # Load best model
    best_model_path = os.path.join(OUTPUT_DIR, "best_joint_model.pth")
    if os.path.exists(best_model_path):
        joint_model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model weights")
    
    # Build speaker embeddings
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    print("Building speaker embeddings...")
    pretrained_embeddings = build_speaker_embeddings(pretrained_verifier, test_dataset.speakers, BASE_DIR)
    finetuned_embeddings = build_speaker_embeddings(finetuned_verifier, test_dataset.speakers, BASE_DIR)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    # Results storage
    separation_results = []
    pretrained_acc = []
    finetuned_acc = []
    
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        mixture = batch["mixture"][0]
        source1 = batch["source1"][0]
        source2 = batch["source2"][0]
        speaker1 = batch["speaker1"][0]
        speaker2 = batch["speaker2"][0]
        
        # Separate using joint model
        separated_sources = joint_model.separator.separate(mixture)
        
        if len(separated_sources) < 2:
            continue
            
        # Evaluate separation quality
        ref_sources = [source1.numpy(), source2.numpy()]
        est_sources = [src.numpy() for src in separated_sources[:2]]
        
        metrics = evaluate_separation(ref_sources, est_sources)
        separation_results.append(metrics)
        
        # Evaluate speaker identification with pretrained model
        pretrained_results = evaluate_identification(
            pretrained_verifier,
            feature_extractor,
            pretrained_embeddings,
            separated_sources,
            [speaker1, speaker2]
        )
        pretrained_acc.append(pretrained_results["accuracy"])
        
        # Evaluate speaker identification with finetuned model
        finetuned_results = evaluate_identification(
            finetuned_verifier,
            feature_extractor,
            finetuned_embeddings,
            separated_sources,
            [speaker1, speaker2]
        )
        finetuned_acc.append(finetuned_results["accuracy"])
    
    # Compute average metrics
    avg_separation = {
        metric: np.mean([result[metric] for result in separation_results])
        for metric in ["SDR", "SIR", "SAR", "PESQ"]
    }
    
    avg_pretrained_acc = np.mean(pretrained_acc)
    avg_finetuned_acc = np.mean(finetuned_acc)
    
    # Print and save results
    print("\n===== RESULTS =====")
    print("\nSeparation Performance:")
    print(f"SDR: {avg_separation['SDR']:.2f} dB")
    print(f"SIR: {avg_separation['SIR']:.2f} dB")
    print(f"SAR: {avg_separation['SAR']:.2f} dB")
    print(f"PESQ: {avg_separation['PESQ']:.2f}")
    
    print("\nSpeaker Identification:")
    print(f"Pretrained model accuracy: {avg_pretrained_acc:.2f}")
    print(f"Finetuned model accuracy: {avg_finetuned_acc:.2f}")
    print(f"Improvement: {(avg_finetuned_acc - avg_pretrained_acc):.2f}")
    
    # Save results to file
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write("===== RESULTS =====\n\n")
        f.write("Separation Performance:\n")
        f.write(f"SDR: {avg_separation['SDR']:.2f} dB\n")
        f.write(f"SIR: {avg_separation['SIR']:.2f} dB\n")
        f.write(f"SAR: {avg_separation['SAR']:.2f} dB\n")
        f.write(f"PESQ: {avg_separation['PESQ']:.2f}\n\n")
        
        f.write("Speaker Identification:\n")
        f.write(f"Pretrained model accuracy: {avg_pretrained_acc:.2f}\n")
        f.write(f"Finetuned model accuracy: {avg_finetuned_acc:.2f}\n")
        f.write(f"Improvement: {(avg_finetuned_acc - avg_pretrained_acc):.2f}\n")


if __name__ == "__main__":
    main()
