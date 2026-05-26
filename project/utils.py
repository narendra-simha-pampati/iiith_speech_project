import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

# Define emotions and speaker maps
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad'] # ps is pleasant surprise
EMOTION_TO_IDX = {emo: idx for idx, emo in enumerate(EMOTIONS)}

# Define carrier phrase vocabulary
# "say the word <word>"
# Total vocabulary will be: <pad>, <unk>, say, the, word, plus 200 target words.
vocab = ['<pad>', '<unk>', 'say', 'the', 'word']

def get_dataset_df(dataset_dir="/Users/narendra/Downloads/dataset_tess"):
    """
    Scans the TESS dataset folder, filters out duplicate nested folders,
    normalizes speakers and emotions, and returns a clean DataFrame.
    """
    wav_files = glob.glob(os.path.join(dataset_dir, "**/*.wav"), recursive=True)
    records = []
    
    for path in wav_files:
        # Deduplicate: TESS has a nested duplicate folder. We skip if "TESS Toronto emotional speech set data"
        # appears more than once in the relative path.
        subpath = path[len(dataset_dir):]
        parts_dir = subpath.split(os.sep)
        occurrences = sum(1 for p in parts_dir if p == "TESS Toronto emotional speech set data")
        if occurrences > 1:
            continue
            
        basename = os.path.basename(path)
        name_without_ext = basename[:-4]
        parts = name_without_ext.split("_")
        
        # Standard format is {speaker}_{word}_{emotion}
        if len(parts) < 3:
            continue
            
        speaker = parts[0]
        # Map speaker 'OA' typo to 'OAF'
        if speaker == 'OA':
            speaker = 'OAF'
            
        emotion = parts[-1].lower()
        # Map 'pleasant_surprised' folder emotion / 'ps' name to 'ps'
        if emotion in ['ps', 'pleasant_surprised', 'pleasant_surprise']:
            emotion = 'ps'
            
        word = "_".join(parts[1:-1])
        
        if emotion not in EMOTION_TO_IDX:
            continue
            
        records.append({
            "path": path,
            "speaker": speaker,
            "word": word.lower(),
            "emotion": emotion,
            "label": EMOTION_TO_IDX[emotion]
        })
        
    df = pd.DataFrame(records)
    # Deduplicate in case of absolute path duplicates
    df = df.drop_duplicates(subset=["speaker", "word", "emotion"])
    return df

def build_vocab(df):
    """
    Builds the vocabulary from all unique target words.
    """
    global vocab
    unique_words = sorted(list(df["word"].unique()))
    vocab = ['<pad>', '<unk>', 'say', 'the', 'word'] + unique_words
    word_to_idx = {w: idx for idx, w in enumerate(vocab)}
    idx_to_word = {idx: w for w, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

def tokenize_text(word, word_to_idx):
    """
    Tokenizes the phrase "say the word <word>" to indices.
    """
    phrase = ["say", "the", "word", word]
    indices = []
    for w in phrase:
        indices.append(word_to_idx.get(w, word_to_idx['<unk>']))
    return np.array(indices, dtype=np.int64)

def preprocess_audio_and_extract_mfcc(path, target_sr=16000, duration=2.0, n_mfcc=40):
    """
    Loads audio, trims silence, pads/truncates to duration, and extracts MFCCs.
    """
    try:
        y, sr = librosa.load(path, sr=target_sr)
        # Trim silence (top_db=20 is standard for active speech)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Pad or truncate to target duration
        target_length = int(target_sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
            
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Transpose so time steps is the first dimension: [time_steps, n_mfcc]
        return mfcc.T
    except Exception as e:
        print(f"Error processing {path}: {e}")
        # Return dummy features
        time_steps = int(target_sr * duration / 512) + 1
        return np.zeros((time_steps, n_mfcc), dtype=np.float32)

def get_word_disjoint_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Splits target words into train, validation, and test sets to ensure disjoint vocabularies.
    """
    unique_words = sorted(list(df["word"].unique()))
    np.random.seed(random_seed)
    np.random.shuffle(unique_words)
    
    n_words = len(unique_words)
    n_train = int(n_words * train_ratio)
    n_val = int(n_words * val_ratio)
    
    train_words = set(unique_words[:n_train])
    val_words = set(unique_words[n_train:n_train + n_val])
    test_words = set(unique_words[n_train + n_val:])
    
    train_df = df[df["word"].isin(train_words)].copy()
    val_df = df[df["word"].isin(val_words)].copy()
    test_df = df[df["word"].isin(test_words)].copy()
    
    return train_df, val_df, test_df

class MultimodalDataset(Dataset):
    def __init__(self, cached_samples):
        self.samples = cached_samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "speech": torch.tensor(sample["speech_feat"], dtype=torch.float32),
            "text": torch.tensor(sample["text_feat"], dtype=torch.long),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "word": sample["word"],
            "path": sample["path"]
        }

def cache_dataset_features(df, word_to_idx, target_sr=16000, duration=2.0, n_mfcc=40):
    """
    Pre-extracts and caches features for all records in df.
    """
    cached_samples = []
    total = len(df)
    print(f"Caching features for {total} samples...")
    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 500 == 0 or idx + 1 == total:
            print(f"  Processed {idx + 1}/{total}...")
            
        speech_feat = preprocess_audio_and_extract_mfcc(row["path"], target_sr, duration, n_mfcc)
        text_feat = tokenize_text(row["word"], word_to_idx)
        
        cached_samples.append({
            "speech_feat": speech_feat,
            "text_feat": text_feat,
            "label": row["label"],
            "word": row["word"],
            "path": row["path"]
        })
    return cached_samples
