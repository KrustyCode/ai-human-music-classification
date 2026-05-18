import numpy as np
import pandas as pd
import librosa as lb
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# ---- Configuration ----
DATASET_DIR     = "dataset"
FILE_FORMAT     = ".wav"
SAMPLE_RATE     = 16_000 #Hz
CLIP_DURATION   = 5 #sec
REAL_MUSIC      = "music_caps"
AI_MUSIC        = [
    "music_ldm",
    "audio_ldm2",
    "mustango",
    "music_gen_medium",
    "stable_audio_open"
]

RANDOM_SEED     = 42
np.random.seed(RANDOM_SEED)

TEST_SIZE       = 0.15
VALIDATION_SIZE = 0.15
TRAIN_SIZE      = 0.7

# ---- Mel Spectrogram ----
N_MELS          = 128
N_FFT           = 2048
HOP_LENGTH      = 512
WINDOW_TYPE     = "hann"

# ---- MFCC ----



def find_data(folder: Path):
    for data in folder.rglob("*"):
        if data.is_file() and data.suffix.lower() == FILE_FORMAT:
            yield data

def balancing_dataset(df: pd.DataFrame):
    real_df = df[df.label == 0]
    fake_df = df[df.label == 1]
    n_real = len(real_df)

    fake_sources = sorted(fake_df["source"].unique())
    per_source = max(1, n_real // len(fake_sources))

    sampled = []
    for src in fake_sources:
        chunk = fake_df[fake_df["source"] == src]
        take = min(len(chunk), per_source)
        sampled.append(chunk.sample(n=take, random_state=RANDOM_SEED))
    
    fake_balanced = pd.concat(sampled, ignore_index=True)

    # final trim if rounding pushed us over
    if len(fake_balanced) > n_real:
        fake_balanced = fake_balanced.sample(n=n_real, random_state=RANDOM_SEED)

    balanced_df = pd.concat([real_df, fake_balanced], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"After balancing: {len(balanced_df):,} clips")
    print(balanced_df["label"].value_counts().rename({0: "Human", 1: "AI"}))
    print("\nAI-side breakdown:")
    print(balanced_df[balanced_df.label == 1]["source"].value_counts())

    return balanced_df

def build_index(data_dir: dict):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"DATA_DIR '{data_dir}' doesn't exist."
            f"Upload FakeMusicCaps Dataset to Work Directory and update DATA_DIR"
        )
    
    rows =[]

    for dataset in AI_MUSIC:
        dataset_path = root/dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(
                f"Dataset '{dataset}' doesn't exist."
            )
        
        for file in find_data(dataset_path):
            rows.append({
                "filepath": str(file),
                "label": 1,
                "source": dataset
            })
    
    human_music_path = root/REAL_MUSIC
    if not human_music_path.exists() or not human_music_path.is_dir():
        raise FileNotFoundError(
            f"Dataset '{REAL_MUSIC}' doesn't exist."
        )
    
    for file in find_data(human_music_path):
        rows.append({
            "filepath": str(file),
            "label": 0,
            "source": REAL_MUSIC
        })
    

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No audio files found. Please check DATASET_DIR '{data_dir}'.")
    
    print(f"Found {len(df)} audio files.")
    print(f"\nCounts by label:\n {df['label'].value_counts()}")
    print(f"\nCounts by source:\n {df['source'].value_counts()}")

    df = balancing_dataset(df)

    return df


def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE):
    y, sr = lb.load(
        file_path,
        sr=sample_rate,
        mono=True,
    )
    return y.astype(np.float32)


def chunk_into_windows(y: np.ndarray,
                       window_samples: int = (CLIP_DURATION*SAMPLE_RATE) ) -> list:
    """Split a 1-D audio array into fixed-length windows.

    Clips shorter than one window are zero-padded.
    Tails of length > 25% of a window are kept and zero-padded.
    """
    n = len(y)
    if n <= window_samples:
        out = np.zeros(window_samples, dtype=np.float32)
        out[:n] = y
        return [out]

    chunks = []
    n_full = n // window_samples
    for i in range(n_full):
        chunks.append(y[i * window_samples : (i + 1) * window_samples])
    rem = n - n_full * window_samples
    if rem > window_samples // 4:
        tail = np.zeros(window_samples, dtype=np.float32)
        tail[:rem] = y[n_full * window_samples:]
        chunks.append(tail)
    
    return chunks

def standardize(spec: np.ndarray) -> np.ndarray:
    """Per-sample zero-mean, unit-variance normalization."""
    return (spec - spec.mean()) / (spec.std() + 1e-6)

def compute_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute a log-mel-spectrogram in dB scale."""
    mel = lb.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW_TYPE,
    )
    return lb.power_to_db(mel, ref=np.max).astype(np.float32)

def compute_chromagram(audio: np.ndarray) -> np.ndarray:
    """Compute a chromagram"""
    chroma = lb.feature.chroma_stft(
        y= audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return chroma.astype(np.float32)

def compute_mfcc(audio: np.ndarray, n=13) -> np.ndarray:
    """Compute a mfcc"""
    mfcc = lb.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=n,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return mfcc.astype(np.float32)

def normalization(audio: np.ndarray):

    max_val = np.max(np.abs(audio))

    if max_val > 0:
        audio = audio / max_val

    return audio.astype(np.float32)

def preprocess_clip(filepath: str, feature: str) -> list:
    audio = load_audio(filepath)
    audio = normalization(audio)
    out = []
    for chunk in chunk_into_windows(audio):
        match feature:
            case "mel":
                spec = compute_mel_spectrogram(chunk)
            case "mfcc":
                spec = compute_mfcc(chunk)
            case "mfcc_20":
                spec = compute_mfcc(chunk, 20)
            case "mfcc_40":
                spec = compute_mfcc(chunk, 40)
            case _:
                raise ValueError(    
                    f"Unknown feature extraction type: '{feature}'. "
                    f"Available options are: 'mel', 'mfcc', 'chroma'."
                )
        spec = standardize(spec)
        out.append(spec)
    return out

def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED):
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df["label"],
        random_state=seed
    )

    val_ratio = VALIDATION_SIZE / (
        VALIDATION_SIZE + TEST_SIZE
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df["label"],
        random_state=seed
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True)}


def load_data(x_path: Path, y_path: Path, s_path: Path):
    x = np.load(x_path)
    y = np.load(y_path)
    s = np.load(s_path)
    
    return x, y, s


def get_feature(feature: str):
    cache_dir = Path(f"{feature}_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_partitions = ["train", "val", "test"]

    dataset_paths = {
        part: {
            "x_path": cache_dir / f"{part}_x.npy",
            "y_path": cache_dir / f"{part}_y.npy",
            "s_path": cache_dir / f"{part}_s.npy",
        }
        for part in dataset_partitions
    }

    out = {}

    cache_complete = True

    for part, paths in dataset_paths.items():
        x_path = paths["x_path"]
        y_path = paths["y_path"]
        s_path = paths["s_path"]

        if x_path.exists() and y_path.exists() and s_path.exists():
            out[part] = load_data(
                x_path,
                y_path,
                s_path
            )
        else:
            cache_complete = False
            break

    if cache_complete:
        return out
    
    df  = build_index(DATASET_DIR)
    split_dfs = split_data(df)

    for part, partition_df in split_dfs.items():
        x = []
        y = []
        s = []

        n_files = len(partition_df)

        for idx, row in enumerate(partition_df.itertuples()):
            filepath = row.filepath
            label    = row.label
            source   = row.source

            try:
                specs = preprocess_clip(filepath, feature)   # list of 2-D arrays

                # ── FIX: extend, not append ────────────────────────────────
                # Each file may produce multiple chunks; flatten them all
                # into x so every element is one (H, W) spectrogram.
                x.extend(specs)                  # was: x.append(spec)
                y.extend([label]  * len(specs))  # was: y.append(label)
                s.extend([source] * len(specs))  # was: s.append(source)
                # ──────────────────────────────────────────────────────────

                print(
                    f"[{part}] [{idx+1}/{n_files}] "
                    f"Processed: {filepath}  ({len(specs)} chunk(s))"
                )

            except Exception as e:
                print(f"[{part}] Error processing {filepath}: {e}")

        x = np.stack(x, axis=0).astype(np.float32)  # (N, H, W)
        y = np.array(y)
        s = np.array(s)

        x = x[..., np.newaxis]   # (N, H, W, 1)

        np.save(dataset_paths[part]["x_path"], x)
        np.save(dataset_paths[part]["y_path"], y)
        np.save(dataset_paths[part]["s_path"], s)

        out[part] = (x, y, s)

    print(f"{feature} has been extracted")

    return out