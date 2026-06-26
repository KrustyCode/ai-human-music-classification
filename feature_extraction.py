import numpy as np
import pandas as pd
import librosa as lb
from sklearn.model_selection import train_test_split
from pathlib import Path
from scipy import fftpack
from collections.abc import Iterator

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
N_FFT           = 1024
HOP_LENGTH      = 256
WINDOW_TYPE     = "hann"

# ---- MFCC ----
def find_data(folder: Path) -> Iterator[Path]:
    for data in folder.rglob("*"):
        if data.is_file() and data.suffix.lower() == FILE_FORMAT:
            yield data

def balancing_dataset(df: pd.DataFrame) -> pd.DataFrame:
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

def build_index(data_dir: dict) -> pd.DataFrame:
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


def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file, only return the signal array"""
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

def compute_stft(audio: np.ndarray) -> np.ndarray:
    """"Compute a STFT"""
    stft = lb.stft(
        y = audio,
        n_fft=512,
        hop_length=HOP_LENGTH
    )
    return lb.power_to_db(np.abs(stft)**2, ref=np.max).astype(np.float32)

def hz_to_bark(hz):
    b = 26.81 * hz / (1960 + hz) - 0.53
    return np.where(b < 2, b + 0.15 * (2 - b), b)

def bark_filterbank(sr, n_fft, n_bands=24, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    freqs = lb.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs_bark = hz_to_bark(freqs)
    bark_edges = np.linspace(hz_to_bark(fmin), hz_to_bark(fmax), n_bands + 1)
    bark_edges[-1] += 1e-6  # avoid dropping the Nyquist bin

    fb = np.zeros((n_bands, len(freqs)))
    for i in range(n_bands):
        mask = (freqs_bark >= bark_edges[i]) & (freqs_bark < bark_edges[i + 1])
        fb[i, mask] = 1.0
    return fb, freqs

def compute_bark(audio: np.ndarray) -> np.ndarray:
    """"Compute a Bark Spectrogram"""

    # Step 1: Get STFT power spectrogram
    D = np.abs(lb.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2

    # 2. Bark filterbank (rectangular, as you built it)
    fb, freqs = bark_filterbank(SAMPLE_RATE, N_FFT)

    # 3. Fold linear-frequency power into Bark bands
    bark_spec = fb @ D   # shape: (n_bands, n_frames)

    # 4. Convert to dB, consistent with your other spectrogram features
    bark_db = lb.power_to_db(bark_spec, ref=np.max)

    return bark_db.astype(np.float32)

def equal_loudness_weight(freqs_hz):
    w = 2 * np.pi * freqs_hz
    num = (w**2 + 56.8e6) * w**4
    den = (w**2 + 6.3e6)**2 * (w**2 + 0.38e6) * (w**6 + 9.58e26)
    return num / den

# ---------- Step 3: Levinson-Durbin (autocorrelation -> LPC) ----------

def levinson_durbin(r, order):
    # r: autocorrelation sequence, r[0] = energy
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i-1:0:-1])
        k = -acc / e
        a_new = a.copy()
        a_new[1:i] = a[1:i] + k * a[i-1:0:-1]
        a_new[i] = k
        a = a_new
        e *= (1 - k**2)
        if e <= 0:
            e = 1e-10
    return a, e  # a[0]=1, a[1:] are LPC coeffs; e = prediction error (gain)

# ---------- Step 4: LPC -> cepstral coefficients (standard recursion) ----------

def lpc_to_cepstrum(a, gain, n_cep):
    order = len(a) - 1
    c = np.zeros(n_cep)
    c[0] = np.log(max(gain, 1e-10))
    for m in range(1, n_cep):
        acc = 0.0
        for k in range(1, m):
            if k <= order:
                acc += k * c[k] * (-a[m - k]) if (m - k) <= order else 0
        if m <= order:
            c[m] = -a[m] + acc / m
        else:
            c[m] = acc / m
    return c

# ---------- Full PLP extraction ----------

def compute_plp(audio, sr: int =SAMPLE_RATE, n_fft: int =N_FFT, hop_length: int =HOP_LENGTH, n_bands: int=24,
                 lpc_order: int =12, n_cep: int =13) -> np.ndarray:

    # 1. Power spectrum
    D = np.abs(lb.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2

    # 2. Bark-warp the power spectrum
    fb, freqs = bark_filterbank(sr, n_fft, n_bands=n_bands)
    bark_spec = fb @ D  # (n_bands, n_frames)

    # 3. Equal-loudness weighting (per Bark band, using band center freq)
    band_centers = np.array([
        np.mean(freqs[fb[i] > 0]) if np.any(fb[i] > 0) else 0
        for i in range(n_bands)
    ])
    eql = equal_loudness_weight(band_centers + 1e-6)
    eql = eql / np.max(eql)  # normalize to avoid scale blowup
    bark_spec = bark_spec * eql[:, np.newaxis]

    # 4. Intensity-loudness power law (cube-root compression)
    bark_spec = np.power(np.maximum(bark_spec, 1e-10), 1.0 / 3.0)

    n_frames = bark_spec.shape[1]
    plp_feats = np.zeros((n_cep, n_frames))

    for t in range(n_frames):
        spec = bark_spec[:, t]
        # mirror spectrum to make it symmetric/even before inverse FFT
        full_spec = np.concatenate([spec, spec[-2:0:-1]])
        # 5. Inverse FFT -> pseudo-autocorrelation
        autocorr = np.fft.ifft(full_spec).real
        autocorr = autocorr[:lpc_order + 1]
        autocorr[0] += 1e-6  # stabilize

        # 6. Levinson-Durbin -> LPC coefficients
        a, gain = levinson_durbin(autocorr, lpc_order)

        # 7. LPC -> cepstral coefficients
        plp_feats[:, t] = lpc_to_cepstrum(a, gain, n_cep)

    return plp_feats.astype(np.float32)

def compute_mfcc(audio: np.ndarray, n: int =13) -> np.ndarray:
    """Compute a mfcc"""
    mfcc = lb.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=n,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return mfcc.astype(np.float32)

def linear_filterbank(
        sr: int = SAMPLE_RATE, n_fft: int = N_FFT, 
        n_filters:int = 128):
    

    freqs = np.linspace(
        0,
        sr/2,
        n_fft//2 + 1
    )

    edges = np.linspace(
        0,
        sr/2,
        n_filters + 2
    )

    fb = np.zeros((n_filters, len(freqs)))

    for i in range(n_filters):

        left   = edges[i]
        center = edges[i+1]
        right  = edges[i+2]

        fb[i] = np.maximum(
            0,
            np.minimum(
                (freqs-left)/(center-left),
                (right-freqs)/(right-center)
            )
        )
    return fb

def compute_lfcc(audio: np.ndarray, n: int =13) -> np.ndarray:
    """Compute a lfcc"""
    filter_banks = linear_filterbank()
    stft = np.abs(lb.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    filtered_stft = lb.amplitude_to_db(np.dot(filter_banks, stft), ref=np.max)
    lfcc = fftpack.dct(filtered_stft, axis=0, type=2, norm='ortho')[:n]
    return lfcc.astype(np.float32)
    

def normalization(audio: np.ndarray) -> np.ndarray:
    """Normalize audio signal amplitude"""
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
            case "chroma":
                spec = compute_chromagram(chunk)
            case "bark":
                spec = compute_bark(chunk)
            case "plp":
                spec = compute_plp(chunk)
            case "plp_20":
                spec = compute_plp(chunk, n_cep=20)
            case "plp_40":
                spec = compute_plp(chunk, n_cep=40)
            case "stft":
                spec = compute_stft(chunk)
            case "lfcc":
                spec = compute_lfcc(chunk)
            case "lfcc_20":
                spec = compute_lfcc(chunk, 20)
            case "lfcc_40":
                spec = compute_lfcc(chunk, 40)
            case _:
                raise ValueError(    
                    f"Unknown feature extraction type: '{feature}'. "
                )
        spec = standardize(spec)
        out.append(spec)
    return out

def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED) -> dict:
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


def load_data(x_path: Path, y_path: Path, s_path: Path) -> list:
    x = np.load(x_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    s = np.load(s_path, mmap_mode='r')
    return x, y, s


def get_feature(feature: str) -> dict:
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