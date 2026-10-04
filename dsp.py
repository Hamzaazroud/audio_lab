"""Audio Lab - fonctions DSP : filtres, séparation voix/instru, reverb, pitch, download.

Tout est en numpy/scipy (rapide au démarrage). librosa n'est importé qu'en secours
(décodage m4a/aac) et pour l'estimation de f0 (pYIN).
"""
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from math import gcd

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100


# ---------- I/O ----------
def _prep(data, fs, sr):
    """data (canaux, n) -> stéréo float32 à la fréquence sr."""
    y = np.repeat(data, 2, axis=0) if data.shape[0] == 1 else data[:2]
    if fs != sr:
        g = gcd(int(sr), int(fs))
        y = signal.resample_poly(y, sr // g, fs // g, axis=1)
    return np.ascontiguousarray(y, dtype=np.float32)


def load_audio(path, sr=SR):
    """Charge un fichier entier -> (y[2, n] float32, sr).
    soundfile d'abord (wav/flac/ogg/mp3, rapide), librosa/audioread en secours (m4a...)."""
    try:
        data, fs = sf.read(path, dtype="float32", always_2d=True)
        return _prep(data.T, fs, sr), sr
    except Exception:  # noqa: BLE001
        import librosa

        y, _ = librosa.load(path, sr=sr, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        return np.ascontiguousarray(y, dtype=np.float32), sr


_FULL = {"key": None, "y": None}


def _full(path, sr=SR):
    """Décodage complet mis en cache (pour les formats qu'on ne peut pas lire par morceaux)."""
    key = (path, os.path.getmtime(path), sr)
    if _FULL["key"] != key:
        _FULL["y"], _ = load_audio(path, sr)
        _FULL["key"] = key
    return _FULL["y"]


def audio_info(path):
    """(durée en s, fréquence d'échantillonnage) sans décoder tout le fichier."""
    try:
        i = sf.info(path)
        if i.frames <= 0:
            raise ValueError("durée inconnue")
        return i.frames / i.samplerate, i.samplerate
    except Exception:  # noqa: BLE001  (m4a...) -> décodage complet, une seule fois
        return _full(path).shape[1] / SR, SR


def load_excerpt(path, start, dur, sr=SR):
    """Charge seulement [start, start+dur] secondes -> (y[2, n] float32, sr).
    Lecture partielle (seek) = beaucoup plus rapide qu'un décodage complet."""
    try:
        with sf.SoundFile(path) as f:
            fs = f.samplerate
            f.seek(min(int(start * fs), max(f.frames - 1, 0)))
            data = f.read(int(dur * fs), dtype="float32", always_2d=True)
        return _prep(data.T, fs, sr), sr
    except Exception:  # noqa: BLE001
        y = _full(path, sr)
        a = int(start * sr)
        return np.ascontiguousarray(y[:, a:a + int(dur * sr)]), sr


def save_wav(path, y, sr):
    sf.write(path, y.T, sr)


def normalize(y, peak=0.95):
    m = np.max(np.abs(y))
    return y if m == 0 else y * (peak / m)


def download(url, out_dir="downloads"):
    """Télécharge l'audio d'une URL en WAV (yt-dlp + ffmpeg requis).
    À utiliser pour du contenu dont tu as le droit (perso, libre de droits...)."""
    import yt_dlp

    os.makedirs(out_dir, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_dir}/%(title)s.%(ext)s",
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    return os.path.splitext(path)[0] + ".wav"


# ---------- Filtres ----------
def butter_filter(y, sr, kind, cutoff, order=4):
    """kind: 'highpass' | 'lowpass' | 'bandpass' (cutoff = (f1, f2) pour bandpass).
    Butterworth en SOS + sosfiltfilt (phase nulle)."""
    nyq = sr / 2
    wn = [c / nyq for c in cutoff] if kind == "bandpass" else cutoff / nyq
    sos = signal.butter(order, wn, btype=kind, output="sos")
    return signal.sosfiltfilt(sos, y, axis=-1)


# ---------- Séparation voix / instru ----------
def split_center(y, sr, bass_cut=120):
    """Méthode DSP classique : la voix est (souvent) au centre.
    voix ~ Mid = (L+R)/2 ; instru ~ Side = (L-R)/2 (annulation de phase).
    Approximatif, mais 100% traitement du signal."""
    L, R = y[0], y[1]
    mid = (L + R) / 2
    side = (L - R) / 2
    # la voix vit surtout au-dessus de ~120 Hz : on enlève les basses (kick/basse)
    vocal = butter_filter(mid, sr, "highpass", bass_cut)
    vocal = np.stack([vocal, vocal])
    instru = np.stack([side, side])
    return vocal, instru


def separate_demucs(y, sr):
    """Séparation par deep learning (Demucs). pip install demucs. Lent sur CPU."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.wav")
        save_wav(inp, y, sr)
        subprocess.run(
            [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", tmp, inp],
            check=True,
        )
        base = os.path.join(tmp, "htdemucs", "in")
        v, _ = load_audio(os.path.join(base, "vocals.wav"), sr)
        i, _ = load_audio(os.path.join(base, "no_vocals.wav"), sr)
    return v, i


# ---------- Effets ----------
def reverb(y, sr, decay=1.5, wet=0.3, predelay_ms=20):
    """Réverb par convolution avec une réponse impulsionnelle synthétique
    (bruit blanc x enveloppe exponentielle, RT60 = decay). IR différente par canal -> largeur stéréo."""
    n = int(sr * decay)
    t = np.arange(n) / sr
    env = np.exp(-6.9 * t / decay)  # -60 dB à t = decay
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        rng = np.random.default_rng(ch)
        ir = rng.standard_normal(n) * env
        ir = signal.sosfilt(  # assombrit la queue (absorption des aigus)
            signal.butter(2, 6000 / (sr / 2), output="sos"), ir
        )
        ir = np.concatenate([np.zeros(int(sr * predelay_ms / 1000)), ir])
        ir /= np.sqrt(np.sum(ir**2))
        out[ch] = signal.fftconvolve(y[ch], ir)[: y.shape[1]]
    return (1 - wet) * y + wet * out


def _stretch(x, rate, n_fft=2048, hop=512):
    """Time-stretch par vocodeur de phase, 100% vectorisé en float32/complex64
    (STFT -> interpolation des modules -> accumulation de phase par cumsum -> ISTFT).
    rate < 1 = plus long."""
    _, _, Z = signal.stft(x, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
    steps = np.arange(0, Z.shape[1] - 1, rate)
    idx = steps.astype(int)
    a = (steps - idx).astype(np.float32)
    mag = (1 - a) * np.abs(Z[:, idx]) + a * np.abs(Z[:, idx + 1])
    om = ((2 * np.pi * hop / n_fft) * np.arange(Z.shape[0]))[:, None].astype(np.float32)
    dph = np.angle(Z[:, idx + 1]) - np.angle(Z[:, idx]) - om
    dph -= np.float32(2 * np.pi) * np.round(dph / np.float32(2 * np.pi))  # repliement [-pi, pi]
    dph += om
    ph = np.empty_like(dph)
    ph[:, 0] = np.angle(Z[:, 0])
    ph[:, 1:] = ph[:, :1] + np.cumsum(dph[:, :-1], axis=1)
    _, out = signal.istft(mag * np.exp(1j * ph), nperseg=n_fft, noverlap=n_fft - hop)
    return out


def pitch_shift(y, sr, semitones):
    """Change la fréquence fondamentale (en demi-tons) sans changer la durée :
    time-stretch x p (vocodeur de phase) puis rééchantillonnage /p."""
    if semitones == 0:
        return y
    p = 2 ** (semitones / 12)
    fr = Fraction(p).limit_denominator(200)  # p ~ num/den (erreur < 0.1 cent)
    n = y.shape[-1]
    out = np.zeros((y.shape[0], n), dtype=np.float32)
    for c in range(y.shape[0]):
        s = _stretch(y[c].astype(np.float32), 1 / p)
        r = signal.resample_poly(s, fr.denominator, fr.numerator)[:n]
        out[c, : len(r)] = r
    return out


def estimate_f0(y, sr, max_seconds=20):
    """Estime la f0 médiane (Hz) avec pYIN, sur du mono (librosa chargé à la demande)."""
    import librosa

    mono = y.mean(axis=0)[: int(sr * max_seconds)]
    f0, voiced, _ = librosa.pyin(
        mono, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else None


def spectrum(y, sr):
    """Densité spectrale (Welch) en dB sur du mono -> (f, dB)."""
    mono = y.mean(axis=0)
    f, p = signal.welch(mono, sr, nperseg=4096)
    return f, 10 * np.log10(p + 1e-12)