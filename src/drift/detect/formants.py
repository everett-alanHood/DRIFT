"""
Stage 1 (Detect) - FR1: Extract F1/F2 formant frequencies via LPC + STFT.

How it works, in short:
    1. Resample the audio so the Nyquist frequency equals the formant ceiling.
       LPC then only "spends" its poles on the 0..ceiling band, where F1-F3 live.
    2. Pre-emphasize to flatten the natural spectral tilt of voiced speech
       (otherwise LPC over-fits the low frequencies and misses F2/F3).
    3. STFT: compute per-frame energy and spectral flatness. These gate which
       frames are voiced/vowel-like enough to analyze. Silence, fricatives, and
       noise get NaN instead of garbage formants.
    4. LPC per gated frame: fit an all-pole model of the vocal tract, find the
       roots of the polynomial, and turn each complex root into a resonance
       (frequency + bandwidth). The lowest plausible resonances are F1, F2, F3.

Design constraints this module respects:
    - NFR2: pure function of an in-memory array. Nothing is written to disk.
    - NFR3: no speaker metadata is taken as input. In particular, the formant
      ceiling is a single configurable value rather than the usual
      "5000 Hz for male / 5500 Hz for female" rule, which would require sex as an
      input. A speaker-adaptive ceiling (e.g., Escudero et al. 2009's
      optimal-ceiling method, chosen from the audio alone) is a planned upgrade.

This module finds formants in *frames*. Picking out which frames belong to
which vowel (via transcripts / forced alignment) happens in vowels.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import scipy.signal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormantConfig:
    """Analysis parameters. Defaults follow common phonetics practice (Praat-like)."""

    max_formant_hz: float = 5500.0   # formant ceiling; analysis sr = 2x this
    n_formants: int = 3              # how many formants to report (F1..F3)
    lpc_order: int | None = None     # None -> rule of thumb (see `order`)
    frame_ms: float = 25.0           # analysis window length
    hop_ms: float = 10.0             # step between frames
    preemph_from_hz: float = 50.0    # pre-emphasis corner (Praat default)

    # Resonance plausibility filters
    min_formant_hz: float = 90.0     # below this is usually the glottal source, not a formant
    max_bandwidth_hz: float = 400.0  # broader peaks are rarely true formants

    # Frame gating (computed from the STFT)
    energy_floor_db: float = -35.0   # frames this far below the loudest frame are skipped
    max_flatness: float = 0.3        # high spectral flatness = noise-like (fricatives, silence)

    @property
    def analysis_sr(self) -> int:
        return int(round(2 * self.max_formant_hz))

    @property
    def order(self) -> int:
        # Classic rule: (sample rate in kHz) + 2. Roughly one resonance per kHz
        # of bandwidth (2 poles each), plus 2 poles to absorb glottal/radiation
        # shape. With the default 5500 Hz ceiling: 11 kHz -> order 13.
        if self.lpc_order is not None:
            return self.lpc_order
        return int(round(self.analysis_sr / 1000)) + 2


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class FormantTrack:
    """Frame-by-frame formant estimates.

    Arrays all have shape (n_frames,) except `formants` / `bandwidths`,
    which are (n_frames, n_formants). Frames that were gated out or had no
    plausible resonances hold NaN.
    """

    times: np.ndarray        # frame centers, seconds
    formants: np.ndarray     # Hz, column k = F(k+1)
    bandwidths: np.ndarray   # Hz
    voiced: np.ndarray       # bool, frame passed the STFT gate

    @property
    def f1(self) -> np.ndarray:
        return self.formants[:, 0]

    @property
    def f2(self) -> np.ndarray:
        return self.formants[:, 1]

    def median(self) -> np.ndarray:
        """Median of each formant over voiced frames (NaN-safe). Handy for quick checks."""
        return np.nanmedian(self.formants, axis=0)


# ---------------------------------------------------------------------------
# Core pieces
# ---------------------------------------------------------------------------

def preemphasize(y: np.ndarray, sr: int, from_hz: float) -> np.ndarray:
    """First-order high-pass that boosts ~6 dB/octave above `from_hz`."""
    alpha = np.exp(-2.0 * np.pi * from_hz / sr)
    return scipy.signal.lfilter([1.0, -alpha], [1.0], y)


def stft_gate(y: np.ndarray, frame_len: int, hop: int, cfg: FormantConfig) -> np.ndarray:
    """Use the STFT to decide which frames are worth running LPC on.

    A frame passes if it is (a) loud enough relative to the loudest frame and
    (b) "peaky" rather than noise-like (low spectral flatness). Vowels are loud
    and peaky; silence and fricatives fail one or both.
    """
    S = np.abs(librosa.stft(y, n_fft=frame_len, hop_length=hop,
                            win_length=frame_len, center=True)) ** 2
    energy_db = librosa.power_to_db(S.sum(axis=0), ref=np.max)
    flatness = librosa.feature.spectral_flatness(S=S, power=1.0)[0]
    return (energy_db > cfg.energy_floor_db) & (flatness < cfg.max_flatness)


def lpc_resonances(frame: np.ndarray, sr: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit LPC to one frame and return (frequencies, bandwidths) of its resonances, sorted.

    Each complex-conjugate root pair r of the LPC polynomial is one resonance:
        frequency = angle(r) * sr / (2*pi)
        bandwidth = -ln|r| * sr / pi
    Roots near the unit circle (|r| ~ 1) are sharp peaks -> likely formants.
    """
    a = librosa.lpc(frame, order=order)       # Burg's method
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]         # keep one of each conjugate pair
    freqs = np.angle(roots) * sr / (2 * np.pi)
    bws = -np.log(np.abs(roots)) * sr / np.pi
    idx = np.argsort(freqs)
    return freqs[idx], bws[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_formants(y: np.ndarray, sr: int, cfg: FormantConfig | None = None) -> FormantTrack:
    """Estimate F1..Fn for every frame of a mono speech signal.

    Args:
        y:   mono audio, float, any sample rate.
        sr:  sample rate of `y`.
        cfg: analysis settings (defaults are fine for adult conversational speech).

    Returns:
        FormantTrack with per-frame estimates (NaN where nothing reliable was found).
    """
    cfg = cfg or FormantConfig()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("extract_formants expects mono audio (1-D array).")

    # 1. Resample to 2x the formant ceiling
    target_sr = cfg.analysis_sr
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

    frame_len = int(round(cfg.frame_ms * sr / 1000))
    hop = int(round(cfg.hop_ms * sr / 1000))

    # 2. STFT-based gating (center=True -> frame i is centered at i*hop).
    #    Done BEFORE pre-emphasis: pre-emphasis tilts a flat noise spectrum,
    #    which would make noise look "peaky" and slip through the flatness test.
    voiced = stft_gate(y, frame_len, hop, cfg)
    n_frames = len(voiced)

    # 3. Pre-emphasis (only affects what LPC sees)
    y = preemphasize(y, sr, cfg.preemph_from_hz)

    # Frame the signal the same way the STFT did (pad by half a window each side)
    y_pad = np.pad(y, frame_len // 2)
    window = np.hamming(frame_len)

    formants = np.full((n_frames, cfg.n_formants), np.nan)
    bandwidths = np.full((n_frames, cfg.n_formants), np.nan)

    # 4. LPC on each voiced frame
    for i in np.flatnonzero(voiced):
        frame = y_pad[i * hop: i * hop + frame_len]
        if len(frame) < frame_len:
            break
        frame = frame * window
        try:
            freqs, bws = lpc_resonances(frame, sr, cfg.order)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue  # ill-conditioned frame (e.g., near-silent); leave NaN

        keep = (freqs > cfg.min_formant_hz) & (bws < cfg.max_bandwidth_hz)
        freqs, bws = freqs[keep][: cfg.n_formants], bws[keep][: cfg.n_formants]
        formants[i, : len(freqs)] = freqs
        bandwidths[i, : len(bws)] = bws

    times = np.arange(n_frames) * hop / sr
    return FormantTrack(times=times, formants=formants, bandwidths=bandwidths, voiced=voiced)
