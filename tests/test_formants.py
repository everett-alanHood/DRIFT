"""
Tests for drift.detect.formants (FR1).

We can't check against CORAAL here, so we synthesize vowels with *known*
formants using a classic source-filter model: a glottal pulse train passed
through a cascade of second-order resonators. If the extractor recovers the
formants we put in, the LPC pipeline is doing its job.

Reference targets are rough adult values (Peterson & Barney 1952 style).
"""

import numpy as np
import pytest
import scipy.signal

from drift.detect.formants import FormantConfig, extract_formants


SR = 16_000  # CORAAL/Whisper-typical rate; the extractor resamples internally


def synth_vowel(formants_hz, f0=120.0, dur=0.5, sr=SR, bw_hz=(80, 100, 120), seed=0):
    """Source-filter vowel: impulse train at f0 -> resonator per formant."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    # Impulse train with slight jitter so it isn't perfectly periodic
    period = sr / f0
    pulses = np.zeros(n)
    t = 0.0
    while t < n:
        pulses[int(t)] = 1.0
        t += period * (1 + 0.01 * rng.standard_normal())
    # Glottal roll-off (~ -12 dB/oct)
    source = scipy.signal.lfilter([1.0], [1.0, -1.96, 0.9604], pulses)

    y = source
    for f, bw in zip(formants_hz, bw_hz):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        y = scipy.signal.lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], y)
    y = y / np.max(np.abs(y))
    return y + 1e-4 * rng.standard_normal(n)  # a little noise floor


@pytest.mark.parametrize(
    "vowel, target",
    [
        ("i (beet)", (280, 2250, 2900)),
        ("ae (bat)", (690, 1660, 2490)),
        ("a (bot)", (730, 1090, 2440)),
        ("u (boot)", (310, 870, 2250)),
    ],
)
def test_recovers_known_formants(vowel, target):
    y = synth_vowel(target)
    track = extract_formants(y, SR)
    f1, f2 = track.median()[:2]
    # 10% tolerance: LPC on a 120 Hz pulse train is biased toward harmonics,
    # especially for low F1, so exact recovery isn't expected.
    assert f1 == pytest.approx(target[0], rel=0.10), f"{vowel}: F1={f1:.0f}"
    assert f2 == pytest.approx(target[1], rel=0.10), f"{vowel}: F2={f2:.0f}"


def test_silence_is_gated_out():
    y = np.concatenate([np.zeros(SR // 4), synth_vowel((700, 1200, 2500)), np.zeros(SR // 4)])
    track = extract_formants(y, SR)
    # Frames in the leading silence should be unvoiced and NaN
    lead = track.times < 0.2
    assert not track.voiced[lead].any()
    assert np.isnan(track.f1[lead]).all()
    # ...and most frames in the vowel should be analyzed
    mid = (track.times > 0.3) & (track.times < 0.7)
    assert track.voiced[mid].mean() > 0.9


def test_white_noise_is_mostly_rejected():
    y = np.random.default_rng(1).standard_normal(SR)
    track = extract_formants(y, SR)
    assert track.voiced.mean() < 0.1


def test_output_shapes_and_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # NFR2 sanity check: nothing gets written to cwd
    y = synth_vowel((500, 1500, 2500))
    track = extract_formants(y, SR, FormantConfig(n_formants=3))
    assert track.formants.shape == (len(track.times), 3)
    assert track.bandwidths.shape == track.formants.shape
    assert list(tmp_path.iterdir()) == []


def test_rejects_stereo():
    with pytest.raises(ValueError):
        extract_formants(np.zeros((2, SR)), SR)
