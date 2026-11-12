"""
Tests for drift.corpora.coraal.

Uses a tiny fake corpus (short synthetic WAVs in a temp folder) laid out like
data/coraal/, so no real CORAAL audio is needed to run the tests.
"""

import numpy as np
import pytest
import soundfile as sf

from drift.corpora.coraal import find_recordings, iter_chunks


NATIVE_SR = 44_100


@pytest.fixture
def fake_corpus(tmp_path):
    root = tmp_path / "coraal"
    # Filenames are deliberately arbitrary: the loader must not depend on them.
    specs = {
        "ATL": [("interview one.wav", 7.0, 1), ("rec.wav", 3.0, 1)],
        "DCA": [("rec.wav", 5.0, 2),            # same name as in ATL, and stereo
                ("nested/LOUD.WAV", 2.0, 1)],   # subfolder + uppercase extension
    }
    for comp, files in specs.items():
        for name, dur, ch in files:
            path = root / comp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            t = np.arange(int(dur * NATIVE_SR)) / NATIVE_SR
            y = 0.1 * np.sin(2 * np.pi * 220 * t)
            if ch == 2:
                y = np.stack([y, y], axis=1)
            sf.write(path, y, NATIVE_SR, format="WAV")
            # macOS AppleDouble junk file, like the ones in the real CORAAL archives
            (path.parent / f"._{path.name}").write_bytes(b"\x00\x05\x16\x07junk")
    # Non-audio files are ignored
    (root / "ATL" / "notes.txt").write_text("not audio")
    return root


def by_key(recs, key):
    return next(r for r in recs if r.key == key)


def test_finds_all_wavs_regardless_of_name(fake_corpus):
    recs = find_recordings(fake_corpus)
    assert sorted(r.key for r in recs) == [
        "ATL/interview one.wav", "ATL/rec.wav", "DCA/nested/LOUD.WAV", "DCA/rec.wav",
    ]
    assert {r.component for r in recs} == {"ATL", "DCA"}
    assert by_key(recs, "ATL/interview one.wav").duration_s == pytest.approx(7.0, abs=1e-3)


def test_ids_are_opaque_unique_and_stable(fake_corpus):
    recs = find_recordings(fake_corpus)
    for r in recs:
        # NFR3: nothing from the filename leaks into the ID
        assert r.path.stem.lower() not in r.id
    # ATL/rec.wav and DCA/rec.wav share a name but must not share an ID
    assert len({r.id for r in recs}) == len(recs)
    # Stable across calls
    assert [r.id for r in recs] == [r.id for r in find_recordings(fake_corpus)]


def test_ids_survive_moving_the_corpus(fake_corpus, tmp_path):
    before = {r.key: r.id for r in find_recordings(fake_corpus)}
    moved = fake_corpus.rename(tmp_path / "somewhere_else")
    after = {r.key: r.id for r in find_recordings(moved)}
    assert before == after


def test_chunks_cover_whole_file(fake_corpus):
    rec = by_key(find_recordings(fake_corpus), "ATL/interview one.wav")  # 7 s
    chunks = list(iter_chunks(rec, chunk_s=3.0, sr=16_000))
    assert [c.start_s for c in chunks] == pytest.approx([0.0, 3.0, 6.0])
    total = sum(len(c.audio) for c in chunks) / 16_000
    assert total == pytest.approx(7.0, abs=0.01)
    assert all(c.audio.dtype == np.float32 and c.audio.ndim == 1 for c in chunks)


def test_stereo_is_downmixed(fake_corpus):
    rec = next(r for r in find_recordings(fake_corpus) if r.channels == 2)
    chunk = next(iter_chunks(rec, chunk_s=1.0))
    assert chunk.audio.ndim == 1


def test_missing_root_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_recordings(tmp_path / "nope")