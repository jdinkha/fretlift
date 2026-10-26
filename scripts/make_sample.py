"""Synthesize a public-domain test recording with a known answer key.

The melody is the "Ode to Joy" theme from Beethoven's Symphony No. 9 (1824),
which is in the public domain. The audio is generated here with Karplus-Strong
plucked-string synthesis, so no third-party recording is involved and nothing
copyrighted ever needs to be committed to the repo.

Writes:
  samples/ode_to_joy.wav         the audio
  samples/ode_to_joy.notes.json  ground-truth notes, for scripts/evaluate.py

Usage:
  python scripts/make_sample.py [--bpm 100] [--out samples]
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 44100

# (MIDI pitch, length in beats). C major, 4/4.
E4, F4, G4, D4, C4 = 64, 65, 67, 62, 60
PHRASE_A = [(E4, 1), (E4, 1), (F4, 1), (G4, 1), (G4, 1), (F4, 1), (E4, 1), (D4, 1), (C4, 1), (C4, 1), (D4, 1), (E4, 1)]
MELODY = (
    PHRASE_A + [(E4, 1.5), (D4, 0.5), (D4, 2)]
    + PHRASE_A + [(D4, 1.5), (C4, 0.5), (C4, 2)]
)


def pluck(pitch: int, seconds: float, rng: np.random.Generator, decay: float = 0.996) -> np.ndarray:
    """Karplus-Strong: a noise burst circulating through a damped delay line."""
    freq = 440.0 * 2 ** ((pitch - 69) / 12)
    period = int(round(SR / freq))
    n = int(SR * seconds)
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        j = i % period
        out[i] = buf[j]
        buf[j] = decay * 0.5 * (buf[j] + buf[(j + 1) % period])
    fade = min(n, int(0.01 * SR))  # damp the string at note-off to avoid clicks
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def render(bpm: float, lead_in: float = 0.5, seed: int = 0) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    total = lead_in + sum(b for _, b in MELODY) * beat + 1.0
    audio = np.zeros(int(SR * total))
    notes = []
    t = lead_in
    for pitch, beats in MELODY:
        dur = beats * beat * 0.95  # slight gap so repeated notes re-articulate
        start = int(t * SR)
        sound = pluck(pitch, dur, rng)
        audio[start : start + len(sound)] += sound
        notes.append({"start": round(t, 4), "end": round(t + dur, 4), "pitch": pitch})
        t += beats * beat
    audio /= np.max(np.abs(audio)) * 1.1
    return audio, notes


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype("<i2").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--out", default="samples")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio, notes = render(args.bpm)
    write_wav(out / "ode_to_joy.wav", audio)
    (out / "ode_to_joy.notes.json").write_text(json.dumps({"tempo": args.bpm, "notes": notes}, indent=1))
    print(f"wrote {out / 'ode_to_joy.wav'} ({len(audio) / SR:.1f}s) and the answer key")


if __name__ == "__main__":
    main()
