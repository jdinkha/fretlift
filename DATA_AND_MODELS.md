# Data and model statement

This file records what models and data the project uses and under what terms.
Keep it up to date whenever a model, dataset or dependency changes.

## Phase 1: pretrained models (no training in this repo)

fretlift doesn't ship model weights. The libraries below download their own
pretrained weights the first time they run.

| Component | Used for | License (verify before each release) |
| --- | --- | --- |
| [Demucs](https://github.com/facebookresearch/demucs) (`htdemucs_6s`) | Guitar source separation | MIT |
| [Basic Pitch](https://github.com/spotify/basic-pitch) | Note transcription | Apache-2.0 |
| [librosa](https://github.com/librosa/librosa) | Audio loading, beat tracking | ISC |
| [alphaTab](https://github.com/CoderLine/alphaTab) (viewer, via CDN) | Tab rendering and playback | MPL-2.0 |
| [mir_eval](https://github.com/craffel/mir_eval) (optional) | Evaluation metrics | MIT |

Check each project's repository for the current license of its code **and its
weights**, which aren't always the same. These licenses are compatible with
using the libraries from MIT-licensed code, but that's worth re-confirming if
you ever bundle or redistribute weights.

## Test data

- **No audio is committed to this repository.**
- `scripts/make_sample.py` synthesizes test audio from scratch (Karplus-Strong
  synthesis) for a public-domain composition: the "Ode to Joy" theme from
  Beethoven's Symphony No. 9 (1824). The answer key comes from the same script.
- Unit tests use hand-written note lists.

## Phase 2 and later: training data policy

Training data will be limited to sources whose terms allow it, each recorded here
with its license and how it's used:

| Dataset | Content | Planned use | License |
| --- | --- | --- | --- |
| GuitarSet | Acoustic guitar with hexaphonic (per-string) annotations | Fine-tuning and evaluation | check before use |
| SynthTab | Large synthetic guitar-tab audio dataset | Pretraining | check before use |
| Own recordings | Recordings by the author, with tabs | Fine-tuning, demos | author's own |

Not used: scraped tabs from tab websites, or commercial recordings without permission.
Datasets built from scraped user-made tabs (for example DadaGP) will only be
considered after reviewing their terms, and will be documented here if used.
