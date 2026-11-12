# Training your own transcription model

This is Phase 2: replace Basic Pitch with a model trained for guitar, that
also predicts which string each note is on. The plan:

1. **Measure first.** Build evaluation sets for the cases you care about and
   get Basic Pitch's scores, so every change has a number attached.
2. **Train on clean data.** GuitarSet, leave-one-player-out.
3. **Augment.** Amp/distortion, pitch shifting (down-tuning) and separation
   artifacts, then compare against step 2 on distorted audio.

Everything below runs on a laptop CPU for small tests, but training wants a
GPU. A free Colab GPU is enough: see `notebooks/train_colab.ipynb`.

## Install

```bash
# On a GPU machine, install the PyTorch build for your CUDA version first:
# https://pytorch.org/get-started/locally/
pip install -e ".[train,ml,dev]"
pytest -q   # the torch tests run once torch is installed
```

`[ml]` (Basic Pitch, Demucs) is needed for the baseline and for
`mix-separate`. If Basic Pitch won't install alongside your PyTorch, use a
separate virtual environment for baseline evaluation; it shares the same data
and manifests.

## The model, briefly

```
waveform -> CQT (2 bins per semitone) -> harmonic stack -> 5 conv layers -> BiGRU -> frame   [T x 56 pitches]
                                                                                   -> onset   [T x 56]
                                                                                   -> string  [T x 56 x 6]
```

- **CQT + harmonic stacking** (the Basic Pitch trick): copies of the spectrum
  are shifted so each note's harmonics line up with its fundamental. With
  distortion the harmonics are often louder than the fundamental, so seeing
  them together matters.
- **Separate onset head:** a note starts here. This is what splits repeated
  notes and soft re-plucks, the main failure on gentle acoustic songs.
- **String head, independent of pitch:** for each sounding pitch, which string
  plays it. The model never needs to know the tuning; a drop-C riff and the
  same riff in E standard differ only in pitch. At decode time the tuning
  turns (pitch, string) into a fret, and impossible combinations are ruled
  out.
- About 1M parameters at the defaults (`--channels 32 --hidden 256`).

Labels are frame-level targets built from note lists
(`training/labels.py: make_targets`); decoding turns probabilities back into
notes (`training/decode.py`).

## Step 1: evaluation sets and the baseline

### Data

All datasets become **manifests**: one JSON line per recording with its audio,
labels, a **group** and **tags**. Groups keep related recordings together when
splitting (every take by a GuitarSet player; every amp render of an EGDB take),
so the model is never tested on something it effectively trained on. Tags
(`acoustic`, `electric`, `amp`, `separated`, ...) are how results get broken
down.

**GuitarSet** (acoustic, per-string labels from a hexaphonic pickup, ~3 h, 6 players):

```bash
fretlift-data guitarset --root data/guitarset --download
fretlift-data stats --manifest data/guitarset/manifest.jsonl
```

Each excerpt gives two items, the microphone recording (`mic`) and the
pickup mix (`pickup`). Groups are `guitarset-player-00` ... `-05`.

**EGDB** (electric; DI recordings re-rendered through several amps). Download
it from the authors (see their paper and project page), then:

```bash
fretlift-data egdb --root data/egdb
```

The converter looks for `audio_<tone>/` folders and MIDI/JAMS labels with
matching file names. I couldn't confirm EGDB's exact layout, so check the
`stats` output: if it can't find the files it prints what it did find, and
`fretlift-data folder` (below) works per tone folder as a fallback. EGDB takes
are split 80/10/10 by take, all tones together.

**Your own recordings: the most valuable eval set.** A dozen short clips of
exactly what fails today (gentle fingerpicking, palm-muted metal riffs) tell
you more than any benchmark. Put audio and labels side by side with matching
names:

```
mine/audio/riff1.wav     mine/labels/riff1.mid   (or .json / .jams)
```

```bash
fretlift-data folder --audio mine/audio --labels mine/labels --out data/mine \
    --name mine --tags electric,distorted --split test --tuning drop-d
```

Ways to get labels for your own playing:

- **MIDI guitar pickup** (hexaphonic, e.g. Roland GK or Fishman TriplePlay):
  record the pickup's MIDI and the guitar audio at the same time. If the MIDI
  has one track or channel per string, string labels come for free
  (`--midi-strings auto` detects that). Better still: record the **DI** signal
  and re-amp it through any amp sims you like. Every render keeps the same
  labels, which gives you real labelled metal data.
- **Play to a click from a tab you wrote:** write the part in MuseScore or Guitar
  Pro, export MIDI, record yourself playing along to its click, then nudge
  the MIDI in a DAW until it lines up.
- **Correct a transcription:** run `fretlift --save-notes`, fix the notes in
  a DAW piano roll, and export MIDI.

Label quality matters more than quantity for an eval set: 10 carefully
aligned clips beat 100 sloppy ones.

### Baseline numbers

```bash
fretlift-eval --basic-pitch --split test \
    --manifest data/guitarset/manifest.jsonl --test-groups guitarset-player-05 \
    --out reports/basic-pitch-gs-p05.json

fretlift-eval --basic-pitch --split test --distort high_gain \
    --manifest data/guitarset/manifest.jsonl --test-groups guitarset-player-05

fretlift-eval --basic-pitch --split test --manifest data/mine/manifest.jsonl
```

Columns:

| Column | Meaning |
| --- | --- |
| note P / R / F1 | right pitch, onset within 50 ms (the standard metric) |
| +offset F1 | also right length (offset within 20% of the note) |
| tab F1 | right pitch, onset **and string** (so the right fret) |
| string acc | of the notes found, how many are on the right string |

For Basic Pitch, strings come from the Phase 1 fingering search, so tab F1
compares "audio model's strings" against "heuristic strings".

`--distort clean_amp|crunch|high_gain` runs the test audio through a fixed,
seeded amp preset: a repeatable stand-in for distorted guitar until you have
real labelled metal recordings. Real recordings beat it; use both.

### Threshold tuning: one setting, or several?

Thresholds trade missed notes against invented ones, and the best trade-off
depends on the material. Soft acoustic attacks need a low onset threshold;
distortion adds harmonic and noise energy that a low threshold turns into
false notes. Measure it on **validation** data:

```bash
fretlift-sweep --basic-pitch --split val \
    --manifest data/guitarset/manifest.jsonl --val-groups guitarset-player-04 \
    --manifest data/mine/manifest.jsonl --out reports/sweep-basic-pitch.json
```

It prints the best setting per tag, the best single setting, and how much F1
each tag loses with the single setting. If nothing loses more than ~0.02, use
one setting. If one kind of recording loses a lot, keep a separate setting
for it (e.g. `--onset-threshold 0.3` for fingerstyle acoustic).

Don't sweep on the test split; tuned-on-test scores look better than they are.
Your own small eval set is best kept as test-only, so tune on GuitarSet/EGDB
validation data plus any of your recordings you put in a val split.

Retune for your own model: its probabilities are scaled differently from
Basic Pitch's. Training does this automatically (see below).

## Step 3: train on clean data

Leave-one-player-out: train on four players, validate on one, test on one.

```bash
fretlift-train --manifest data/guitarset/manifest.jsonl \
    --val-groups guitarset-player-04 --test-groups guitarset-player-05 \
    --out runs/gs-clean
```

Defaults: 30 epochs x 500 steps x 16 clips of 5 s, AdamW at 1e-3 with warmup
and cosine decay, mixed precision on CUDA. Each epoch ends with a validation
pass (note F1 and tab F1), and `best.pt` follows the best validation note F1.

After the last epoch it:

1. tunes onset/frame thresholds on the validation set (stored in `best.pt`
   and `thresholds.json`), and
2. evaluates the test set **once** with those thresholds (`test_report.json`).

The run directory also has `history.csv` (losses and validation scores per
epoch) and `last.pt` (`--resume runs/gs-clean/last.pt` continues a run).

For a number you can quote, repeat with each player held out (6 folds) and
average. A single fold is fine while iterating.

Things to watch in `history.csv`:

- **Loss falling, validation F1 flat or falling:** overfitting. GuitarSet is
  small; augmentation (step 4) is the main fix, `--hidden 128` another.
- **Onset loss stays high:** try `--onset-pos-weight 8`.
- **Validation tab F1 far below note F1:** the string head is the weak point.
  GuitarSet has string labels for every note, so this improves with more
  epochs and data; check `string acc` in the eval table.

## Step 4: augmentation

### Amp, EQ, pitch shift, reverb, noise

```bash
fretlift-data preview-augment data/guitarset/audio_mono-mic/.../00_BN1-129-Eb_comp_mic.wav --out previews/
```

Listen to the previews before training on them. Each clip gets a random
chain: pitch shift (-4 to +2 semitones, labels moved, strings kept), EQ, an
amp (`clean_amp`, `crunch` or `high_gain`: waveshaping distortion with tone
shaping), a cabinet, room reverb, noise and a level change.

**Cabinet impulse responses** make the amp tones far more realistic than the
built-in filter model. Put some cab IR WAVs (many are free; check their
licenses) in a folder:

```bash
fretlift-train --manifest data/guitarset/manifest.jsonl \
    --val-groups guitarset-player-04 --test-groups guitarset-player-05 \
    --augment full --ir-dir irs/ --val-distort high_gain --out runs/gs-aug
```

`--val-distort high_gain` validates on both clean and distorted audio and
picks the best checkpoint on the average, so you don't trade away clean
accuracy without noticing. The test report covers both too.

### Separation artifacts

The real pipeline transcribes Demucs's guitar stem, not clean audio. Mix
labelled guitar with accompaniment, separate it, and train on the result:

```bash
fretlift-data mix-separate --manifest data/guitarset/manifest.jsonl --tags mic \
    --accompaniment accompaniment/ --out data/gs-separated --device cuda
```

This adds new items (tagged `separated`) with the original labels and groups,
so splits stay consistent. Pass both manifests to `fretlift-train`. Use
accompaniment you're allowed to use, such as your own recordings or openly
licensed stems (Slakh2100's non-guitar tracks, for example; check the license).
Not separated commercial songs. Demucs is slow on CPU; run this on the GPU
machine.

### Adding EGDB

```bash
fretlift-train --manifest data/guitarset/manifest.jsonl --manifest data/egdb/manifest.jsonl \
    --val-groups guitarset-player-04 --test-groups guitarset-player-05 \
    --augment full --val-distort high_gain --out runs/gs-egdb-aug
```

EGDB items carry their own split, so they're validated and tested on held-out
takes automatically. If its MIDI has no per-string tracks, those notes still
train pitch and onset; the string head just skips them.

## Comparing runs

Keep a table like this in the README. It's the core of the project's story:

| Model | GuitarSet note F1 | GuitarSet tab F1 | high_gain note F1 | Own metal clips note F1 |
| --- | --- | --- | --- | --- |
| Basic Pitch + heuristic fingering | | | | |
| Trained, clean | | | | |
| Trained, + amp augmentation | | | | |
| Trained, + EGDB + separation | | | | |

```bash
fretlift-eval --model runs/gs-aug/best.pt --split test \
    --manifest data/guitarset/manifest.jsonl --test-groups guitarset-player-05 --distort high_gain
fretlift-eval --model runs/gs-aug/best.pt --split test --manifest data/mine/manifest.jsonl
```

## Using your model

`fretlift` looks for a checkpoint at `~/.local/share/fretlift/model.pt` (or
wherever `FRETLIFT_MODEL` points) and uses it instead of Basic Pitch:

```bash
mkdir -p ~/.local/share/fretlift
cp runs/gs-aug/best.pt ~/.local/share/fretlift/model.pt
fretlift song.wav -o out/
```

The onset and frame thresholds tuned at the end of training are stored in
`best.pt` and used by default; `--onset-threshold` / `--frame-threshold`
override them. The strings the model hears go to the fingering search as
strong hints (`FretConfig.w_string`), so the tab follows the model unless that
would mean an awkward jump for a single note. For tunings with a different
number of strings than the model was trained on, the fingering search places
the notes on its own.

Without a checkpoint, `fretlift` warns and falls back to Basic Pitch, which
is deprecated.

## Troubleshooting

- **"No items selected":** check `--split` and the group names
  (`fretlift-data stats` lists them).
- **Out of GPU memory:** `--batch-size 8` or `--clip-seconds 4`.
- **Data loading is the bottleneck** (GPU utilization low): augmentation runs
  on the CPU, so raise `--workers` up to your CPU count. On Colab (2 CPUs),
  `--augment full` epochs are CPU-bound.
- **Fast Basic Pitch decoding unavailable:** the sweep falls back to
  re-running Basic Pitch per setting. It's correct, just slower; use
  `--onsets/--frames` with fewer values.
- **The first epoch is slow:** audio is decoded and cached on first use
  (`.cache/fretlift/`); later runs reuse it.
