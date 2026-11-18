# fretlift

Generate a guitar tab from a local audio file.

fretlift runs entirely on your own machine. It separates the guitar from the
mix, transcribes the notes, finds the beat, picks a playable string and fret for
every note, and writes a tab you can open in MuseScore, Guitar Pro, TuxGuitar or
the bundled browser viewer.

```
audio ──► Demucs ──► transcriber ──► notes ──► quantize ──► fingering ──► MusicXML
 (mix)   (guitar     (pitch, onset,  (sec)     to beats     (Viterbi       + ASCII tab
          stem)       string)                  grid)        search)
            librosa beat tracking (on the full mix) ──┘
```

> **Status:** moving from Phase 1 (pretrained models plus a heuristic fingering
> search) to Phase 2, a transcription model you train yourself. See
> [Transcription model](#transcription-model) and the [roadmap](#roadmap).

## Intended use

fretlift is a transcription aid for:

- recordings of **your own playing** (practice, songwriting, lessons)
- **public-domain** music and recordings
- music you are **licensed** or otherwise have the rights to transcribe

It processes local files only. It has no download feature, uploads nothing, and
stores nothing outside the output folder you choose.

**You are responsible for how you use it and what you do with its output.**
A tab of a song is a representation of the underlying composition, which is
usually protected by copyright regardless of whether a person or a program
wrote the tab. Publishing or sharing tabs of songs you don't have rights to may
infringe that copyright. This project doesn't give legal advice; if you plan to
publish anything, check the law where you live.

## Install

Python 3.10+. The `ml` extra needs Python 3.10 or 3.11 on Linux and Windows: Basic Pitch pins
TensorFlow < 2.15.1, which has no wheels for Python 3.12 or newer.

```bash
git clone https://github.com/jdinkha/fretlift && cd fretlift
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"          # Demucs, Basic Pitch, librosa
pip install -e ".[dev,eval]"    # optional: tests, linting, evaluation
```

The core (quantization, fingering, export) only needs NumPy, so `pip install -e ".[dev]"`
is enough to run the tests.

A GPU speeds up Demucs a lot but isn't required. Expect a few minutes per song on CPU.

## Usage

```bash
# Full mix: separate the guitar first
fretlift song.wav -o out/

# Already a solo guitar recording: skip separation (much faster)
fretlift my_take.wav --no-separate -o out/

# Drop D, capo 2, known tempo, keep the raw notes for debugging
fretlift song.flac --tuning drop-d --capo 2 --tempo 92 --save-notes -o out/

# Two guitars panned hard left and right: one tab per side (separates once)
fretlift song.wav --channel both -o out/    # out/song.left.tab.txt, out/song.right.tab.txt

# ...and strip what's centred in both channels (usually bass bleed)
fretlift song.wav --channel both --remove-center -o out/
```

Outputs:

| File | What it is |
| --- | --- |
| `out/<name>.musicxml` | Tab with string/fret for every note. Opens in MuseScore, Guitar Pro, TuxGuitar, alphaTab |
| `out/<name>.tab.txt` | Plain-text tab |
| `out/<name>.notes.json` | Raw transcribed notes and beats (`--save-notes`) |
| `out/<name>.guitar.wav` | Separated guitar stem (`--keep-stem`) |
| `out/<name>.guitar.left.wav`, `.right.wav` | One side of the stem (`--keep-stem` with `--channel`) |

Run `fretlift --help` for every option (tunings, time signature, grid
resolution, transcription thresholds).

### Transcription model

The transcriber is a model trained for guitar that also hears which string each
note is on. fretlift doesn't ship weights, so you train it yourself (a free
Colab GPU is enough): see [docs/TRAINING.md](docs/TRAINING.md). Then install
the checkpoint where `fretlift` looks for it:

```bash
mkdir -p ~/.local/share/fretlift
cp runs/gs-aug/best.pt ~/.local/share/fretlift/model.pt
# or, for a one-off: FRETLIFT_MODEL=runs/gs-aug/best.pt fretlift song.wav
```

Until a checkpoint is there, `fretlift` warns and falls back to
[Basic Pitch](https://github.com/spotify/basic-pitch). That fallback is
**deprecated**: it's weak on isolated guitar stems and will be removed once a
trained model is the norm.

### Viewer

`viewer/index.html` renders a tab and plays it back with
[alphaTab](https://alphatab.net). It runs in your browser and never uploads the file.

```bash
python -m http.server 8000
# open http://localhost:8000/viewer/ and drop in out/<name>.musicxml
```

## Try it on the bundled sample

There's no audio in this repo. Instead, `scripts/make_sample.py` synthesizes a
plucked-string performance of a public-domain melody (Beethoven's "Ode to Joy"
theme, 1824) together with an exact answer key.

```bash
python scripts/make_sample.py                     # -> samples/ode_to_joy.wav + .notes.json
fretlift samples/ode_to_joy.wav --no-separate --save-notes -o out/
python scripts/evaluate.py samples/ode_to_joy.notes.json out/ode_to_joy.notes.json
```

`evaluate.py` reports note-level precision, recall and F1 (mir_eval, 50 ms
onset tolerance), the standard transcription metric.

## How it works

| Stage | Module | Tool |
| --- | --- | --- |
| Source separation | `separation.py` | [Demucs](https://github.com/facebookresearch/demucs) `htdemucs_6s` (has a guitar stem) |
| Note transcription | `transcription.py` | Your trained model (`training/`): pitch, onset and string per note. Falls back to Basic Pitch (deprecated) |
| Beat tracking | `beats.py` | [librosa](https://librosa.org), run on the full mix because drums make the beat easier to find |
| Quantization | `quantize.py` | Snaps onsets to a grid between detected beats (handles tempo drift); groups simultaneous notes into chords |
| Fingering | `fretting.py` | Viterbi search over (fingering, hand position) states |
| Export | `export/` | MusicXML with a 6-line TAB staff, plus ASCII |

### Fingering assignment

The same pitch can be played in up to four or five places on the neck, so a list
of notes isn't a tab yet. `fretting.py` enumerates every way to play each note
or chord (one note per string), then runs dynamic programming over the whole
piece with these costs:

- **stretch:** wide spans cost more, spans beyond four frets cost a lot
- **height:** a slight preference for lower positions
- **hand shifts:** a flat cost to move the hand, plus a little per fret moved;
  open strings can be played from any position
- **heard strings:** when the trained model says which string a note is on,
  putting it anywhere else costs extra (`FretConfig.w_string`), so the model
  decides and the search only overrules it where the result would be awkward

Hand position is part of the search state. Without it, a melody can "creep" up
the neck one cheap step at a time, which is the classic failure of greedy
approaches. With it, a one-octave C major scale comes out in a single
position, and the "Ode to Joy" answer key comes out in standard first position.

## Known limitations

- **Dense mixes** (distorted, double-tracked rhythm guitars) separate and
  transcribe poorly. Clean and acoustic recordings work far better.
- **Techniques** (bends, slides, hammer-ons, palm muting, harmonics) are not detected yet.
- **Rhythm:** no downbeat detection (measure 1 starts on the first beat with a
  note), no triplets, a fixed time signature.
- **Fingering** is a hand-tuned heuristic. It's playable, but not always what
  the original guitarist played.
- **One voice:** each chord lasts until the next onset, the usual way tabs are
  written, but it loses independent bass lines.

## Roadmap

1. **Phase 1:** pretrained models + heuristic fingering + MusicXML.
2. **Phase 2 (in progress):** a model trained for guitar that predicts pitch,
   onset and string from audio, evaluated with tab-level F1 (right string *and*
   fret). The training code and the pipeline hook are in; next is training on
   GuitarSet with amp augmentation, then removing Basic Pitch.
3. **Phase 3:** a symbolic model that reranks fingerings to make them more idiomatic.

Other ideas: downbeat and time-signature detection, technique detection,
Guitar Pro export, in-browser inference.

## Installing

```bash
pip install -e ".[dev]"
pytest          # core tests run without any ML dependencies
ruff check .
```

## License

Code: [MIT](LICENSE). Third-party models and libraries keep their own licenses;
see [DATA_AND_MODELS.md](DATA_AND_MODELS.md).
