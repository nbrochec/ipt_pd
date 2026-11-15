# ipt~ for Pure Data

Real-time recognition of instrumental playing techniques in Pure Data.
`ipt~` runs a TorchScript classifier trained with
[ipt_recognition](https://github.com/nbrochec/ipt_recognition) on live audio
and outputs the most likely playing technique and the full class distribution.

This is the Pd port of the Max external from
[ipt_tilde](https://github.com/nbrochec/ipt_tilde): the arguments, settings
and outputs are the same, and both use the same models through
[libipt](https://github.com/nbrochec/libipt).

## Install

Copy the `ipt~` folder into `~/Documents/Pd/externals/` (or any folder in
Pd's search path).

Models copied into this `ipt~` folder are found by name from any patch,
e.g. `[ipt~ flute_ircam]`. Pretrained models (electric guitar, flute,
trumpet) are on
[Hugging Face](https://huggingface.co/nbrochec/ipt_models), with their cards in
[ipt_models](https://github.com/DYCI2/ipt_models).

### plugdata

`ipt~` works in the plugdata standalone app (tested with plugdata 0.9.3,
which embeds Pd 0.55.2, on macOS Apple Silicon). Copy the `ipt~` folder into
`~/Documents/plugdata/Externals/`, which is in plugdata's default search
paths, then restart plugdata.

For the time being, it does not work in the plugdata plugins (AU, VST3, CLAP, LV2).

## Use

```
[adc~ 1]
|
[ipt~ flute.ts MPS -sensitivity 0.5]
|        |           |            |
index    name        distribution info (latency, classnames)
```

- Arguments: `ipt~ <model.ts> [CPU|MPS|CUDA]`. The model path is absolute
  (`~/` works) or a name looked up, in this order, next to the patch, in
  `[declare]` paths and Pd's search paths, and in the `ipt~` folder itself:
  models copied next to the external are found by name, like models in the
  Max package's `help` folder. `.ts` is added if missing. A device the
  platform cannot have falls back to CPU.
- Settings are messages (`sensitivity 0.5`) or creation flags
  (`-sensitivity 0.5`, `@sensitivity 0.5` also works):

| Setting | Default | Meaning |
|---|---|---|
| `enabled` | 1 | 0 stops feeding audio to the model |
| `sensitivity` | 1 | 0..1; smoothing time is `(1 - sensitivity) × sensitivityrange` ms, 1 = no smoothing |
| `sensitivityrange` | 2000 | ms, upper bound of the smoothing time |
| `threshold` | -80 | dB RMS gate; quieter audio is not classified (≤ -80 disables it) |
| `window` | 20 | ms, length of the RMS gate window |
| `confidence` | 0 | below this top probability, outputs `-1` and `no_confidence` |
| `period` | 0 | ms; 0 = every inference, > 0 = at most once per period, results in between are integrated |
| `verbose` | 0 | 1 prints libipt's detailed error messages |

- `classnames` sends `classnames <c1> <c2> …` out of the right outlet.
- Outlets, left to right: class index (`-1` below `confidence`), class name
  (`no_confidence` below `confidence`), distribution (always sent), info
  (`latency <ms>` after each output). As usual in Pd they fire right to left.

<!-- Differences with the Max `ipt~`:

- the object is not created when the model file cannot be found;
- outputs fire right to left (Max sends them left to right);
- smoothing is timed on the audio (samples consumed), not on the wall clock;
- `period` integrates every result of the period: results are drained and
  smoothed continuously and only the output is paced.

Inference runs on a background thread, so Pd's `-batch` (faster than real
time) mode is not supported: the model would not keep up and audio would be
dropped. Deleting an `ipt~` never blocks Pd: the model is unloaded in the
background, and quitting Pd waits for it. -->

## Build

macOS, Apple Silicon, Xcode, CMake ≥ 3.19, Pd vanilla in `/Applications`
(for `m_pd.h`).

```bash
git clone --recurse-submodules <this repository>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j 8
```

The result is `dist/ipt~/`. On the first configure, libipt downloads
libtorch 2.4.1 into `libipt/libs/libtorch` and r8brain into the build tree.
To reuse existing copies:

- libtorch: symlink an existing tree to `libipt/libs/libtorch`;
- r8brain: add `-DFETCHCONTENT_SOURCE_DIR_R8BRAIN_SRC=/path/to/r8brain-free-src`.

`m_pd.h` is taken from the newest `/Applications/Pd-*.app`; pass
`-DPD_INCLUDE_DIR=/path/to/pd/src` to use another one.

Headless test (needs Pd in `/Applications`):

```bash
tests/run_tests.sh
```

How long deleting an `ipt~` blocks Pd while a real model loads (should be
well under 1 ms):

```bash
tests/free_latency.sh /path/to/flute_ircam.ts
```

Keep to the Pd 0.55 API: plugdata 0.9.3 embeds Pd 0.55.2, and every Pd
symbol the external uses must exist there.

## Credits and license

ipt for Pd is based on the original work by Nicolas Brochec, Joakim
Borg and Marco Fiorini, IRCAM, RepMus REACH team. CC-BY-NC-4.0, here: [ipt_tilde](https://github.com/DYCI2/ipt_tilde)
