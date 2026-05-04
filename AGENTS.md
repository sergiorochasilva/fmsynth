# AGENTS.md

This repository contains a full FM-synthesis experiment pipeline: dataset generation, audio synthesis engines, model training, latent encoding, downstream prediction, resynthesis, and evaluation.

## Repository Overview

Main components:

- `fm_synth.py`, `fm_synth2.py`, `fm_synth3.py`: the synthesis engines.
- `generate_dataset*.py`: dataset builders that render WAV files and store metadata.
- `autoencoder_training*.py`: waveform autoencoders that learn latent representations.
- `enconding*.py`: scripts that apply a trained encoder to a dataset and export latent vectors.
- `pos_encoder_model_classification*.py`: classifiers trained on latent vectors.
- `pos_encoder_model_regression*.py`: regressors trained on latent vectors.
- `model_training*.py`, `tcn_training*.py`, `rnn_training*.py`: direct waveform-to-parameter training experiments.
- `model_encoder_classification*.py`, `model_pre_encoder_fine_classification*.py`: raw-audio multi-task encoder/classification experiments.
- `pred_nsynth*.py`: parameter prediction scripts for NSynth-like inputs.
- `resynth_nsynth*.py`: waveform resynthesis from predicted parameters.
- `evaluating.py`: audio-distance evaluation between original and resynthesized files.
- `grid_search*.py`: hyperparameter search utilities.

## Naming Pattern

Follow the existing experiment naming scheme when creating a new variant.

Common pattern:

`<family>_<scope>_<dataset>_<synth_version>_<experiment_version>.py`

Examples:

- `generate_dataset5.py`
- `autoencoder_training_big5_fmsynth3_0_1.py`
- `enconding_big5_fmsynth3_0_1.py`
- `pos_encoder_model_classification_training_big5_fmsynth3_0_1.py`
- `pos_encoder_model_regression_training_big5_fmsynth3_0_1.py`
- `model_encoder_classification_training_big5_fmsynth3_0_1.py`
- `model_pre_encoder_fine_classification_training_big5_fmsynth3_0_1.py`

Interpretation:

- `family` describes the script role: dataset, training, encoding, prediction, resynthesis, evaluation, search.
- `scope` describes the learning stage or model family: `autoencoder`, `pos_encoder`, `model_encoder`, `model_pre_encoder`, etc.
- `dataset` should match the dataset folder and encoded output folder, for example `big4` or `big5`.
- `synth_version` should match the synth code family, currently `fmsynth3`.
- `experiment_version` should increment when the architecture or training recipe changes.

## Dataset Conventions

Dataset scripts usually produce:

- `dataset_bigX/parameters.csv`
- `dataset_bigX/meta.json`
- `dataset_bigX/sample_*.wav`

The dataset number should line up with all downstream scripts:

- `dataset_big4` -> `*_big4_*`
- `dataset_big5` -> `*_big5_*`

Latent encoding scripts usually produce:

- `dataset_bigX_encoded/latent_bigX_fmsynth3_0_1.npy`
- `dataset_bigX_encoded/encoding_metadata_bigX_fmsynth3_0_1.json`

## Model Families

There are three major model families in this repository:

1. Direct waveform-to-parameter models.
2. Autoencoder-based latent models.
3. Raw-audio multitask encoder/classification models.

When adding a new experiment, decide first which family it belongs to, then follow the matching naming pattern.

## How to Add a New Experiment

1. Decide the dataset target first: `big4`, `big5`, or a new `bigN`.
2. Keep the dataset generator, encoder, and downstream model names aligned.
3. Create a new script instead of mutating an older experiment in place, unless the goal is a bug fix.
4. Make the top-of-file comment block explicit:
   - what the model does,
   - what architecture it uses,
   - what goes in,
   - what comes out.
5. Save all artifacts into a dedicated `OUTPUT_DIR`.
6. Make sure `results.json` summarizes the experiment.
7. If the experiment produces new outputs or changes the experiment structure, update `README.md` immediately after the change.

## Commenting Standard

Every generated Python script should begin with a short comment block or docstring that explains:

- the purpose of the script,
- the architecture or algorithm used,
- the input data,
- the output artifacts.

This is especially important for model scripts, because researchers need to reproduce the experiment quickly without reading the full implementation first.

## Documentation Rule

After every repository alteration, update `README.md` so the public documentation stays synchronized with the codebase.

If you add a new experiment, new dataset, or new evaluation path, update the README in the same change.

## Practical Advice

- Prefer additive changes over rewriting old experiments.
- Keep dataset names, latent folders, model filenames, and metadata filenames consistent.
- If you create a `big6`-style experiment, mirror the whole chain:
  - dataset generation,
  - autoencoder or encoder training,
  - encoding,
  - downstream training,
  - prediction,
  - resynthesis,
  - evaluation,
  - README update.

