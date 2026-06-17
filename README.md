# FMSYNTH

Manual FM synthesis study and experiment repository for predicting and reconstructing synthesizer parameters.

This repository aims to provide a reproducible path to:
- generate controlled FM datasets,
- train supervised models from raw audio or latent embeddings,
- evaluate prediction quality,
- resynthesize audio from predicted parameters.

## FM Synthesizers

| File | Summary |
|---|---|
| [`fm_synth.py`](/home/sergio/@pessoal/fmsynth/fm_synth.py) | First version of the FM engine. It implements a simple synthesizer with ADSR envelopes and served as the basis for the initial experiments. The main sample rate is `22050 Hz`. |
| [`fm_synth2.py`](/home/sergio/@pessoal/fmsynth/fm_synth2.py) | Second version, with higher-rate rendering and decimation to reduce aliasing. It uses `48000 Hz` internally and outputs `16000 Hz`. |
| [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py) | Most complete version, used by the `big3`, `big4`, and `big5` datasets. It supports 6 operators, per-operator envelopes, detune, feedback, LFO, key scaling, and anti-aliasing protection. It renders at `96000 Hz` and usually outputs `16000 Hz`. |

## Datasets

| Dataset | Generator | Main Characteristics |
|---|---|---|
| `dataset_big` | [`generate_dataset.py`](/home/sergio/@pessoal/fmsynth/generate_dataset.py) | First FM corpus in the repository. It uses [`fm_synth.py`](/home/sergio/@pessoal/fmsynth/fm_synth.py), has `1 s` duration, about `20000` samples, and `sample_*.wav` files with metadata in `parameters.json`. |
| `dataset_big2` | [`generate_dataset2.py`](/home/sergio/@pessoal/fmsynth/generate_dataset2.py) | Intermediate corpus using [`fm_synth2.py`](/home/sergio/@pessoal/fmsynth/fm_synth2.py), with `4 s` duration, about `5000` samples, and `16000 Hz` output audio. It was used to test better anti-aliasing and a more musical parameter distribution. |
| `dataset_big3` | [`generate_dataset3.py`](/home/sergio/@pessoal/fmsynth/generate_dataset3.py) | Large corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with `4 s` duration, `50000` samples, a parameter CSV, and audio balanced by algorithms, styles, and envelope curves. |
| `dataset_big4` | [`generate_dataset4.py`](/home/sergio/@pessoal/fmsynth/generate_dataset4.py) | Large and more controlled corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with `4 s` duration, `10000` samples, a parameter CSV, and balancing across algorithms, styles, and curves. It was the main basis for the `big4` models. |
| `dataset_big5` | [`generate_dataset5.py`](/home/sergio/@pessoal/fmsynth/generate_dataset5.py) | Most recent corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with `4 s` duration, `10000` samples, a parameter CSV, and audio generated from a balanced schedule with incremental WAV generation. This is the current dataset for the `big5` experiments. |
| `dataset_big6` | [`generate_dataset6.py`](/home/sergio/@pessoal/fmsynth/generate_dataset6.py) | Sharded corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with `4 s` duration, `10000` samples, `sample_*.wav` files, `audio_big6_manifest.json`, and `audio_big6_shards/shard_*.npy` caches for faster transfer and training. |

## Colab Notebooks

| File | Summary |
|---|---|
| [`goolge_colab/dataset.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/dataset.ipynb) | Colab notebook that defines the FM synth runtime used to generate the dataset inside the Colab session and writes the example output audio to Google Drive. |
| [`goolge_colab/gcp_generate_dataset_big4_fmsynth3_0_1.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/gcp_generate_dataset_big4_fmsynth3_0_1.ipynb) | Colab-ready dataset generator for the big4 corpus, writing `parameters.csv`, `sample_*.wav`, and `meta.json` to `/content/drive/MyDrive/Unifesp/fmsynth/dataset` when that is the active Drive root. It can bootstrap the FM synth symbols from `goolge_colab/dataset.ipynb` if `fm_synth3.py` is not available. Run this before the training notebooks. |
| [`goolge_colab/gcp_model_training_big4_fmsynth3_0_2.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/gcp_model_training_big4_fmsynth3_0_2.ipynb) | Colab-ready training notebook for the Drive `dataset` folder, with a deeper residual 1D CNN regressor, `tf.data` input pipeline, mixed precision, and Google Drive I/O rooted under `/content/drive/MyDrive/Unifesp/fmsynth` when available. |
| [`goolge_colab/gcp_model_training_big4_fmsynth3_0_3.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/gcp_model_training_big4_fmsynth3_0_3.ipynb) | Aggressive Colab-ready multi-head regressor for the Drive `dataset` folder, with grouped numeric heads, separate categorical heads, a deeper residual CNN backbone, multi-scale pooled features, and Google Drive I/O rooted under `/content/drive/MyDrive/Unifesp/fmsynth` when available. |
| [`goolge_colab/gcp_model_training_big4_fmsynth3_0_4.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/gcp_model_training_big4_fmsynth3_0_4.ipynb) | Most aggressive Colab-ready hybrid CNN+attention regressor so far, with delayed downsampling, wider residual stages, temporal self-attention blocks, heavier shared heads, and a larger default batch size to push VRAM usage harder. |
| [`goolge_colab/gcp_model_training_big4_fmsynth3_0_5.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/gcp_model_training_big4_fmsynth3_0_5.ipynb) | Heavier Colab-ready hybrid CNN+attention regressor, with wider early feature maps, later downsampling, deeper residual stages, stronger attention blocks, and a much larger default batch size to pressure GPU memory harder. |
| [`goolge_colab/gcp_model_training_big4_fmsynth3_0_6.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/gcp_model_training_big4_fmsynth3_0_6.ipynb) | Heavy Colab-ready hybrid CNN+token-attention regressor, with explicit temporal compression before attention, wider layers, deeper residual stages, and large batches that should stay inside Colab GPU limits more safely. |
| [`goolge_colab/gcp_model_training_big4_fmsynth3_0_7.ipynb`](/home/sergio/@pessoal/fmsynth/goolge_colab/gcp_model_training_big4_fmsynth3_0_7.ipynb) | Cached Colab-ready hybrid CNN+token-attention regressor, with the dataset copied to local disk once and audio preloaded into `.npy` arrays so the training loop no longer decodes WAV files on the fly. |

## Experimented Models

| Script | Summary |
|---|---|
| [`model_training_big3_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big3_fmsynth3_0_1.py) | Initial CNN regressor trained directly on raw waveform. It was one of the first supervised FM-parameter prediction experiments. |
| [`model_training_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big4_fmsynth3_0_1.py) | Refined CNN regressor for `dataset_big4`, with train/validation/test split, output scaler, and more complete artifacts. |
| [`model_training_big4_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/model_training_big4_fmsynth3_0_2.py) | Regression experiment with projection and regression submodels, designed to inspect the internal representation more clearly. |
| [`model_training_big4_fmsynth3_0_3.py`](/home/sergio/@pessoal/fmsynth/model_training_big4_fmsynth3_0_3.py) | CNN with aggressive pooling and flatten features for multi-output regression. |
| [`model_training_big4_fmsynth3_0_4.py`](/home/sergio/@pessoal/fmsynth/model_training_big4_fmsynth3_0_4.py) | Multi-head CNN for numerical parameters, with better output organization and logs. |
| [`model_training_big4_fmsynth3_0_5.py`](/home/sergio/@pessoal/fmsynth/model_training_big4_fmsynth3_0_5.py) | Deeper variant of the multi-head CNN architecture, adjusting pooling and shared trunk capacity. |
| [`model_training_big4_fmsynth3_0_6.py`](/home/sergio/@pessoal/fmsynth/model_training_big4_fmsynth3_0_6.py) | More mature CNN regressor variant, with dilated blocks, parameter-type grouped heads, and per-head loss weights. |
| [`model_training_big6_fmsynth3_0_3.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_3.py) | Hybrid `dataset_big6` regressor with a waveform tower plus a log-mel spectrogram tower, wider heads, stronger categorical weighting, and a larger default batch to use more VRAM with better prediction quality. |
| [`model_training_big6_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_2.py) | Quality-oriented `dataset_big6` regressor, with a residual multiscale backbone, stratified splits, style-resampled training batches, AdamW, and smoothed categorical losses. |
| [`model_training_big6_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_1.py) | Deeper sharded multi-head CNN regressor for `dataset_big6`, consuming the audio shards via a lazy loader, with smaller kernels, later downsampling, and a larger default batch size tuned to push VRAM usage closer to 10 GB on 16 GB cards. |
| [`tcn_training_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/tcn_training_big4_fmsynth3_0_1.py) | TCN-based regressor focused on longer temporal dependencies in the waveform. |
| [`rnn_training_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/rnn_training_big4_fmsynth3_0_1.py) | BiGRU regressor over frame sequences, used as a comparison against the CNN models. |
| [`autoencoder_training_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/autoencoder_training_big4_fmsynth3_0_1.py) | Raw-waveform autoencoder with a CNN encoder and MLP decoder. It learns a reusable latent representation. |
| [`autoencoder_training_big5_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/autoencoder_training_big5_fmsynth3_0_1.py) | Same autoencoder idea, adapted for `dataset_big5`. |
| [`enconding_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/enconding_big4_fmsynth3_0_1.py) | Encoding script that applies the trained encoder to `dataset_big4` and exports `.npy` latent vectors. |
| [`enconding_big5_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/enconding_big5_fmsynth3_0_1.py) | Equivalent encoding version for `dataset_big5`. |
| [`pos_encoder_model_classification_training_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/pos_encoder_model_classification_training_big4_fmsynth3_0_1.py) | Multi-head classifier that predicts categorical attributes from autoencoder latent embeddings. |
| [`pos_encoder_model_classification_training_big5_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/pos_encoder_model_classification_training_big5_fmsynth3_0_1.py) | Same latent classifier architecture, trained on `dataset_big5_encoded`. |
| [`pos_encoder_model_regression_training_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/pos_encoder_model_regression_training_big4_fmsynth3_0_1.py) | Multi-head regressor over latent vectors, focused on recovering numerical synthesizer parameters. |
| [`pos_encoder_model_regression_training_big5_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/pos_encoder_model_regression_training_big5_fmsynth3_0_1.py) | Same latent regressor idea, adapted for the `dataset_big5` embeddings. |
| [`model_encoder_classification_training_big5_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_encoder_classification_training_big5_fmsynth3_0_1.py) | Joint encoder + classifier trained directly from raw audio, focused only on `algorithm`, with no active decoder. |
| [`model_encoder_classification_training_big5_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/model_encoder_classification_training_big5_fmsynth3_0_2.py) | More stable `algorithm` prediction variant: it receives raw audio, computes log-mel features inside the model, uses a compact 2D CNN, stratified split, `float32`, lower learning rate, and gradient clipping. |
| [`model_pre_encoder_fine_classification_training_big5_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_pre_encoder_fine_classification_training_big5_fmsynth3_0_1.py) | Two-stage variant: autoencoder pretraining followed by classification fine-tuning using the pretrained encoder weights. |

## Other Features

| File(s) | Purpose |
|---|---|
| [`evaluating.py`](/home/sergio/@pessoal/fmsynth/evaluating.py) | Computes distances between original and resynthesized audio, including FFT, STFT, and log-mel metrics. |
| [`grid_search.py`](/home/sergio/@pessoal/fmsynth/grid_search.py) | Hyperparameter-search runner for regression model families. |
| [`grid_search_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/grid_search_big4_fmsynth3_0_1.py) | Grid-search variant specific to the `big4` experiments. |
| [`pred_nsynth.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth.py) | Predicts FM parameters for NSynth files using a trained model. |
| [`pred_nsynth_big4_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth_big4_fmsynth3_0_2.py), [`pred_nsynth_big4_fmsynth3_0_4.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth_big4_fmsynth3_0_4.py) | Prediction variants for the `big4` models. |
| [`resynth_nsynth.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth.py) | Reconstructs audio from predicted parameters and compares it with the originals. |
| [`resynth_nsynth_big4_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth_big4_fmsynth3_0_2.py), [`resynth_nsynth_big4_fmsynth3_0_4.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth_big4_fmsynth3_0_4.py) | Resynthesis variants for the `big4` models. |
| [`generate_dataset.py`](/home/sergio/@pessoal/fmsynth/generate_dataset.py), [`generate_dataset2.py`](/home/sergio/@pessoal/fmsynth/generate_dataset2.py), [`generate_dataset3.py`](/home/sergio/@pessoal/fmsynth/generate_dataset3.py), [`generate_dataset4.py`](/home/sergio/@pessoal/fmsynth/generate_dataset4.py), [`generate_dataset5.py`](/home/sergio/@pessoal/fmsynth/generate_dataset5.py), [`generate_dataset6.py`](/home/sergio/@pessoal/fmsynth/generate_dataset6.py) | Dataset generators used by the experiments. |

## Suggested Reproducibility Flow

1. Generate the desired dataset with the corresponding script.
2. If you are using Colab, run `goolge_colab/gcp_generate_dataset_big4_fmsynth3_0_1.ipynb` first so `dataset` exists in Google Drive under the same `Unifesp/fmsynth` root.
3. Train the autoencoder or the joint encoder/classifier model.
4. Generate embeddings with the encoding script if the experiment uses latents.
5. Train classification or regression models over the latent vectors.
6. Run `pred_nsynth*.py` to predict parameters from test audio.
7. Run `resynth_nsynth*.py` to synthesize audio from the predicted parameters.
8. Use `evaluating.py` to measure the distance between original and resynthesized audio.

## Notes

- The `big4` and `big5` scripts use the same FM synthesis family, but different datasets.
- Each experiment stores artifacts in the folder indicated by `OUTPUT_DIR` inside its script.
- `results.json` files are the best starting point for comparing experiments.
- Repository maintenance, naming, and documentation rules are described in [`AGENTS.md`](/home/sergio/@pessoal/fmsynth/AGENTS.md).
