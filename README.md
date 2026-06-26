# FMSYNTH

Manual FM synthesis study and experiment repository for predicting and reconstructing synthesizer parameters.

This repository aims to provide a reproducible path to:
- generate controlled FM datasets,
- train supervised models from raw audio or latent embeddings,
- evaluate prediction quality,
- resynthesize audio from predicted parameters.

## Experiment Report

For a consolidated narrative of the experimental history, dataset evolution, model families, and reported results, see `experiments_report.md`.

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
| `dataset_big7` | [`generate_dataset7.py`](/home/sergio/@pessoal/fmsynth/generate_dataset7.py) | Sharded corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with `4 s` duration, `50000` samples, `sample_*.wav` files, `audio_big7_manifest.json`, and `audio_big7_shards/shard_*.npy` caches. It removes phase, style, envelope-curve, and carrier-level targets to reduce ambiguity. |
| `dataset_big8` | [`generate_dataset8.py`](/home/sergio/@pessoal/fmsynth/generate_dataset8.py) | Compact benchmark corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with shorter audio, a smaller balanced algorithm set, and a reduced target space designed for faster iteration and easier learning. |
| `dataset_big9` | [`generate_dataset9.py`](/home/sergio/@pessoal/fmsynth/generate_dataset9.py) | Compact benchmark corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with the same fast-iteration layout as `big8` but with non-zero structural FM indices and an explicit `ratio_carrier` target to restore algorithmic observability. |
| `dataset_big10` | [`generate_dataset10.py`](/home/sergio/@pessoal/fmsynth/generate_dataset10.py) | Cleaned follow-up corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with balanced algorithm/ratio pairs, fixed detune, fixed feedback, fixed LFO, fixed key scaling, and the same non-zero structural FM indices. |
| `dataset_big11` | [`generate_dataset11.py`](/home/sergio/@pessoal/fmsynth/generate_dataset11.py) | Larger benchmark corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with `4 s` duration, `10000` samples, balanced algorithm/ratio coverage, fixed nuisance controls, and sharded `int16` audio caches for simpler transfer to remote training machines. |
| `dataset_big12` | [`generate_dataset12.py`](/home/sergio/@pessoal/fmsynth/generate_dataset12.py) | Large follow-up corpus using [`fm_synth3.py`](/home/sergio/@pessoal/fmsynth/fm_synth3.py), with `4 s` duration, `50000` samples by default, balanced algorithm/ratio coverage, fixed nuisance controls, and sharded `int16` audio caches for large-scale training and remote transfer. |

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
| [`model_training_big6_fmsynth3_0_6.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_6.py) | Resumable `dataset_big6` regressor, with automatic epoch checkpoints, auto-resume from the latest saved weights, and the same balanced waveform-only backbone as `0_5`. |
| [`model_training_big6_fmsynth3_0_5.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_5.py) | Balanced local `dataset_big6` regressor, with a wider waveform-only backbone, multi-kernel stem, deeper heads, and a higher default batch to push GPU usage closer to the 3.5 GB range. |
| [`model_training_big6_fmsynth3_0_4.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_4.py) | Lightweight local-friendly `dataset_big6` regressor, with a compact waveform-only backbone, small batch defaults, mixed precision, and fallback to the full CSV when `meta.json` is missing. |
| [`model_training_big6_fmsynth3_0_3.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_3.py) | Hybrid `dataset_big6` regressor with a waveform tower plus a log-mel spectrogram tower, wider heads, stronger categorical weighting, and a larger default batch to use more VRAM with better prediction quality. |
| [`model_training_big6_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_2.py) | Quality-oriented `dataset_big6` regressor, with a residual multiscale backbone, stratified splits, style-resampled training batches, AdamW, and smoothed categorical losses. |
| [`model_training_big6_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big6_fmsynth3_0_1.py) | Deeper sharded multi-head CNN regressor for `dataset_big6`, consuming the audio shards via a lazy loader, with smaller kernels, later downsampling, and a larger default batch size tuned to push VRAM usage closer to 10 GB on 16 GB cards. |
| [`model_training_big7_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big7_fmsynth3_0_1.py) | Leaner `dataset_big7` regressor, trained on a simplified target set without phase, style, envelope-curve, or carrier-level outputs, while keeping the balanced waveform backbone and resumable checkpoints. |
| [`model_training_big8_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big8_fmsynth3_0_1.py) | Compact `dataset_big8` multitask model, using a log-mel front-end and a small 2D CNN backbone with configurable depth and width for fast search over easier targets. |
| [`model_training_big8_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/model_training_big8_fmsynth3_0_2.py) | Refined `dataset_big8` multitask model, with `log2(frequencia_base)` regression plus an auxiliary frequency-bin head to improve both precision and learnability. |
| [`model_training_big9_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big9_fmsynth3_0_1.py) | Compact `dataset_big9` multitask model, with a raw-waveform 1D CNN backbone and joint prediction of `algorithm`, discrete `ratio_carrier`, and `log2(frequencia_base)` for the first corrected `big9` formulation. |
| [`model_training_big9_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/model_training_big9_fmsynth3_0_2.py) | Follow-up `dataset_big9` experiment that keeps the 1D CNN backbone but replaces the categorical `ratio_carrier` head with a regression target in `log2` space to reduce target ambiguity and improve learnability. |
| [`model_training_big9_fmsynth3_0_3.py`](/home/sergio/@pessoal/fmsynth/model_training_big9_fmsynth3_0_3.py) | Spectral `dataset_big9` experiment that converts each waveform into a mean log-magnitude spectrum and then applies a 1D residual CNN, with auxiliary ratio classification plus continuous ratio and base-frequency regression heads. |
| [`model_training_big10_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big10_fmsynth3_0_1.py) | Clean `dataset_big10` follow-up using the raw-waveform 1D CNN backbone, with balanced algorithm/ratio coverage and frozen nuisance controls to test whether easier observability finally produces stable learning. |
| [`model_training_big10_fmsynth3_0_3.py`](/home/sergio/@pessoal/fmsynth/model_training_big10_fmsynth3_0_3.py) | Follow-up `dataset_big10` experiment that keeps the raw-waveform 1D CNN backbone but removes the discrete `ratio_carrier` classification head, concentrating the objective on `algorithm`, `ratio_log2`, and `frequencia_base` regression. |
| [`model_training_big10_fmsynth3_0_4.py`](/home/sergio/@pessoal/fmsynth/model_training_big10_fmsynth3_0_4.py) | Spectral `dataset_big10` follow-up that converts the waveform into a differentiable log-mel representation inside the model and then applies a residual 1D CNN, keeping the same multitask heads while testing whether the feature representation is the main learning bottleneck. |
| [`model_training_big10_fmsynth3_0_12.py`](/home/sergio/@pessoal/fmsynth/model_training_big10_fmsynth3_0_12.py) | Algorithm-centered `dataset_big10` variant that combines a deeper multiresolution CNN trunk with auxiliary `ratio_log2` and `freq_log2` regressions, while using stratification by `algorithm` and `ratio_carrier` to preserve paired coverage. |
| [`model_training_big10_fmsynth3_0_13.py`](/home/sergio/@pessoal/fmsynth/model_training_big10_fmsynth3_0_13.py) | Next `dataset_big10` refinement, simplifying the previous multitask formulation into a direct `algorithm` classifier with focal loss and the same continuous auxiliary regressions for `ratio_log2` and `freq_log2`. |
| [`model_training_big10_fmsynth3_0_15.py`](/home/sergio/@pessoal/fmsynth/model_training_big10_fmsynth3_0_15.py) | Hierarchically factorized `dataset_big10` variant that keeps the direct `algorithm` head but adds auxiliary `algorithm` prefix and suffix classifiers, while preserving continuous regressions for `ratio_log2` and `freq_log2`. |
| [`model_training_big10_fmsynth3_0_16.py`](/home/sergio/@pessoal/fmsynth/model_training_big10_fmsynth3_0_16.py) | Data-augmented `dataset_big10` refinement that keeps the `0_13` direct `algorithm` objective and continuous regressions, but adds light waveform shift/gain/noise augmentation during training to reduce overfitting to phase and alignment. |
| [`model_training_big11_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big11_fmsynth3_0_1.py) | `dataset_big11` follow-up to `0_13`, keeping the multiresolution log-mel front-end and direct `algorithm` objective, but retraining on longer `4 s` clips from the larger `10000`-sample corpus. |
| [`model_training_big12_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/model_training_big12_fmsynth3_0_1.py) | `dataset_big12` follow-up to the `big11` line, reusing the multiresolution log-mel front-end and direct `algorithm` objective while targeting the larger `50000`-sample corpus for a more demanding large-scale test. The run now has a consolidated checkpointed result with early stopping and saved weights/model artifacts. |
| [`model_training_big12_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/model_training_big12_fmsynth3_0_2.py) | Wider and deeper `dataset_big12` variant with the same multiresolution log-mel front-end, tested as a scaling hypothesis against `0_1`. |
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
| [`evaluating_nsynth_pitch_baseline.py`](/home/sergio/@pessoal/fmsynth/evaluating_nsynth_pitch_baseline.py) | Evaluates NSynth resynthesis with FFT, STFT, and log-mel distances, comparing the model output against both a fixed `440 Hz` sine and a pitch-matched sine derived from NSynth metadata. |
| [`grid_search.py`](/home/sergio/@pessoal/fmsynth/grid_search.py) | Hyperparameter-search runner for regression model families. |
| [`grid_search_big4_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/grid_search_big4_fmsynth3_0_1.py) | Grid-search variant specific to the `big4` experiments. |
| [`pred_nsynth.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth.py) | Predicts FM parameters for NSynth files using a trained model. |
| [`pred_nsynth_big4_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth_big4_fmsynth3_0_2.py), [`pred_nsynth_big4_fmsynth3_0_4.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth_big4_fmsynth3_0_4.py) | Prediction variants for the `big4` models. |
| [`pred_nsynth_big12_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth_big12_fmsynth3_0_1.py) | NSynth prediction script for the `big12` model, exporting `algorithm`, `ratio_carrier`, and `frequencia_base` for the resynthesis stage. |
| [`pred_nsynth_big12_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/pred_nsynth_big12_fmsynth3_0_2.py) | NSynth prediction script for the `big12` `0_2` model, kept separate so the scaling test can be evaluated independently. |
| [`resynth_nsynth.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth.py) | Reconstructs audio from predicted parameters and compares it with the originals. |
| [`resynth_nsynth_big4_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth_big4_fmsynth3_0_2.py), [`resynth_nsynth_big4_fmsynth3_0_4.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth_big4_fmsynth3_0_4.py) | Resynthesis variants for the `big4` models. |
| [`resynth_nsynth_big12_fmsynth3_0_1.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth_big12_fmsynth3_0_1.py) | NSynth resynthesis script for the `big12` model, using predicted parameters plus corpus priors for the remaining FM structural indices. |
| [`resynth_nsynth_big12_fmsynth3_0_2.py`](/home/sergio/@pessoal/fmsynth/resynth_nsynth_big12_fmsynth3_0_2.py) | NSynth resynthesis script for the `big12` `0_2` model, using algorithm and ratio-conditioned priors for the remaining FM structural indices. |
| [`evaluating_nsynth_pitch_baseline_v2.py`](/home/sergio/@pessoal/fmsynth/evaluating_nsynth_pitch_baseline_v2.py) | Extended NSynth evaluator that adds a harmonic pitch baseline to the fixed `440 Hz` and pitch-matched sine baselines. |
| [`generate_dataset.py`](/home/sergio/@pessoal/fmsynth/generate_dataset.py), [`generate_dataset2.py`](/home/sergio/@pessoal/fmsynth/generate_dataset2.py), [`generate_dataset3.py`](/home/sergio/@pessoal/fmsynth/generate_dataset3.py), [`generate_dataset4.py`](/home/sergio/@pessoal/fmsynth/generate_dataset4.py), [`generate_dataset5.py`](/home/sergio/@pessoal/fmsynth/generate_dataset5.py), [`generate_dataset6.py`](/home/sergio/@pessoal/fmsynth/generate_dataset6.py), [`generate_dataset7.py`](/home/sergio/@pessoal/fmsynth/generate_dataset7.py), [`generate_dataset8.py`](/home/sergio/@pessoal/fmsynth/generate_dataset8.py), [`generate_dataset9.py`](/home/sergio/@pessoal/fmsynth/generate_dataset9.py), [`generate_dataset10.py`](/home/sergio/@pessoal/fmsynth/generate_dataset10.py), [`generate_dataset11.py`](/home/sergio/@pessoal/fmsynth/generate_dataset11.py), [`generate_dataset12.py`](/home/sergio/@pessoal/fmsynth/generate_dataset12.py) | Dataset generators used by the experiments. |

## Suggested Reproducibility Flow

1. Generate the desired dataset with the corresponding script.
2. If you are using Colab, run `goolge_colab/gcp_generate_dataset_big4_fmsynth3_0_1.ipynb` first so `dataset` exists in Google Drive under the same `Unifesp/fmsynth` root.
3. Train the autoencoder or the joint encoder/classifier model.
4. Generate embeddings with the encoding script if the experiment uses latents.
5. Train classification or regression models over the latent vectors.
6. Run `pred_nsynth*.py` to predict parameters from test audio.
7. Run `resynth_nsynth*.py` to synthesize audio from the predicted parameters.
8. Use `evaluating.py` or `evaluating_nsynth_pitch_baseline.py` to measure the distance between original and resynthesized audio, depending on whether the baseline should be fixed at `440 Hz` or matched to the source pitch.

## Notes

- The `big4` and `big5` scripts use the same FM synthesis family, but different datasets.
- Each experiment stores artifacts in the folder indicated by `OUTPUT_DIR` inside its script.
- `results.json` files are the best starting point for comparing experiments.
- Repository maintenance, naming, and documentation rules are described in [`AGENTS.md`](/home/sergio/@pessoal/fmsynth/AGENTS.md).
