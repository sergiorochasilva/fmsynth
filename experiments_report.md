# Consolidated Experimental Report

## Abstract

This document summarizes the experimental trajectory of the `fmsynth` repository, covering dataset evolution, model families, and the auxiliary prediction, resynthesis, and evaluation pipelines. The goal is to present the sequence of hypotheses explored during development, with emphasis on how the problem formulation was gradually refined as empirical evidence revealed parameter ambiguity, redundancy, and limitations in generalization capacity.

The numerical metrics reported here were extracted from the locally preserved training artifacts and should be interpreted as internal evidence for the project. They were not designed for direct comparison across datasets with different formulations, but rather to document the methodological progression of the work.

## 1. Dataset Evolution

The repository shows a clear progression of synthetic corpora, moving from simpler initial formulations to increasingly controlled, scalable, and, ultimately, more identifiable variants.

| Dataset | Duration | Approximate volume | Methodological characteristic |
|---|---:|---:|---|
| `dataset_big` | `1 s` | `~20000` samples | First FM corpus in the project. It was used to validate the generation, storage, and supervised training pipeline. |
| `dataset_big2` | `4 s` | `~5000` samples | Intermediate corpus focused on aliasing reduction and a more musical parameter distribution. |
| `dataset_big3` | `4 s` | `50000` samples | Large corpus based on `fm_synth3`, with balancing across algorithms, styles, and envelope curves. |
| `dataset_big4` | `4 s` | `10000` samples | More controlled corpus, used as the main comparison base in the first mature phase of the experiments. |
| `dataset_big5` | `4 s` | `10000` samples | Incremental evolution of `big4`, with more reproducible generation and support for partial resume workflows. |
| `dataset_big6` | `4 s` | `10000` samples | Sharded corpus stored in `int16`, designed for easier transfer and remote training with lower I/O overhead. |
| `dataset_big7` | `4 s` | `50000` samples | Methodologically simplified variant in which targets with low direct observability in normalized audio were removed. |
| `dataset_big8` | `2 s` | `3072` samples | Compact benchmark corpus with shorter audio, a smaller balanced algorithm set, and a reduced target space designed for faster iteration and easier learning. |
| `dataset_big9` | `2 s` | `3072` samples | Compact benchmark corpus that restores non-zero structural FM indices and adds `ratio_carrier` as an explicit target, with the goal of making the algorithm classes acoustically separable. |
| `dataset_big10` | `2 s` | `4032` samples | Balanced follow-up corpus with the same structural FM indices as `big9`, but with a denser algorithm/ratio schedule and fully fixed nuisance controls to test a cleaner inverse-learning formulation. |
| `dataset_big11` | `4 s` | `10000` samples | Larger benchmark corpus using the same corrected target logic as `big10`, but with longer clips, balanced algorithm/ratio coverage, fixed nuisance controls, and sharded `int16` audio caches for remote transfer. |
| `dataset_big12` | `4 s` | `50000` samples | Large-scale follow-up to `big11`, designed to test whether the corrected formulation benefits further from a substantially larger sharded corpus without reintroducing ambiguous targets. |
| `dataset_big13` | `4 s` | `~50000` samples in the generated corpus; `4096` examples used in the final learning subset | Next inverse-learning cycle, designed to balance learnability and synthesizer control by restoring learnable structural FM indices, continuous `ratio_carrier`, detune, feedback, LFO, key scaling, and ADSR envelope targets while keeping phase, carrier level, and envelope-curve ambiguity fixed. |
| `dataset_big16` | `4 s` | `50000` samples | Hierarchical inverse-learning corpus that reduces nuisance-parameter entropy, restores the structural `algorithm` topology as a learnable signal, and introduces a coarse `algorithm_family` label for staged training before exact classification. |
| `dataset_big18` | `4 s` | `50000` samples | Envelope-focused follow-up to `big17`, using structured ADSR archetypes and a split `env_mod`/`env_car` formulation so the modulator and carrier envelopes can be learned separately, while keeping the rest of the inverse problem compatible with the synthetic-to-NSynth transfer goal. |
| `dataset_big19` | `4 s` | `20000` samples by default | Stage-specific corpus for the first cascade stage, with a reduced ratio schedule, narrowed nuisance controls, non-zero structural FM indices, fixed phase, and the same synthetic engine as the earlier corpora. |
| `dataset_big24` | `4 s` | `50000` samples by default | Follow-up algorithm corpus that restores the full `big12` ratio schedule while keeping fixed nuisance controls and controlled envelope archetypes, aiming to preserve variation without collapsing topology observability. |

From an experimental standpoint, the dataset sequence reveals three major changes:

1. an increase in scale, moving from smaller corpora to a `50000`-example corpus;
2. improved generation control and reproducibility;
3. removal of poorly identifiable targets, such as phase, style, envelope curves, and carrier level, in an attempt to make the inverse problem better conditioned.

The `dataset_big7` should be understood as a reformulation hypothesis rather than a validated conclusion. At the time of writing, it represents a methodological investigation stage, not a consolidated benchmark.

The newer `dataset_big8` introduces a compact learning benchmark rather than another full-scale corpus. Its purpose is to provide a faster and more controllable setting in which model and dataset variations can be tested with lower computational cost.

The subsequent `dataset_big9` preserves the compact benchmark profile but corrects the main observability flaw identified in `big8`: the structural FM indices are no longer collapsed to zero, and `ratio_carrier` becomes a supervised target. This makes the carrier topology itself part of the learnable signal instead of a hidden nuisance variable.

The newer `dataset_big10` keeps the same correction to structural observability, but rebalances the schedule into a larger compact corpus and freezes the remaining nuisance controls more aggressively. In methodological terms, `big10` is a cleaner follow-up intended to test whether the weak learning signal observed in `big9` was caused by target ambiguity or by insufficient separation in the remaining nuisance dimensions.

The subsequent `dataset_big11` extends that corrected formulation to longer `4 s` clips and a larger sharded corpus. Its role is to test whether the same inverse mapping becomes more learnable when the acoustic context is longer and the transfer path is operationally simpler.

The `dataset_big12` keeps the `big11` formulation but increases the sample count dramatically. This makes it a scale test rather than a formulation change: the target space remains the same, while the available coverage of the parameter manifold becomes much denser.

The newer `dataset_big13` changes the inverse problem again. Instead of only testing scale, it restores a broader set of learnable synthesizer controls so that the model can predict not just `algorithm`, `ratio_carrier`, and frequency, but also the FM structural indices, detune, feedback, LFO, key scaling, and amplitude envelopes. The purpose is to increase synthesizer controllability without reintroducing the most ambiguous variables, such as phase and envelope curves.

The subsequent `dataset_big16` narrows the problem in a different way. Rather than reducing control further, it keeps the larger corpus but lowers nuisance entropy and introduces a hierarchical `algorithm_family` target so that the model can first learn coarse topological structure and only then refine the exact algorithm class. This formulation explicitly tests whether staged supervision is more effective than monolithic multitask prediction for the topology component of the inverse problem.

## 2. Model Families

### 2.1 Direct Regression From Raw Audio

The `model_training_big*.py` line represents the most important family in the project. These models receive normalized waveforms and attempt to predict the FM synthesizer parameters in multiple outputs. The formulation combines continuous regression and categorical classification, turning the task into a multi-task learning problem.

The architectural base evolved from a simple convolutional baseline to versions with:

- deeper convolutional blocks;
- more controlled downsampling;
- specialized heads for parameter groups;
- differentiated task weights;
- in `big6`, shard-based audio loading and automatic checkpoint resumption.

### 2.2 Autoencoders and Latent Space Models

The `autoencoder_training*.py` scripts train waveform autoencoders with a convolutional encoder and a decoder for reconstruction. The main goal is not direct parameter prediction, but the learning of a compact latent representation that preserves relevant acoustic information.

Subsequently, the `enconding*.py` scripts export latent vectors, and the `pos_encoder_model_classification_training*.py` and `pos_encoder_model_regression_training*.py` families evaluate how much parameter information is still preserved in those latents.

### 2.3 End-to-End Classification and Fine-Tuning

The `model_encoder_classification_training*.py` and `model_pre_encoder_fine_classification_training*.py` families investigate classifiers trained directly on raw audio. In one stage, the encoder is trained jointly with the supervised task; in another, the encoder is first pretrained as an autoencoder and then fine-tuned for classification.

This line is important because it tests a different hypothesis from direct regression: instead of recovering all parameters, it first attempts to recover only the most critical categorical variable, such as `algorithm`.

### 2.4 Sequential Models and Transfer Pipeline

The `tcn_training*.py` and `rnn_training*.py` families explore temporal dependencies with sequential models such as TCN and BiGRU.

In addition, the `pred_nsynth*.py`, `resynth_nsynth*.py`, and `evaluating.py` scripts form the transfer and perceptual evaluation circuit:

- the model predicts parameters from external audio;
- the predicted parameters are converted back into audio;
- the distance between original and resynthesized audio is measured in multiple spectral domains.

This pipeline closes the loop between supervised prediction and acoustic evaluation.

## 3. Consolidated Results

### 3.1 Direct Regression in `big4` and `big6`

| Experiment | Dataset | Architectural summary | Main result | Synthetic interpretation |
|---|---|---|---|---|
| `model_training_big4_fmsynth3_0_1` | `dataset_big4` | Direct 1D CNN, with a more complete training pipeline than the earliest tests | `RMSE ≈ 78.22`, `MAE ≈ 6.79` | Functional initial baseline, but still with substantial error. |
| `model_training_big4_fmsynth3_0_2` | `dataset_big4` | Regressor with projection and dedicated submodels | `RMSE ≈ 75.95`, `MAE ≈ 6.36` | Small improvement over the previous baseline. |
| `model_training_big4_fmsynth3_0_3` | `dataset_big4` | Aggressive CNN with pooling and multi-output flatten features | `RMSE ≈ 85.48`, `MAE ≈ 6.85`, mean categorical accuracy `≈ 56.85%` | The categorical side improved, but continuous regression worsened. |
| `model_training_big4_fmsynth3_0_4` | `dataset_big4` | Multi-head CNN for numerical parameters | `RMSE ≈ 77.98`, `MAE ≈ 5.64`, mean categorical accuracy `≈ 55.94%` | Best overall balance in the `big4` line. |
| `model_training_big4_fmsynth3_0_6` | `dataset_big4` | Deeper CNN with dilated blocks and grouped heads | `RMSE ≈ 113.30`, `MAE ≈ 7.93`, mean categorical accuracy `≈ 55.90%` | Increased depth did not yield global gains; continuous error degraded. |
| `model_training_big6_fmsynth3_0_2` | `dataset_big6` | Multiscale residual backbone with stratified sampling | `RMSE ≈ 77.31`, `MAE ≈ 5.86`, mean categorical accuracy `≈ 27.90%` | Larger scale helped little on categorical targets, but kept regression stable. |
| `model_training_big6_fmsynth3_0_3` | `dataset_big6` | Waveform tower plus log-mel tower | `RMSE ≈ 75.70`, `MAE ≈ 5.82`, `algorithm ≈ 23.92%` | Slight regression improvement, with a modest gain in `algorithm` prediction. |
| `model_training_big6_fmsynth3_0_6` | `dataset_big6` | Balanced resumable CNN with epoch checkpoints | `RMSE ≈ 70.76`, `MAE ≈ 5.28`, `algorithm ≈ 24.14%` | Best continuous result in the `big6` line, still with meaningful categorical difficulty. |
| `model_training_big10_fmsynth3_0_4` | `dataset_big10` | Differentiable log-mel frontend plus residual 1D CNN | `val_algorithm ≈ 43.31%`, `val_freq_log2_MAE ≈ 0.2605`, `val_ratio_log2_MAE ≈ 0.6041`, `best val_loss ≈ 1.83` | Clear improvement over the raw-waveform `0_3`, indicating that the spectral representation substantially improves learnability. |
| `model_training_big11_fmsynth3_0_1` | `dataset_big11` | Multiresolution log-mel front-end with direct `algorithm` objective and continuous auxiliary regressions | `algorithm_accuracy ≈ 74.92%`, `cross-entropy ≈ 0.5369`, `ratio_MAE ≈ 0.4915`, `freq_MAE ≈ 42.04 Hz` | Strong categorical learning relative to the earlier compact benchmarks, suggesting that the corrected formulation and sharded corpus improve the inverse mapping. |

The scripts `model_training_big6_fmsynth3_0_1`, `model_training_big6_fmsynth3_0_4`, and `model_training_big6_fmsynth3_0_5` belong to the same structural lineage, but the workspace does not currently preserve a numeric summary comparable to the versions listed above.

### 3.2 Autoencoders

| Experiment | Dataset | Architectural summary | Main result | Synthetic interpretation |
|---|---|---|---|---|
| `autoencoder_training_big4_fmsynth3_0_1` | `dataset_big4` | Waveform autoencoder with CNN encoder and dense/hybrid decoder | `test_loss ≈ 2.2925`, `waveform_MAE ≈ 0.2537`, `log-mel_MAE ≈ 2.7825`, `STFT_MAE ≈ 1.1492` | Stable and consistent reconstruction. |
| `autoencoder_training_big5_fmsynth3_0_1` | `dataset_big5` | Same principle, adapted to the `big5` corpus | `test_loss ≈ 2.2985`, `waveform_MAE ≈ 0.2606`, `log-mel_MAE ≈ 2.7869`, `STFT_MAE ≈ 1.1588` | Nearly equivalent to `big4`, suggesting stable reconstruction behavior. |

### 3.3 Latent-Space Models

| Experiment | Dataset | Architectural summary | Main result | Synthetic interpretation |
|---|---|---|---|---|
| `pos_encoder_model_classification_training_big4_fmsynth3_0_1` | `dataset_big4_encoded` | MLP classifier over latent vectors | mean accuracy `≈ 56.51%`, `cross-entropy ≈ 1.1615` | The latent representation preserved strong categorical information. |
| `pos_encoder_model_classification_training_big5_fmsynth3_0_1` | `dataset_big5_encoded` | Same classifier, trained on `big5` | mean accuracy `≈ 27.98%`, `cross-entropy ≈ 1.3675` | Generalization was considerably worse in `big5`. |
| `pos_encoder_model_regression_training_big4_fmsynth3_0_1` | `dataset_big4_encoded` | MLP regressor over latents | `RMSE ≈ 101.66`, `MAE ≈ 7.88`, `freq_MAE ≈ 377.48 Hz` | Numerical recovery was possible, but with high error. |

In parallel, `enconding_big4_fmsynth3_0_1` and `enconding_big5_fmsynth3_0_1` were the operational steps responsible for transforming the autoencoder-reconstructed audio into latent vectors, enabling the models above to operate outside the waveform domain.

### 3.4 End-To-End Classification and Fine-Tuning

| Experiment | Dataset | Architectural summary | Main result | Synthetic interpretation |
|---|---|---|---|---|
| `model_encoder_classification_training_big5_fmsynth3_0_1` | `dataset_big5` | Encoder plus direct classifier on raw audio, focused on `algorithm` | accuracy `≈ 11.72%`, `cross-entropy ≈ 2.0828` | Weak baseline, indicating that the initial configuration extracted little useful information. |
| `model_encoder_classification_training_big5_fmsynth3_0_2` | `dataset_big5` | More stable variant with internal log-mel features, compact 2D CNN, `float32`, lower learning rate, and gradient clipping | accuracy `≈ 28.53%`, `cross-entropy ≈ 1.7756` | Clear gain over the previous version, although the task remains difficult. |

The `model_pre_encoder_fine_classification_training_big5_fmsynth3_0_1` experiment represents a two-stage formulation: autoencoder pretraining followed by classifier fine-tuning. It is conceptually important to the repository history, but no single final numeric consolidation is preserved in the current workspace for direct comparison.

### 3.5 Large-Scale Validation and NSynth Transfer

The `dataset_big12` / `model_training_big12_fmsynth3_0_1` line is the current large-scale transfer test. The corpus keeps the corrected `big11` formulation but increases the sample count to `50000`, while the model preserves the multiresolution log-mel front-end and the direct `algorithm` objective with auxiliary continuous regressions for `ratio_carrier` and `frequencia_base`.

The long run completed with early stopping after epoch `19`, restoring the best weights from the validation optimum and saving both a `.keras` model and checkpoint weights. This makes the experiment a consolidated result rather than only a preliminary checkpoint.

| Experiment | Dataset | Architectural summary | Main result | Synthetic interpretation |
|---|---|---|---|---|
| `model_training_big12_fmsynth3_0_1` | `dataset_big12` | Multiresolution log-mel CNN with direct `algorithm` classification and auxiliary continuous regressions for `ratio_carrier` and `frequencia_base` | `algorithm_accuracy ≈ 96.90%`, `algorithm_crossentropy ≈ 0.1052`, `ratio_MAE ≈ 0.1368`, `freq_MAE ≈ 24.57 Hz`, `freq_log2_MAE ≈ 0.1080` | Strong consolidation of the corrected large-scale formulation; the model learns the target space much more reliably than the earlier compact and waveform-only variants. |
| `model_training_big12_fmsynth3_0_2` | `dataset_big12` | Wider and deeper multiresolution log-mel CNN variant, trained as a scaling hypothesis against `0_1` | `algorithm_accuracy ≈ 96.26%`, `algorithm_crossentropy ≈ 0.1201`, `ratio_MAE ≈ 0.1571`, `freq_MAE ≈ 24.37 Hz`, `freq_log2_MAE ≈ 0.1096` | The larger network converged, but did not surpass `0_1` on the main categorical objective; the gain in frequency error is marginal and does not offset the weaker classification and ratio recovery. |
| `model_training_big19_fmsynth3_0_1` | `dataset_big19` | Direct `algorithm` classifier with a multi-resolution log-mel front-end and residual 1D CNN backbone | `algorithm_accuracy ≈ 66.90%`, `algorithm_crossentropy ≈ 0.8114` | The stage-specific corpus alone does not reproduce the learnability of the `big12` formulation, suggesting that the remaining nuisance entropy still matters. |
| `model_training_big20_fmsynth3_0_1` | `dataset_big19` | Hierarchical family-first `algorithm` classifier with teacher-forced exact heads | `algorithm_family_accuracy ≈ 63.17%`, `algorithm_exact_accuracy_oracle ≈ 60.85%`, `algorithm_exact_accuracy_cascade ≈ 36.34%` | The explicit hierarchy improves the coarse family stage, but it does not fix the exact routing problem. |
| `model_training_big21_fmsynth3_0_1` | `dataset_big19` | Separated hierarchical classifier with independently trained family, `series`, and `parallel` models | `algorithm_family_accuracy ≈ 59.51%`, `algorithm_exact_accuracy_oracle ≈ 59.88%`, `algorithm_exact_accuracy_cascade ≈ 34.39%` | Removing shared exact heads did not materially improve the outcome, which indicates that the main bottleneck is still the corpus / signal structure rather than the shared-head routing itself. |
| `model_training_big22_fmsynth3_0_1` | `dataset_big19` | `big12`-style multitask CNN with auxiliary `ratio_carrier` and `frequencia_base` heads | `algorithm_accuracy ≈ 35.55%`, `ratio_MAE ≈ 0.6172`, `freq_MAE ≈ 59.45 Hz` | The auxiliary heads did not transfer cleanly to `dataset_big19`; the weaker corpus appears to dominate the outcome. |
| `model_training_big23_fmsynth3_0_1` | `dataset_big19` | Direct `algorithm` CNN with a deeper frontend, `delta2`, and focal loss | `algorithm_accuracy ≈ 35.55%` on the evaluated run | The stronger loss / frontend combination did not help on this corpus; in fact, it collapsed to a much weaker classifier than the earlier direct baseline. |
| `model_training_big13_fmsynth3_0_2` | `dataset_big13` | Compact multiresolution log-mel CNN that predicts `algorithm`, `ratio_carrier`, `frequencia_base`, FM structural indices, detune, feedback, LFO, key scaling, and ADSR envelopes | `algorithm_accuracy ≈ 37.32%`, `algorithm_crossentropy ≈ 1.2948`, `ratio_log2_MAE ≈ 0.6473`, `freq_log2_MAE ≈ 0.3137` | Harder than the `big12` formulation because the model must now explain a larger parameter set, but the resulting resynthesis is substantially more controlled and spectrally closer to NSynth than the earlier overly restricted formulations. |
| `model_training_big14_fmsynth3_0_1` | `dataset_big13` | Split-model formulation that trains separate predictors for `algorithm`, carrier pitch/ratio, and the remaining modulatory controls | not yet consolidated | This is the next methodological hypothesis: decompose the inverse problem into smaller submodels to reduce head interference and improve control without forcing one backbone to explain every parameter simultaneously. |
| `model_training_big15_fmsynth3_0_1` | `dataset_big13` | Conditioned split-model formulation with teacher forcing, where the carrier model receives the true `algorithm` during training and the modulator model receives both the true `algorithm` and true `carrier` controls | not yet consolidated | This is the next refinement of the split strategy: the inverse problem is still decomposed, but each downstream model is now explicitly conditioned on the previous stage to reduce ambiguity in the cascade. |
| `model_training_big15_fmsynth3_0_2` | `dataset_big13` | Follow-up to the conditioned split formulation, trained on the full `50000`-sample corpus by default and with a dedicated deeper 1D-CNN branch for `algorithm` before reintroducing the cascade | in progress | This version tests whether the weak `algorithm` head in `0_1` was mainly a capacity issue and a data-coverage issue rather than a fundamental limitation of the split formulation itself. |

### 3.6 Hierarchical Learning on `big16`

The `big16` line revisits the `algorithm` problem with a more explicit topological decomposition. The dataset keeps the `50000`-sample scale but reduces nuisance entropy and introduces a coarse `algorithm_family` target, while the training script first learns the family and then learns the exact algorithm conditioned on that family prediction.

| Experiment | Dataset | Architectural summary | Main result | Synthetic interpretation |
|---|---|---|---|---|
| `model_training_big16_fmsynth3_0_1` | `dataset_big16` | Hierarchical 1D CNN inverse model with a family pretraining stage followed by an exact `algorithm` classifier conditioned on the family prediction | `family val_accuracy ≈ 79.22%`, `family test_accuracy ≈ 78.84%`, `exact val_accuracy ≈ 67.80%`, `exact test_accuracy ≈ 68.65%` (oracle) and `≈ 68.74%` (cascade) | The coarse family task is learned reasonably well, but the exact algorithm still remains only moderately separable; the hierarchy helps, yet it does not fully solve the topology ambiguity. |
| `model_training_big16_fmsynth3_0_2` | `dataset_big16` | Refined hierarchical 1D CNN that keeps family pretraining but replaces the exact 4-way head with family-specific `series` and `parallel` subheads, trained with an explicit family condition and auxiliary family supervision | `family val_accuracy ≈ 74.95%`, `family test_accuracy ≈ 78.95%`, `exact val_accuracy` not separately logged in the final report, `exact test_accuracy ≈ 86.05%` (oracle), `series oracle ≈ 88.18%`, `parallel oracle ≈ 83.92%` | The refactor substantially improves exact algorithm discrimination, which is the right direction for resynthesis, but the gains are still mostly categorical rather than clearly spectral. |
| `model_training_big17_fmsynth3_0_1` | `dataset_big13` | Conditioned split 1D CNN cascade with separate `algorithm`, `pitch`, `timbre`, and `envelope` stages | Full `50000`-sample run: `algorithm_accuracy ≈ 60.07%`, `pitch freq_log2_MAE ≈ 0.1285`, `pitch ratio_log2_MAE ≈ 0.1507`, `timbre detune_MAE ≈ 5.99`, `envelope attack MAE ≈ 0.0048`; NSynth resynthesis on `4096` examples yielded `log-mel normalized ≈ 0.3820` | This is the next control-space decomposition after `big16`: it keeps topology separate, removes `ratio_carrier` and `frequencia_base` from the timbre block, and tests whether separating timbre and envelope improves resynthesis fidelity. The full-corpus run is functionally stable and produces a controllable cascade, but the resynthesized NSynth set still does not beat the pitch-matched baseline on normalized log-mel distance, so the decomposition is helpful but not yet sufficient. |

The `big16` resynthesis transfer remains mixed. The `0_2` hierarchy improves exact algorithm recovery, but the waveform-level evaluation still does not translate that gain into a materially better spectral-distance profile. Compared with `0_1`, the `0_2` output is only marginally different in FFT/STFT, and the normalized log-mel distance is still worse than the pitch-matched baseline. That indicates the bottleneck is no longer only algorithm topology, but also the fine-grained acoustic parameters that shape spectral detail.

The next hypothesis, `big18`, focuses exactly on that acoustic-detail bottleneck. Rather than predict the full ADSR block as a single coupled target, it separates the modulator and carrier envelopes into two dedicated branches. The dataset also becomes more envelope-friendly by sampling structured archetypes instead of a broad unconstrained envelope distribution. The experimental objective is to test whether lowering envelope entanglement improves downstream resynthesis fidelity before revisiting the still-weak algorithm stage.

The first local `big18` test, run on a `4096`-sample training subset for rapid iteration, produced the following final metrics:

| Block | Main result |
|---|---|
| `algorithm` | `accuracy ≈ 34.88%` on the held-out test split |
| `pitch` | `ratio_log2_MAE ≈ 0.4043`, `freq_log2_MAE ≈ 49.62 Hz` |
| `timbre` | remained comparatively weak, with several structural heads still above `1.0` MAE in raw units |
| `env_mod` | `attack MAE ≈ 0.0041`, `decay MAE ≈ 0.0175`, `release MAE ≈ 0.0122`, `sustain MAE ≈ 0.0742` |
| `env_car` | `attack MAE ≈ 0.0057`, `decay MAE ≈ 0.0374`, `release MAE ≈ 0.0222`, `sustain MAE ≈ 0.1084` |

The corresponding NSynth resynthesis evaluation on `4096` files showed that the `big18` cascade improved coarse spectral alignment, but not the full perceptual proxy space:

| Stage | FFT distance mean | STFT distance mean | Log-mel distance mean, raw | Log-mel distance mean, normalized |
|---|---:|---:|---:|---:|
| model output | `15395.79` | `3202.93` | `1141.68` | `0.37974` |
| pitch-matched sine baseline | `33355.78` | `6526.78` | `1103.03` | `0.31818` |
| fixed `440 Hz` baseline | `33508.81` | `7290.35` | `1092.08` | `0.32752` |
| harmonic baseline | `26228.55` | `4783.16` | `1139.15` | `0.34782` |

These values indicate that the split-envelope hypothesis is promising for local envelope prediction, but the overall cascade still does not outperform the pitch-matched baseline on normalized log-mel distance. In practical terms, the envelope split reduced entanglement and improved the envelope heads substantially, yet the remaining `algorithm` and `timbre` bottlenecks are still large enough to dominate the resynthesis quality.

The NSynth transfer circuit was then executed with the full-corpus `big17` checkpoint set:

| Stage | Count | FFT distance mean | STFT distance mean | Log-mel distance mean, raw | Log-mel distance mean, normalized |
|---|---:|---:|---:|---:|---:|
| model output | `4096` | `17217.11` | `3577.62` | `1110.55` | `0.38200` |
| pitch-matched sine baseline | `4096` | `33355.78` | `6526.78` | `1103.03` | `0.31818` |
| fixed `440 Hz` baseline | `4096` | `33508.81` | `7290.35` | `1092.08` | `0.32752` |
| harmonic baseline | `4096` | `26228.55` | `4783.16` | `1139.15` | `0.34782` |

These results indicate that the full-corpus `big17` model improves FFT distance relative to the pitch-matched and fixed `440 Hz` baselines, but it still trails the harmonic baseline on FFT and all three baselines on STFT. The log-mel metric is also still worse than the pitch-matched baseline after normalization, so the gain is concentrated in coarse spectral alignment rather than in the full time-frequency envelope.

The harmonic baseline is materially more challenging. It is still stronger than the model on FFT and STFT, while the model only clearly wins against the simpler `440 Hz` and pitch-matched baselines in FFT. This suggests that the current inverse model captures the coarse corpus structure, but the resynthesis chain still needs better control over the full spectral envelope.

## 4. Auxiliary and Methodological Experiments

In addition to the families with consolidated metrics, the repository contains experiments that were essential for exploring the problem space, even if they did not always leave a directly comparable final report.

- `model_training_big3_fmsynth3_0_1` marked the initial phase of direct supervised regression from raw audio.
- `tcn_training_big4_fmsynth3_0_1` and `rnn_training_big4_fmsynth3_0_1` investigated a TCN and a BiGRU, respectively, as alternatives to CNNs.
- `model_training_big6_fmsynth3_0_1`, `model_training_big6_fmsynth3_0_4`, `model_training_big6_fmsynth3_0_5`, and `model_training_big7_fmsynth3_0_1` compose the line of structural variants that did not leave, in this workspace, a numeric consolidation comparable to the versions with preserved `results.json` files.
- `model_training_big8_fmsynth3_0_1` introduces a compact log-mel benchmark with configurable depth and width, intended for faster architectural search on a simplified target set.
- `model_training_big8_fmsynth3_0_2` refines the compact benchmark by modeling frequency in log space and adding an auxiliary frequency-bin classification head, aiming to improve both precision and learnability.
- `model_training_big9_fmsynth3_0_1` extends the compact benchmark by restoring non-zero structural FM indices and adding `ratio_carrier` as an explicit target, so that the algorithm classes remain acoustically distinct while preserving the fast-iteration design.
- `model_training_big9_fmsynth3_0_2` keeps the same corpus but reformulates `ratio_carrier` as a regression target in `log2` space, reflecting the hypothesis that the categorical version was too ambiguous for stable learning from raw audio.
- `model_training_big9_fmsynth3_0_3` shifts the same corpus into a spectral 1D representation before feeding it to the CNN, under the hypothesis that frequency-domain structure exposes FM topology more clearly than the raw waveform.
- `model_training_big10_fmsynth3_0_1` continues the compact raw-waveform line with balanced algorithm/ratio coverage and frozen nuisance controls, aiming to determine whether a cleaner corpus can finally make the inverse task more learnable.
- `model_training_big10_fmsynth3_0_3` refines that line by removing the discrete `ratio_carrier` classification head and keeping only the regression objectives for `algorithm`, `ratio_log2`, and `frequencia_base`, thereby testing a less redundant multitask formulation.
- `model_training_big10_fmsynth3_0_4` advances the same benchmark by converting the waveform into a differentiable log-mel representation inside the model and then applying a residual 1D CNN; this version materially improves validation performance, which suggests that the feature representation was indeed a major bottleneck.
- `model_training_big11_fmsynth3_0_1` carries the corrected formulation into the larger `dataset_big11` corpus and achieves a noticeably stronger `algorithm` accuracy, reinforcing the value of preserving structural observability while increasing scale.
- `model_training_big12_fmsynth3_0_1` is the large-scale confirmation of the corrected formulation: the same model family is kept, the corpus grows to `50000` examples, and the final checkpointed run shows that denser coverage of the parameter space does improve learnability.
- `pos_encoder_model_regression_training_big5_fmsynth3_0_1` and `model_pre_encoder_fine_classification_training_big5_fmsynth3_0_1` belong, respectively, to the latent-space regression and supervised fine-tuning lines, but also do not have a preserved final consolidation here.
- `pred_nsynth.py`, `pred_nsynth_big4_fmsynth3_0_2`, `pred_nsynth_big4_fmsynth3_0_4`, `resynth_nsynth.py`, `resynth_nsynth_big4_fmsynth3_0_2`, and `resynth_nsynth_big4_fmsynth3_0_4` form the pipeline for external-audio transfer and reconstruction.
- `pred_nsynth_big14_fmsynth3_0_1`, `resynth_nsynth_big14_fmsynth3_0_1`, and `evaluating_nsynth_big14_fmsynth3_0_1` introduce the split-model transfer pipeline, where algorithm, carrier, and modulator submodels are merged only at prediction time.
- `pred_nsynth_big15_fmsynth3_0_1`, `resynth_nsynth_big15_fmsynth3_0_1`, and `evaluating_nsynth_big15_fmsynth3_0_1` extend that idea into a cascaded pipeline, where the algorithm prediction conditions the carrier model and the carrier prediction conditions the modulator model.
- `pred_nsynth_big15_fmsynth3_0_2`, `resynth_nsynth_big15_fmsynth3_0_2`, and `evaluating_nsynth_big15_fmsynth3_0_2` keep the same cascade but evaluate the stronger `0_2` branch under full-corpus training.
- `evaluating.py` consolidates spectral and perceptual comparison between original and resynthesized audio, while `evaluating_nsynth_pitch_baseline.py` adds a more realistic baseline matched to each NSynth pitch.
- `grid_search.py` and `grid_search_big4_fmsynth3_0_1` document the attempt to automate hyperparameter exploration.

These components are important because they show that the project did not rely on a single model, but instead built a complete experimental infrastructure for studying the inverse FM synthesis problem.

## 5. Comparative Reading

The results support several methodological conclusions:

1. **Increasing dataset size alone did not solve the inverse problem.** The move to `dataset_big6` improved continuous regression, but did not lead to robust learning of the most difficult categorical variables.
2. **Model capacity did not improve monotonically with depth.** In more than one configuration, more aggressive versions worsened continuous regression without delivering proportional categorical gains.
3. **Latent representations were more promising for some categorical attributes than direct regression from raw audio.** Especially in the `big4` line, latent-space classifiers performed substantially better than the most basic end-to-end classifiers.
4. **There is strong evidence of intrinsic ambiguity in part of the parameter space.** Variables such as phase, style, envelope curves, and carrier level tend to be poorly observable when audio is normalized. This explains why the problem formulation had to be simplified in `dataset_big7`.
5. **Topological observability matters as much as parameter scale.** The `big8` diagnosis showed that if structural FM indices are collapsed to zero, distinct algorithms can become acoustically equivalent. The `big9` reformulation explicitly addresses that issue by restoring non-zero structural indices and adding `ratio_carrier` to the prediction target. The follow-up `0_2` experiment then tests whether the same target is easier to learn as a continuous quantity rather than as a categorical class, and `0_3` further tests whether the same problem becomes better conditioned when the input is converted from waveform to spectrum.
6. **Reducing redundant supervision can be as important as enlarging the model.** The `big10` follow-up keeps the same observability correction from `big9`, but removes the discrete `ratio_carrier` head in one of its refinements so the network can concentrate on the most stable objectives instead of optimizing a near-ambiguous categorical split. The subsequent `0_4` experiment then moves the same target set to a log-mel residual CNN and produces a clear validation improvement, supporting the hypothesis that the input representation was a dominant factor in the difficulty of the inverse task.
7. **Scaling still matters after the formulation is corrected.** The `big11` result is stronger than the earlier compact benchmark runs, and `big12` is the next test of whether further scale improvements can be obtained without changing the target definition again.
8. **Transfer validation can improve even when training is not yet fully converged.** The current `big12` checkpoint already beats the pitch-matched and `440 Hz` baselines on normalized log-mel distance in NSynth resynthesis, which suggests that the learned inverse mapping is not merely fitting the synthetic corpus but carrying useful acoustic structure into the external-audio setting.
9. **The controllability/learnability trade-off is real.** The `big13` formulation increases the number of predicted controls and weakens the algorithm head compared with `big12`, but it also produces a more structured resynthesis pipeline and avoids the near-degenerate audio that occurred when too many nuisance parameters were fixed too aggressively.
10. **Problem decomposition is the next testable hypothesis.** The `big14` line splits the inverse mapping into smaller models so the algorithm, carrier pitch, and modulator controls stop competing inside one shared backbone. This is a direct response to the observation that a single multitask network can learn the coarse structure while still underperforming on fine control.
11. **Cascaded conditioning is a stronger version of decomposition.** The `big15` line keeps the split models but feeds the true `algorithm` into the carrier model during training and feeds both the true `algorithm` and true `carrier` into the modulator model during training. This is effectively teacher forcing for the inverse-synthesis cascade.
12. **The `algorithm` branch still needs capacity and full coverage.** The `big15_0_2` refinement is a direct response to the weaker categorical score observed in `0_1`: it trains the algorithm classifier on the full `50000`-sample corpus by default and gives it a dedicated deeper 1D-CNN branch before returning to the conditioned cascade.
13. **Hierarchical supervision helps the coarse topology but not the full exact class yet.** The `big16` run shows that a family-level pretraining stage does learn the broad `algorithm` structure, but the exact classifier still stalls in the upper-60% range. That is better than the earlier unstable formulations, but it is not yet a robust inverse mapping.
14. **Reducing nuisance entropy is necessary but insufficient.** Even after removing or constraining parameters that are weakly identifiable from normalized audio, the NSynth transfer still favors the baselines on STFT and does not fully recover the finer spectral envelope. This means the remaining bottleneck is not just dataset entropy, but also the fidelity with which the model preserves fine temporal-spectral detail.
15. **Making the exact topology hierarchical was the right direction, but it is not the full solution.** The `big16_0_2` subhead refactor pushed exact algorithm accuracy higher, which is relevant for resynthesis, but the spectral metrics stayed close to the previous version. That means we still need better control over the non-topological parameters that shape timbre and envelope.
16. **The next hypothesis is a deeper control-space decomposition.** The `big17` cascade keeps `algorithm` separate, moves `ratio_carrier` and `frequencia_base` into a dedicated pitch stage, separates the remaining timbre controls into their own model, and leaves ADSR envelopes to a final stage. The objective is to reduce interference among heterogeneous targets while preserving enough control to make NSynth-style resynthesis meaningful.
17. **The next practical refinement is stage-specific specialization.** The planned `big19` corpus narrows nuisance variability specifically for `algorithm` learning, while keeping the same FM synthesizer family and non-zero structural indices. The methodological claim is that the first cascade stage should not be trained on the same entropy level as the later pitch, timbre, and envelope stages.
18. **The next refinement after `big19` is explicit family-conditioned routing.** The `big20` hypothesis keeps the same stage-specific corpus but replaces the monolithic exact decision with teacher-forced `series` and `parallel` subheads, so the topology problem is split into a coarse family decision followed by a smaller family-specific exact decision.
19. **If shared exact heads still underperform, the exact family models should be trained separately.** The `big21` hypothesis removes the shared exact heads entirely and trains `algorithm_family`, `series`, and `parallel` as independent classifiers, with routing applied only at inference time. The goal is to determine whether the remaining error is caused more by shared-head interference than by the topology itself.

## 6. Conclusion

In retrospect, the repository built a complete experimental chain, from synthetic corpus generation to resynthesized-audio evaluation. The `big4` line consolidated the infrastructure, and the `big6` line increased scale and operational efficiency, but the performance ceiling on the most ambiguous targets remained limited.

The main continuation hypothesis is that a substantial part of the difficulty lies not only in model capacity, but in the **observability** of the parameters from normalized audio. For this reason, the simplified formulation of `dataset_big7` is methodologically important: it shifts attention toward parameters with stronger acoustic support, making the inverse problem better conditioned and, in principle, more learnable.

The more recent `dataset_big9` moves this idea one step further by preserving fast iteration while restoring the structural differences that separate the algorithms in the synthesizer itself. It therefore serves as the next diagnostic benchmark for testing whether the weak scores observed in `big8` were caused primarily by a flawed target formulation rather than by insufficient model depth alone.

The `dataset_big11` corpus carries that corrected formulation into a larger and longer setting, and the `dataset_big12` extends the same hypothesis to a much denser sampling of the parameter space. In methodological terms, these corpora do not introduce a new inverse problem; they test whether the already corrected problem becomes more robust when coverage is expanded.

The `dataset_big13` changes the hypothesis again by relaxing the target space in a controlled way. It keeps the ambiguous variables fixed, but restores enough synthesizer degrees of freedom to test whether a broader inverse mapping can still be learned while maintaining meaningful control over the generated sound. The resulting NSynth evaluation shows a more balanced behavior: the model improves strongly over the trivial baselines on FFT and STFT distances, while still trailing the pitch-matched baseline on normalized log-mel distance.

The `big14` split-model experiment is the next step after that observation. It tests whether decomposing the inverse mapping into separate subproblems can improve the controllability of the final synthesis chain without having to give one backbone responsibility for every parameter at once.

The `big15` refinement makes that cascade explicit. It preserves the split architecture, but conditions each downstream model on the upstream stage so that the model is trained to solve the residual problem after the earlier prediction has already supplied the coarse structure.

The `big16` experiment brings the hierarchy back into the topology stage itself. Instead of only cascading algorithm, carrier, and modulator predictors, it first learns a coarse `algorithm_family` decision and then conditions the exact `algorithm` classifier on that coarse label. The result is a more stable coarse classifier, but the exact branch still stops short of the accuracy needed for a fully reliable resynthesis chain.

The next experimental line, `big17`, changes the decomposition axis. Rather than further splitting topology, it splits acoustic control into successive stages: a pitch model for `ratio_carrier` and `frequencia_base`, a timbre model for the remaining non-envelope controls, and a final envelope model. This is a cleaner formulation of the inverse problem when the main failure mode is interference between semantically different parameter groups.

In the smoke-test-scale validation performed after this refactor, the cascade was operational end-to-end, but the resynthesis metrics did not yet surpass the pitch-matched baseline on normalized log-mel distance. The result is still useful methodologically because it confirms that the new decomposition is implementable, but it also shows that architecture alone is not yet enough to close the spectral gap.

In academic terms, the central result of this experimental history is the demonstration that inverse FM synthesis, when formulated as multi-task regression over raw audio, requires not only more data and more depth, but also a careful selection of targets whose recovery is genuinely identifiable in the signal.

## 7. Current Improvement Plan

The most recent improvement cycle was driven by a staged refinement hypothesis rather than by a simple increase in model depth. The plan can be summarized as follows:

1. **Reduce the entropy of the first cascade stage.** The `algorithm` predictor should not be trained together with all remaining FM controls. Its dataset should vary the synthesizer in a controlled but still acoustically separable way, so that the topology class remains a learnable signal.
2. **Use a stage-specific corpus for `algorithm`.** The `dataset_big19` corpus was created for this purpose: it keeps non-zero structural FM indices, narrows nuisance ranges, fixes phase, and reduces the ratio schedule so the first stage sees cleaner topological evidence.
3. **Train the first stage independently.** A dedicated `algorithm` classifier is then trained on that stage-specific corpus, with the objective of improving the most fragile categorical decision before the rest of the cascade is revisited.
4. **Reintroduce the inverse problem in stages.** Once the `algorithm` stage becomes stable, the next steps are to train the pitch/carrier stage, then the timbre stage, and finally the envelope stage, each with explicit conditioning on the upstream prediction.
5. **Keep NSynth as a transfer test only.** The synthetic corpora remain the training source of truth. NSynth is used only to evaluate generalization and resynthesis quality, not as a training corpus.
6. **Prefer decomposition over blind scaling.** If a joint model remains weak after the stage-specific reformulation, the next hypothesis is to split the problem further, rather than to keep increasing the depth of a single multitask network.
7. **Refine `algorithm` hierarchically if the direct classifier plateaus.** If the stage-specific direct `algorithm` head still saturates below the level needed for stable cascade behavior, the next step is to split the topology problem into a coarse family decision followed by family-specific exact algorithm subheads, with the rest of the synthesizer held as quiet as possible during the first stage.
8. **Revisit the corpus when the `big19` model variants remain weak.** The `big22` and `big23` diagnostics show that simply reusing the `big12`-style model on `dataset_big19` is not enough; the next corpus revision, `dataset_big24`, restores the full `big12` ratio schedule while keeping controlled envelope variation so the later cascade stages can be tested against a less collapsed topology schedule.
9. **Test a compromise corpus instead of over-simplifying further.** `dataset_big24` restores the full `big12` ratio schedule but keeps controlled envelope variation, so the next question is whether the weak `big19` behavior came from collapsing the topology schedule too much rather than from envelope diversity itself.

This plan is intentionally narrower than the earlier monolithic experiments. Its purpose is to reduce target entanglement at the point where the inverse problem is hardest, while preserving enough variability for the model to generalize beyond trivial synthetic cases.
