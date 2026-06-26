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

The NSynth transfer circuit was then executed with the saved checkpoint:

| Stage | Count | FFT distance mean | STFT distance mean | Log-mel distance mean, raw | Log-mel distance mean, normalized |
|---|---:|---:|---:|---:|---:|
| model output | `4096` | `26576.70` | `6956.43` | `132.05` | `0.00819` |
| pitch-matched sine baseline | `4096` | `33355.78` | `6526.78` | `143.91` | `0.00892` |
| fixed `440 Hz` baseline | `4096` | `33508.81` | `7290.35` | `184.87` | `0.01146` |

These results indicate that the final model outperforms both baselines on normalized log-mel distance, and also improves FFT distance relative to the pitch-matched sine baseline. The pitch-matched baseline still remains stronger on STFT distance, which suggests that the resynthesis is not yet uniformly better across all spectral metrics.

The harmonic baseline introduced in the `v2` evaluation is materially more challenging. It narrows the FFT gap and remains stronger than the model on STFT distance, while still losing to the model on normalized log-mel distance. This suggests that the current inverse model captures the corpus well, but the resynthesis chain still needs better control over fine spectral envelope.

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

## 6. Conclusion

In retrospect, the repository built a complete experimental chain, from synthetic corpus generation to resynthesized-audio evaluation. The `big4` line consolidated the infrastructure, and the `big6` line increased scale and operational efficiency, but the performance ceiling on the most ambiguous targets remained limited.

The main continuation hypothesis is that a substantial part of the difficulty lies not only in model capacity, but in the **observability** of the parameters from normalized audio. For this reason, the simplified formulation of `dataset_big7` is methodologically important: it shifts attention toward parameters with stronger acoustic support, making the inverse problem better conditioned and, in principle, more learnable.

The more recent `dataset_big9` moves this idea one step further by preserving fast iteration while restoring the structural differences that separate the algorithms in the synthesizer itself. It therefore serves as the next diagnostic benchmark for testing whether the weak scores observed in `big8` were caused primarily by a flawed target formulation rather than by insufficient model depth alone.

The `dataset_big11` corpus carries that corrected formulation into a larger and longer setting, and the `dataset_big12` extends the same hypothesis to a much denser sampling of the parameter space. In methodological terms, these corpora do not introduce a new inverse problem; they test whether the already corrected problem becomes more robust when coverage is expanded.

In academic terms, the central result of this experimental history is the demonstration that inverse FM synthesis, when formulated as multi-task regression over raw audio, requires not only more data and more depth, but also a careful selection of targets whose recovery is genuinely identifiable in the signal.
