"""Runtime wrapper that converts `autoencoder_training_big4_fmsynth3_0_1.py` into a
`dataset_big5` run.

Architecture:
- Same raw-audio CNN autoencoder as the big4 script
- 1D CNN encoder over the waveform
- latent bottleneck
- MLP decoder that reconstructs the waveform

Data flow:
- Input: `dataset_big5/parameters.csv` plus `sample_*.wav`
- Output: trained autoencoder and encoder `.keras` files, latent arrays,
  preprocessing bundle, history plots, and `results.json`
"""

from __future__ import annotations

from pathlib import Path


def _load_and_rewrite_big4_source() -> str:
    source_path = Path(__file__).with_name("autoencoder_training_big4_fmsynth3_0_1.py")
    source = source_path.read_text(encoding="utf-8")

    replacements = [
        ("dataset_big4", "dataset_big5"),
        ("autoencoder_training_big4_fmsynth3_0_1", "autoencoder_training_big5_fmsynth3_0_1"),
        ("encoder_autoencoder_training_big4_fmsynth3_0_1", "encoder_autoencoder_training_big5_fmsynth3_0_1"),
        ("cnn_autoencoder_big4_0_1", "cnn_autoencoder_big5_0_1"),
        ("cnn_encoder_big4_0_1", "cnn_encoder_big5_0_1"),
        ("x_train_big4.npy", "x_train_big5.npy"),
        ("x_test_big4.npy", "x_test_big5.npy"),
        ("latent_big4.npy", "latent_big5.npy"),
    ]

    for old, new in replacements:
        source = source.replace(old, new)

    return source


exec(compile(_load_and_rewrite_big4_source(), __file__, "exec"))
