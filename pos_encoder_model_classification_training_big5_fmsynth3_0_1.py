"""Runtime wrapper that converts `pos_encoder_model_classification_training_big4_fmsynth3_0_1.py`
into a `dataset_big5` run.

Architecture:
- Same latent-vector classifier as the big4 script
- Dense network over normalized latent vectors
- One softmax head per categorical synthesis parameter
- Shared trunk with per-head capacity for algorithm/style/envelope curves

Data flow:
- Input: `dataset_big5/parameters.csv` and `dataset_big5_encoded/latent_big5_fmsynth3_0_1.npy`
- Output: classifier `.keras`, target preprocessing bundle, test predictions, history, and `results.json`
"""

from __future__ import annotations

from pathlib import Path


def _load_and_rewrite_big4_source() -> str:
    source_path = Path(__file__).with_name(
        "pos_encoder_model_classification_training_big4_fmsynth3_0_1.py"
    )
    source = source_path.read_text(encoding="utf-8")
    return source.replace("big4", "big5")


exec(compile(_load_and_rewrite_big4_source(), __file__, "exec"))
