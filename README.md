# slt_models_tryout

Experiments with pose-based Transformer models for sign language translation, from February 2024 to January 2026.

**Status:** experimental code, not maintained; last updated January 2026.

## Contents

- `src/KeypointsTransformer.py`, `src/LightningKeypointsTransformer.py`: an encoder-decoder Transformer over pose keypoints (1D convolution embedding), trained with PyTorch Lightning and evaluated with BLEU and WER. `src/Translator.py` does greedy and beam search decoding.
- `src/config/`: hyperparameters for GSL, LSA-T and RWTH-PHOENIX-2014T.
- `src/train.ipynb`, `src/results_analysis.ipynb`, `src/dataset_comparison.ipynb`: training and analysis.
- `src/interp/`, `src/get_interp_weights.ipynb`, `src/interp.ipynb`, `src/interp_all.ipynb`: attention interpretability (2024, with Oscar Stanchi).
- `src/expand_db.ipynb`, `src/export_db.py`: paraphrase the RWTH-PHOENIX-2014T training sentences with an LLM (OpenAI API) and export datasets to gzipped pickle files (January 2026).
- Top-level notebooks: conversion of LSA64 keypoints.

Pose transforms come from [posecraft](https://github.com/pedroodb/posecraft) and datasets are loaded with [slt_datasets](https://github.com/pedroodb/slt_datasets). There is no requirements file, and data paths are hardcoded to a local disk.

## Related paper

The attention analysis of *SignAttention: On the Interpretability of Transformer Models for Sign Language Translation* (NeurIPS 2024 workshops, [arXiv:2410.14506](https://arxiv.org/abs/2410.14506)) was developed here. The cleaned-up code for that paper is in [sign_attention](https://github.com/pedroodb/sign_attention).
