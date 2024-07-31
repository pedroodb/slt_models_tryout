import torch
from typing import TypedDict

from posecraft.Pose import Component
from SLTDataset import InputType, OutputType


class HyperParameters(TypedDict):
    INPUT_MODE: InputType
    OUTPUT_MODE: OutputType
    BATCH_SIZE: int
    MAX_FRAMES: int
    MAX_TOKENS: int
    LANDMARKS_USED: list[Component]
    USE_3D: bool
    TRANSFORMS: list[torch.nn.Module]
    D_MODEL: int
    DROPOUT: float
    NUM_ENCODER_LAYERS: int
    NUM_DECODER_LAYERS: int
    USE_CLASS_WEIGHTS: bool
    LR: float
