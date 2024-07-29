import torch
from typing import TypedDict

from posecraft.Pose import Component, Pose
from posecraft.transforms import (
    CenterToKeypoint,
    NormalizeDistances,
    NormalizeFramesSpeed,
    FillMissing,
    FilterLandmarks,
    RandomSampleFrames,
    ReplaceNansWithZeros,
    UseFramesDiffs,
    FlattenKeypoints,
    InterpolateFrames,
)
from SLTDataset import InputType, OutputType

MAX_FRAMES = 200
LANDMARKS_USED: list[Component] = ["body", "lhand", "rhand"]  # , "face"]
USE_3D = False
INPUT_MODE: InputType = "pose"
OUTPUT_MODE: OutputType = "text"
TRANSFORMS: list[torch.nn.Module] = [
    FilterLandmarks(Pose.get_components_mask(LANDMARKS_USED), USE_3D),
    # CenterToKeypoint(),
    # NormalizeDistances(indices=(11, 12), distance_factor=0.2),
    # FillMissing(),
    # InterpolateFrames(MAX_FRAMES),
    # NormalizeSpeed(max_frames=MAX_FRAMES),
    RandomSampleFrames(MAX_FRAMES),
    ReplaceNansWithZeros(),
    # UseFramesDiffs(),
    FlattenKeypoints(),
]

MAX_TOKENS = 50

D_MODEL = 256
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 64
LR = 1e-4
USE_CLASS_WEIGHTS = False


class Hyperparameters(TypedDict):
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


hp: Hyperparameters = {
    # Data hyperparameters
    "INPUT_MODE": INPUT_MODE,
    "OUTPUT_MODE": OUTPUT_MODE,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_FRAMES": MAX_FRAMES,
    "MAX_TOKENS": MAX_TOKENS,
    "LANDMARKS_USED": LANDMARKS_USED,
    "USE_3D": USE_3D,
    "TRANSFORMS": TRANSFORMS,
    # Model hyperparameters
    "D_MODEL": D_MODEL,
    "DROPOUT": DROPOUT,
    "NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS,
    "NUM_DECODER_LAYERS": NUM_DECODER_LAYERS,
    # Training hyperparameters
    "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
    "LR": LR,
}
