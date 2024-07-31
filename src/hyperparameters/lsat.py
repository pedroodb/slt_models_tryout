from .hyperparameters import HyperParameters

from posecraft.Pose import Component, Pose
from posecraft.transforms import (
    FilterLandmarks,
    RandomSampleFrames,
    ReplaceNansWithZeros,
    FlattenKeypoints,
)

LANDMARKS_USED: list[Component] = ["body", "lhand", "rhand"]
USE_3D: bool = False
MAX_FRAMES: int = 80

TRANSFORMS = [
    FilterLandmarks(Pose.get_components_mask(LANDMARKS_USED), USE_3D),
    RandomSampleFrames(MAX_FRAMES),
    ReplaceNansWithZeros(),
    FlattenKeypoints(),
]


hp: HyperParameters = {
    # Data hyperparameters
    "INPUT_MODE": "pose",
    "OUTPUT_MODE": "text",
    "BATCH_SIZE": 64,
    "MAX_FRAMES": MAX_FRAMES,
    "MAX_TOKENS": 40,
    "LANDMARKS_USED": LANDMARKS_USED,
    "USE_3D": USE_3D,
    "TRANSFORMS": TRANSFORMS,
    # Model hyperparameters
    "D_MODEL": 128,
    "DROPOUT": 0.3,
    "NUM_ENCODER_LAYERS": 1,
    "NUM_DECODER_LAYERS": 4,
    # Training hyperparameters
    "USE_CLASS_WEIGHTS": False,
    "LR": 1e-4,
}
