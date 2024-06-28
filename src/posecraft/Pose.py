from typing import Optional, Literal

import torch
from torch import Tensor
import numpy as np


Component = Literal["body", "face", "lhand", "rhand"]

COMPONENT_MAP = (
    ["body" for _ in range(33)]
    + ["face" for _ in range(468)]
    + ["lhand" for _ in range(21)]
    + ["rhand" for _ in range(21)]
)


class Pose:

    @staticmethod
    def load_to_tensor(path: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).float()

    def __init__(self, path: Optional[str] = None, pose: Optional[Tensor] = None):
        if pose is not None:
            self.pose = pose
        elif path is not None:
            self.pose = self.load_to_tensor(path)

    @staticmethod
    def get_components_mask(
        include: list[Component] = [], exclude: list[Component] = []
    ) -> Tensor:
        assert not (
            include and exclude
        ), "Only one of include or exclude should be provided"
        if include:
            return Tensor(
                [True if kp in include else False for kp in COMPONENT_MAP]
            ).bool()
        else:
            return Tensor(
                [False if kp in exclude else True for kp in COMPONENT_MAP]
            ).bool()

    def generate_video(
        self,
        video: Optional[Tensor] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        size=1,
    ):
        if video is not None:
            h, w, _ = video[0].shape
        elif h is not None and w is not None:
            video = torch.zeros((len(self.pose), h, w, 3))
        else:
            raise ValueError("Either use_video or specify h and w")
        for frame_idx in range(len(self.pose)):
            for person in self.pose[frame_idx]:
                for keypoint in person:
                    x = keypoint[0]
                    y = keypoint[1]
                    if not torch.isnan(x) and not torch.isnan(y):
                        x = x * w
                        y = y * h
                        video[
                            frame_idx,
                            int(y) - size : int(y) + size :,
                            int(x) - size : int(x) + size,
                        ] = (
                            255
                            - video[
                                frame_idx,
                                int(y) - size : int(y) + size :,
                                int(x) - size : int(x) + size,
                            ]
                        )
        return video
