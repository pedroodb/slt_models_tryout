import os, json

from typing import Literal, Optional, TypedDict, Callable

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset

# patch numpy types for skvideo: https://github.com/scikit-video/scikit-video/issues/154#issuecomment-1445239790
import numpy

numpy.float = numpy.float64  # type: ignore
numpy.int = numpy.int_  # type: ignore
from skvideo.io import vread, vwrite  # type: ignore


InputType = Literal["video", "pose"]
OutputType = Literal["text", "gloss"]


class Metadata(TypedDict):
    name: str
    id: str
    url: str
    download_link: Optional[str]
    mirror_link: Optional[str]
    input_language: str
    output_language: str
    input_types: list[InputType]
    output_types: list[OutputType]


class SLTDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        input_mode: InputType,
        output_mode: OutputType,
        split: Optional[Literal["train", "val", "test"]] = None,
        transforms: list[Callable[[Tensor], Tensor]] = [],
    ):
        self.data_dir = data_dir
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.split = split
        self.transforms = transforms

        try:
            self.metadata: Metadata = json.load(
                open(os.path.join(data_dir, "metadata.json"))
            )
            print(f"Loaded metadata for dataset: {self.metadata['name']}")
            self.annotations = pd.read_csv(
                os.path.join(data_dir, os.path.join(data_dir, "annotations.csv"))
            )
            if split is not None:
                self.annotations = self.annotations[self.annotations["split"] == split]
            print(
                f"Loaded {split if split is not None else ''} annotations at {os.path.join(data_dir, 'annotations.csv')}"
            )
        except FileNotFoundError:
            raise FileNotFoundError("Metadata or annotations not found")

        if self.output_mode == "text":
            assert "text" in self.annotations.columns, "Text annotations not found"
            self.annotations["text"] = self.annotations["text"].astype(str)
        elif self.output_mode == "gloss":
            assert "gloss" in self.annotations.columns, "Gloss annotations not found"
            self.annotations["gloss"] = self.annotations["gloss"].astype(str)

        self.missing_files = []
        for id in tqdm(self.annotations["id"], desc="Validating files"):
            path = (
                os.path.join(self.data_dir, "poses", f"{id}.npy")
                if self.input_mode == "pose"
                else os.path.join(self.data_dir, "videos", f"{id}.mp4")
            )
            if not os.path.exists(path):
                self.missing_files.append(id)
        if len(self.missing_files) > 0:
            print(
                f"Missing {len(self.missing_files)} files out of {len(self.annotations)} ({round(100 * len(self.missing_files) / len(self.annotations), 2)}%)"
                + (f" from split {split}" if split is not None else "")
            )
        else:
            print("Dataset loaded correctly")
        print()

    def __len__(self) -> int:
        return len(self.annotations)

    def get_pose(self, idx: int) -> Tensor:
        id = self.annotations.iloc[idx]["id"]
        file_path = os.path.join(self.data_dir, "poses", f"{id}.npy")
        return torch.from_numpy(np.load(file_path)).float()

    def get_video(self, idx: int) -> Tensor:
        id = self.annotations.iloc[idx]["id"]
        return torch.from_numpy(
            vread(os.path.join(self.data_dir, "videos", f"{id}.mp4"))
        )

    def get_text(self, idx: int) -> str:
        return self.annotations.iloc[idx]["text"]

    def get_gloss(self, idx: int) -> str:
        return self.annotations.iloc[idx]["gloss"]

    def apply_transforms(self, x_data: Tensor) -> Tensor:
        for transform in self.transforms:
            x_data = transform(x_data)
        return x_data

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        if self.input_mode == "pose":
            x_data = self.get_pose(idx)
        elif self.input_mode == "video":
            x_data = self.get_video(idx)
        if self.output_mode == "text":
            y_data = self.get_text(idx)
        elif self.output_mode == "gloss":
            y_data = self.get_gloss(idx)
        return self.apply_transforms(x_data), y_data

    def visualize_pose(
        self,
        idx: int,
        use_video=True,
        h: Optional[int] = None,
        w: Optional[int] = None,
        size=1,
        transforms: list[Callable[[Tensor], Tensor]] = [],
        out_path: Optional[str] = None,
    ):
        pose = self.get_pose(idx)
        for transform in transforms:
            pose = transform(pose)
        if use_video:
            video = self.get_video(idx)
            h, w, _ = video[0].shape
        elif h is not None and w is not None:
            video = torch.zeros((len(pose), h, w, 3))
        else:
            raise ValueError("Either use_video or specify h and w")
        frames = []
        for frame_idx, original_frame in enumerate(video):
            frame = original_frame.detach().clone()
            for person in pose[frame_idx]:
                for keypoint in person:
                    x = keypoint[0]
                    y = keypoint[1]
                    if not torch.isnan(x) and not torch.isnan(y):
                        x = x * w
                        y = y * h
                        frame[
                            int(y) - size : int(y) + size :,
                            int(x) - size : int(x) + size,
                        ] = (
                            255
                            - frame[
                                int(y) - size : int(y) + size :,
                                int(x) - size : int(x) + size,
                            ]
                        )
                frames.append(frame)
        if out_path is not None:
            vwrite(out_path, frames)
        return frames
