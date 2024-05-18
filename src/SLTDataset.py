import os, json

from typing import Literal, Optional, TypedDict

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from skvideo.io import vread # type: ignore


InputType = Literal['video', 'pose']
OutputType = Literal['text', 'gloss']

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
	def __init__(self, data_dir: str, input_mode: InputType, output_mode: OutputType, split: Optional[Literal['train', 'val', 'test']] = None):
		self.data_dir = data_dir
		self.input_mode = input_mode
		self.output_mode = output_mode
		self.split = split

		try:
			self.metadata: Metadata = json.load(open(os.path.join(data_dir, "metadata.json")))
			print(f"Loaded metadata: {json.dumps(self.metadata, indent=4)}")

			self.annotations = pd.read_csv(os.path.join(data_dir, os.path.join(data_dir, "annotations.csv")))
			if split is not None:
				self.annotations = self.annotations[self.annotations["split"] == split]
			print(f"Loaded annotations at {os.path.join(data_dir, 'annotations.csv')}")
		except FileNotFoundError:
			raise FileNotFoundError("Metadata or annotations not found")

		if self.output_mode == 'text':
			assert 'text' in self.annotations.columns, "Text annotations not found"
		elif self.output_mode == 'gloss':
			assert 'gloss' in self.annotations.columns, "Gloss annotations not found"

		self.missing_files = []
		for id in tqdm(self.annotations["id"], desc="Validating files"):
			path = os.path.join(self.data_dir, 'poses', f'{id}.npy') if self.input_mode == 'pose' else os.path.join(self.data_dir, 'videos', f'{id}.mp4')
			if not os.path.exists(path):
				self.missing_files.append(id)
		if len(self.missing_files) > 0:
			print(f"Missing {len(self.missing_files)} files out of {len(self.annotations)} ({round(100 * len(self.missing_files) / len(self.annotations), 2)}%)" + (f" from split {split}" if split is not None else ""))
		else:
			print("Dataset loaded correctly")

	def __len__(self) -> int:
		return len(self.annotations)

	def __getitem__(self, idx: int):
		id = self.annotations.iloc[idx]["id"]

		if self.input_mode == 'pose':
			file_path = os.path.join(self.data_dir, 'poses', f'{id}.npy')
			x_data = torch.from_numpy(np.load(file_path)).float()
		elif self.input_mode == 'video':
			x_data = torch.from_numpy(vread(os.path.join(self.data_dir, 'videos', f'{id}.mp4')))

		if self.output_mode == 'text':
			y_data = self.annotations.iloc[idx]['text']
		elif self.output_mode == 'gloss':
			y_data = self.annotations.iloc[idx]['gloss']

		return x_data, y_data
