import math
from typing import Optional, Callable
import torch
from torch import Tensor


def norm_positions(pose: Tensor) -> Tensor:
    """
    Normalizes the keypoints of a pose tensor to the difference between the first keypoint and the rest of the keypoints.
    :param pose: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
    """
    frames, people, keypoints, dimensions = pose.shape
    normalized_tensor = torch.zeros_like(pose)
    for frame in range(frames):
        for person in range(people):
            zero_keypoint = pose[frame, person, 0, :]
            for keypoint in range(keypoints):
                normalized_tensor[frame, person, keypoint, :] = (
                    pose[frame, person, keypoint, :] + 0.5 - zero_keypoint
                )
    return normalized_tensor


def get_norm_distances(
    indices: tuple[int, int] = (11, 12), distance_factor: float = 0.2
) -> Callable[[Tensor], Tensor]:
    """
    Returns a function that normalizes the keypoints of a pose tensor to the distance between two keypoints.
    :param indices: A tuple of two integers representing the indices of the keypoints to calculate the distance.
    :param distance_factor: A float representing the factor to multiply the distance between the keypoints.
    """

    def norm_distances(tensor: Tensor) -> Tensor:
        """
        Normalizes the keypoints of a pose tensor to the distance between two keypoints.
        :param tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        frames, people, keypoints, dimensions = tensor.shape
        normalized_tensor = tensor.clone().detach()

        for frame in range(frames):
            for person in range(people):
                x1 = tensor[frame, person, indices[0], 0].item()
                y1 = tensor[frame, person, indices[0], 1].item()
                x2 = tensor[frame, person, indices[1], 0].item()
                y2 = tensor[frame, person, indices[1], 1].item()

                if x1 and y1:
                    factor = distance_factor / math.sqrt(
                        (x2 - x1) ** 2 + (y2 - y1) ** 2
                    )
                else:
                    factor = 1.0

                for keypoint in range(keypoints):
                    keypoint_x = tensor[frame, person, keypoint, 0].item()
                    keypoint_y = tensor[frame, person, keypoint, 1].item()

                    normalized_tensor[frame, person, keypoint, 0] = x1 - factor * (
                        x1 - keypoint_x
                    )
                    normalized_tensor[frame, person, keypoint, 1] = y1 - factor * (
                        y1 - keypoint_y
                    )
        return normalized_tensor

    return norm_distances


def _interpolate(frame_1: Tensor, frame_2: Tensor, factor: float) -> Tensor:
    """
    Interpolates between two frames of keypoints.
    :param frame_1: A list of dictionaries representing the keypoints of the first frame.
    :param frame_2: A list of dictionaries representing the keypoints of the second frame.
    :param factor: A float representing the factor to interpolate between the frames (0.0 to 1.0).
    """
    assert 0.0 <= factor <= 1.0, "Interpolation factor must be between 0.0 and 1.0"
    return frame_1 * factor + frame_2 * (1 - factor)


def _prev_valid(
    tensor: Tensor, frame: int, person: int, keypoint: int
) -> Optional[int]:
    """
    Returns the previous frame where the keypoint was present. If no previous frame is found, returns None.
    :param tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
    :param frame: An integer representing the current frame.
    :param person: An integer representing the current person.
    :param keypoint: An integer representing the current keypoint.
    """
    while torch.isnan(tensor[frame, person, keypoint, 0]):
        frame -= 1
        if frame < 0:
            return None
    return frame


def _next_valid(
    tensor: Tensor, frame: int, person: int, keypoint: int
) -> Optional[int]:
    """
    Returns the next frame where the keypoint was present. If no next frame is found, returns None.
    :param tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
    :param frame: An integer representing the current frame.
    :param person: An integer representing the current person.
    :param keypoint: An integer representing the current keypoint.
    """
    while torch.isnan(tensor[frame, person, keypoint, 0]):
        frame += 1
        if frame >= tensor.shape[0]:
            return None
    return frame


def fill_missing(tensor: Tensor) -> Tensor:
    """
    Fills missing keypoints with the interpolated values between the previous frame where the keypoint was present and the next frame where the keypoint was present.
    If the keypoint is missing in the first frame, it is filled with the next frame where the keypoint is present.
    If the keypoint is missing in the last frame, it is filled with the previous frame where the keypoint is present.
    :param tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
    """
    frames, people, keypoints, dimensions = tensor.shape
    filled_tensor = torch.zeros_like(tensor)
    for frame in range(frames):
        for person in range(people):
            for keypoint in range(keypoints):
                if torch.isnan(tensor[frame, person, keypoint, 0]):
                    prev_frame = _prev_valid(tensor, frame, person, keypoint)
                    next_frame = _next_valid(tensor, frame, person, keypoint)
                    if prev_frame is None:
                        filled_tensor[frame, person, keypoint, :] = tensor[
                            next_frame, person, keypoint, :
                        ]
                    elif next_frame is None:
                        filled_tensor[frame, person, keypoint, :] = tensor[
                            prev_frame, person, keypoint, :
                        ]
                    else:
                        prev_keypoint = tensor[prev_frame, person, keypoint, :]
                        next_keypoint = tensor[next_frame, person, keypoint, :]
                        factor = (frame - prev_frame) / (next_frame - prev_frame)
                        filled_tensor[frame, person, keypoint, :] = _interpolate(
                            prev_keypoint, next_keypoint, factor
                        )
                else:
                    filled_tensor[frame, person, keypoint, :] = tensor[
                        frame, person, keypoint, :
                    ]
    return filled_tensor


def get_norm_speed(
    max_frames: int, use_faces: bool = False
) -> Callable[[Tensor], Tensor]:
    """
    Returns a function that normalizes the keypoints of a pose tensor to the speed of the movement.
    :param max_frames: An integer representing the number of frames to normalize the tensor.
    :param use_faces: A boolean representing whether to use the face keypoints to compute the amount of movement.
    """

    def norm_speed(tensor: Tensor) -> Tensor:
        """
        Normalizes the keypoints of a pose tensor to the speed of the movement.
        :param tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        frames, people, keypoints, dimensions = tensor.shape

        # compute amount of movement per frame
        tensor_no_nans = torch.nan_to_num(tensor, nan=0.0)
        if not use_faces:
            tensor_no_nans = torch.cat(
                [tensor_no_nans[:, :, :32, :], tensor_no_nans[:, :, 500:, :]], dim=2
            )
        movements: list[float] = [
            sum(
                torch.abs(
                    tensor_no_nans[frame_idx, person_idx, :, 0]
                    - tensor_no_nans[frame_idx - 1, person_idx, :, 0]
                )
                .sum()
                .item()
                + torch.abs(
                    tensor_no_nans[frame_idx, person_idx, :, 1]
                    - tensor_no_nans[frame_idx - 1, person_idx, :, 1]
                )
                .sum()
                .item()
                for person_idx in range(people)
            )
            for frame_idx in range(1, frames)
        ]

        # compute the indices to normalize the tensor
        movement_per_frame: float = sum(movements) / (max_frames - 1)
        movement_cum = [0.0] + [sum(movements[: i + 1]) for i in range(len(movements))]
        normalized_indices: list[float] = []
        assert (movement_per_frame * 0 == movement_cum[0]) and (
            round(movement_per_frame * (max_frames - 1), 5)
            == round(movement_cum[-1], 5)
        ), "First and last frame should have 0 and total movement, respectively"
        for i in range(max_frames):
            target_mv = movement_per_frame * i
            j = 0
            while len(normalized_indices) < (i + 1) and j < len(movement_cum):
                if round(movement_cum[j], 5) == round(target_mv, 5):
                    normalized_indices.append(j)
                if round(movement_cum[j], 5) > round(target_mv, 5):
                    normalized_indices.append(
                        j - 1 + (target_mv - movement_cum[j - 1]) / movements[j - 1]
                    )
                j += 1

        # for each index, get the frame from the tensor or interpolate between two frames if the index is not an integer
        normalized_tensor = []
        for idx in normalized_indices:
            if type(idx) == int:
                normalized_tensor.append(tensor[int(idx)])
            else:
                frame_1 = tensor[int(math.floor(idx))]
                frame_2 = tensor[int(math.ceil(idx))]
                factor = idx - int(math.floor(idx))
                interpolated_frame = _interpolate(frame_1, frame_2, factor)
                normalized_tensor.append(interpolated_frame.clone().detach())

        return torch.stack(normalized_tensor)

    return norm_speed


def get_filter_landmarks(
    mask: Tensor, use_3d: bool = False
) -> Callable[[Tensor], Tensor]:
    """
    Returns a function that filters landmarks by mask.
    :param mask: tensor of shape (L,) where L is the number of landmarks.
    :param use_3d: boolean indicating whether to use the 3D coordinates.
    :return: function that filters landmarks by mask.
    """

    def filter_landmarks(datum: Tensor) -> Tensor:
        """
        Filter landmarks by mask.
        :param datum: tensor of shape (S, P, L, D) where S is the number of frames, P is the number of people, L is the number of landmarks, and D is the number of dimensions.
        :return: tensor of same shape as datum but with landmarks filtered by mask.
        """
        # transpose to (L, S, P, D) for filtering
        datum = datum.permute(2, 1, 0, 3)
        datum = datum[mask]
        if not use_3d:
            datum = datum[:, :, :, :2]
        return datum.permute(2, 1, 0, 3)

    return filter_landmarks


def get_pad_truncate(max_len: int) -> Callable[[Tensor], Tensor]:
    """
    Returns a function that pads or truncates the pose tensor to a fixed length.
    :param max_len: An integer representing the maximum length to pad or truncate the tensor.
    """

    def pad_truncate(datum: Tensor) -> Tensor:
        """
        Pads or truncates the pose tensor to a fixed length.
        :param datum: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        frames, people, keypoints, dimensions = datum.shape
        if frames < max_len:
            return torch.cat(
                [datum, torch.zeros(max_len - frames, people, keypoints, dimensions)]
            )
        else:
            return datum[:max_len]

    return pad_truncate


def get_sample_frames_to_fit_max_len(max_len: int) -> Callable[[Tensor], Tensor]:
    """
    Returns a function that randomly samples 1 frame per max_len chunks of frames.
    :param max_len: An integer representing the maximum length to sample the frames.
    """

    def sample_frames_to_fit_max_len(datum: Tensor) -> Tensor:
        """
        Randomly samples 1 frame per max_len chunks of frames.
        :param datum: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
        """
        if datum.size(0) < max_len:
            return torch.cat(
                [
                    datum,
                    torch.zeros(
                        max_len - datum.size(0),
                        datum.size(1),
                        datum.size(2),
                        datum.size(3),
                    ),
                ]
            )
        indices = []
        chunk_size = datum.size(0) // max_len
        for i in range(0, max_len):
            indices.append(torch.randint(i * chunk_size, (i + 1) * chunk_size, [1]))
        return datum[indices, :, :, :]

    return sample_frames_to_fit_max_len


def replace_nans_with_zeros(tensor: Tensor) -> Tensor:
    """
    Replaces NaN values in the tensor with zeros.
    :param tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
    """
    return torch.nan_to_num(tensor, nan=0.0)


def use_frames_diffs(tensor: Tensor) -> Tensor:
    """
    Returns the difference between consecutive frames.
    :param tensor: A tensor of shape (frames, people, keypoints, dimensions) representing a pose.
    """
    return tensor[1:] - tensor[:-1]
