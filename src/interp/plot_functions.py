from typing import Any, Literal, Callable, Optional

from torch import Tensor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from hyperparameters import HyperParameters


def plot_encoder_layers(
    attn_output_weights: list[Tensor],
    hp: HyperParameters,
    output_path: Optional[str] = None,
    figsize: tuple[int, int] = (20, 20),
    transparent: bool = False,
):
    fig, axes = plt.subplots(1, hp["NUM_ENCODER_LAYERS"], figsize=figsize, sharey=True)
    for layer, attn_weights in enumerate(attn_output_weights):
        ax = (
            axes[layer] if hp["NUM_ENCODER_LAYERS"] > 1 else axes
        )  # Handle case with only one layer
        sns.heatmap(
            attn_weights,
            ax=ax,
            square=True,
            cbar=False,
        )
        ax.set_title(f"Layer {layer+1}")

    if output_path is not None:
        file_extension = "png" if transparent else "jpg"
        plt.savefig(
            f"{output_path}attn_self_heatmaps_encoder_layers.{file_extension}",
            dpi=150,
            transparent=transparent,
        )
    return fig, ax


def plot_decoder_layers(
    attn_output_weights: dict[int, list[Tensor]],
    hp: HyperParameters,
    translation: list[str],
    mode: Literal["self", "cross"],
    output_path: Optional[str] = None,
    figsize: tuple[int, int] = (20, 20),
    transparent: bool = False,
):
    tgt_length = len(translation) - 1  # from BOS to EOS-1
    fig, axes = plt.subplots(
        tgt_length, hp["NUM_DECODER_LAYERS"], figsize=figsize, sharey=True
    )
    for layer in attn_output_weights:
        for token, attn_weights in enumerate(attn_output_weights[layer]):
            attn_weights = attn_weights[0]
            ax = axes[token, layer]
            tgt_sent = translation[1 : attn_weights.shape[0] + 1]
            sns.heatmap(
                attn_weights,
                ax=ax,
                square=True,
                cbar=False,
            )  # vmin=0.0, vmax=1.0)
            ax.set_aspect("auto")
            ax.set_yticklabels(tgt_sent, rotation=0)
            if mode == "self":
                ax.set_xticklabels(translation[0 : attn_weights.shape[0]], rotation=90)
            ax.set_title(f"Layer {layer}") if token == 0 else None
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    if output_path:
        file_extension = "png" if transparent else "jpg"
        plt.savefig(
            f"{output_path}/attn_{mode}_heatmaps_decoder_layers.{file_extension}",
            dpi=150,
            transparent=transparent,
        )
    return fig, axes


def reorganize_list(input_list, N):
    grouped_list = []
    for i in range(N):
        grouped_list.extend(input_list[i::N])
    return grouped_list


def plot_intermediate_outputs(
    intermediate_outputs: dict[str, list[Tensor]],
    hp: HyperParameters,
    output_path: str,
    translation: list[str],
    transparent: bool = False,
):
    for k, v in intermediate_outputs.items():
        tgt_length = len(translation) - 1  # from BOS to EOS-1
        fig, axes = plt.subplots(
            tgt_length, hp["NUM_DECODER_LAYERS"], figsize=(20, 20), sharey=True
        )
        attn_output_weights = reorganize_list(v, hp["NUM_DECODER_LAYERS"])
        for layer, attn_weights in enumerate(attn_output_weights):
            i, j = divmod(layer, tgt_length)
            # print(i, j, layer)
            ax = axes[j, i]
            emb_sent = np.arange(hp["D_MODEL"])  # embed_dim
            tgt_sent = translation[1 : attn_weights.shape[0] + 1]
            sns.heatmap(
                attn_weights,
                ax=ax,
                square=True,
                cbar=True,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 8},
            )  # vmin=0.0, vmax=1.0)
            ax.set_aspect("auto")
            ax.set_yticklabels(tgt_sent, rotation=0)
            ax.set_xticklabels(emb_sent, rotation=90)
            ax.set_title(f"Layer {i+1}") if layer % tgt_length == 0 else None

            # Rotate text annotations
            for text in ax.texts:
                text.set_rotation(90)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        if output_path is not None:
            file_extension = "png" if transparent else "jpg"
            plt.savefig(
                f"{output_path}/attn_{k}_heatmaps_decoder_layers.{file_extension}",
                dpi=150,
                transparent=transparent,
            )


def plot_decoder_attn_weights(
    attn_output_weights: list[Tensor],
    hp: HyperParameters,
    output_path: str,
    translation: list[str],
    layer: int,
    norm_func: Callable,
    mode: Literal["heatmap", "lineplot"],
    transparent: bool = False,
    kwargs: dict[str, Any] = {},
) -> np.ndarray:
    attn_output_weights = reorganize_list(attn_output_weights, hp["NUM_DECODER_LAYERS"])
    lower = (len(translation) - 1) * layer
    upper = lower + (len(translation) - 1)
    attn_output_weights = attn_output_weights[lower:upper]
    attn_weights = np.zeros_like(attn_output_weights[-1])
    for i, attn_output_weights in enumerate(attn_output_weights):
        attn_weights[i, :] = norm_func(attn_output_weights[i, :])
    # attn_weights = attn_output_weights[-1]
    # attn_weights = torch.from_numpy(attn_weights)
    # attn_weights = torch.nn.functional.softmax(attn_weights, dim=0).numpy()

    src_sent = np.arange(hp["MAX_FRAMES"])
    tgt_sent = translation[1 : attn_weights.shape[0] + 1]
    if mode == "heatmap":
        sns.heatmap(
            attn_weights,
            xticklabels=src_sent,
            yticklabels=tgt_sent,
            **kwargs,
        )
    elif mode == "lineplot":
        df_attn_weights = pd.DataFrame(attn_weights.T)
        df_attn_weights.columns = tgt_sent
        ax = sns.lineplot(df_attn_weights, dashes=False)
        ax.set_xticks(range(len(src_sent)))
        ax.set_xticklabels(src_sent, rotation=90)

    file_extension = "png" if transparent else "jpg"
    plt.savefig(
        f"{output_path}/attn_weights_{mode}_decoder_layer{layer}.{file_extension}",
        dpi=150,
        bbox_inches="tight",
        transparent=transparent,
    )

    plt.close()

    return attn_weights


def plot_decoder_attn_weights_bars(
    src_pose: Tensor,
    attn_weights: np.ndarray,
    hp: HyperParameters,
    output_path: str,
    translation: list[str],
    layer: int,
    batch_index: int = 0,
) -> FuncAnimation:
    """
    Displays keypoints as a pyplot visualization with an animated bar chart.
    Args:
    - src_pose (torch.Tensor): Tensor of shape (B, F, L) where L is the consecutive x, y values of keypoints.
    - attn_weights (np.ndarray): Array of shape (N, F) where N is the number of words and F is the number of frames.
    """

    def get_update(scatter, keypoints_all_frames, bars, attn_weights, hp):
        def update(frame):
            # Reshape keypoints to have each pair (x, y) in a row
            keypoints = keypoints_all_frames[frame, :].view(
                -1, (3 if hp["USE_3D"] else 2)
            )
            x = keypoints[:, 0]
            y = keypoints[:, 1]
            scatter.set_offsets(torch.stack((x, y), dim=-1))

            bar_heights = attn_weights[:, frame]
            for bar, height in zip(bars, bar_heights):
                bar.set_height(height)

            return scatter, bars

        return update

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # Invert y-axis so origin is at top-left

    translation_filtered = translation[1:]  # Remove BOS
    ax.set_title(" ".join(translation_filtered))

    scatter = ax.scatter([], [], s=10)

    keypoints_all_frames = src_pose[batch_index, :, :]

    ax_bar = fig.add_axes([0.65, 0.63, 0.25, 0.25])
    bars = ax_bar.bar(translation_filtered, attn_weights[:, 0], color="blue")
    ax_bar.set_ylim(0, np.max(attn_weights))
    ax_bar.set_xticks(range(len(translation_filtered)))
    ax_bar.set_xticklabels(translation_filtered, rotation=90)

    func_update = get_update(scatter, keypoints_all_frames, bars, attn_weights, hp)

    num_frames = keypoints_all_frames.shape[0]
    anim = FuncAnimation(fig, func_update, frames=num_frames, interval=50, blit=True)
    anim.save(
        f"{output_path}/attn_weights_sample_bars_decoder_layer{layer}.mp4",
        writer="ffmpeg",
    )

    return anim
