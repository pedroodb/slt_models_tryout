import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from typing import List, Literal, Dict

from hyperparameters import HyperParameters


def reorganize_list(input_list, N):
    grouped_list = []
    for i in range(N):
        grouped_list.extend(input_list[i::N])
    return grouped_list


def plot_encoder_layers(
    attn_output_weights: List[torch.Tensor],
    hp: HyperParameters,
    output_path: str,
    transparent: bool = False,
):
    fig, axes = plt.subplots(1, hp["NUM_ENCODER_LAYERS"], figsize=(10, 5), sharey=True)
    for layer, attn_weights in enumerate(attn_output_weights):
        ax = (
            axes[layer] if hp["NUM_ENCODER_LAYERS"] > 1 else axes
        )  # Handle case with only one layer
        src_sent = np.arange(hp["MAX_FRAMES"])
        sns.heatmap(
            attn_weights,
            ax=ax,
            xticklabels=src_sent,
            yticklabels=src_sent,
            square=True,
            cbar=False,
        )  # vmin=0.0, vmax=1.0)
        ax.set_title(f"Layer {layer+1}")

    file_extension = "png" if transparent else "jpg"
    plt.savefig(
        f"{output_path}/attn_self_heatmaps_encoder_layers.{file_extension}",
        dpi=150,
        transparent=transparent,
    )


def plot_decoder_layers(
    attn_output_weights: List[torch.Tensor],
    hp: HyperParameters,
    output_path: str,
    translation: List[str],
    mode: Literal["self", "cross"],
    transparent: bool = False,
):
    tgt_length = len(translation) - 1  # from BOS to EOS-1
    fig, axes = plt.subplots(
        tgt_length, hp["NUM_DECODER_LAYERS"], figsize=(20, 20), sharey=True
    )
    attn_output_weights = reorganize_list(attn_output_weights, hp["NUM_DECODER_LAYERS"])
    for layer, attn_weights in enumerate(attn_output_weights):
        i, j = divmod(layer, tgt_length)
        # print(i, j, layer)
        ax = axes[j, i]
        src_sent = np.arange(hp["MAX_FRAMES"])
        tgt_sent = translation[1 : attn_weights.shape[0] + 1]
        aux_sent = tgt_sent if mode == "self" else src_sent
        sns.heatmap(
            attn_weights,
            ax=ax,
            xticklabels=aux_sent,
            yticklabels=tgt_sent,
            square=True,
            cbar=False,
        )  # vmin=0.0, vmax=1.0)
        ax.set_aspect("auto")
        ax.set_yticklabels(tgt_sent, rotation=0)
        ax.set_xticklabels(aux_sent, rotation=90)
        ax.set_title(f"Layer {i+1}") if layer % tgt_length == 0 else None

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    file_extension = "png" if transparent else "jpg"
    plt.savefig(
        f"{output_path}/attn_{mode}_heatmaps_decoder_layers.{file_extension}",
        dpi=150,
        transparent=transparent,
    )


def plot_intermediate_outputs(
    intermediate_outputs: Dict[str, List[torch.Tensor]],
    hp: HyperParameters,
    output_path: str,
    translation: List[str],
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
                xticklabels=emb_sent,
                yticklabels=tgt_sent,
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

        file_extension = "png" if transparent else "jpg"
        plt.savefig(
            f"{output_path}/attn_{k}_heatmaps_decoder_layers.{file_extension}",
            dpi=150,
            transparent=transparent,
        )
