import torch
import pandas as pd
from torch import Tensor
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from torchmetrics.functional.text import bleu_score
import lightning as L

from helpers import create_target_mask
from Translator import Translator
from KeypointsTransformer import KeypointsTransformer
from WordLevelTokenizer import WordLevelTokenizer


class LKeypointsTransformer(L.LightningModule):

    def __init__(
        self,
        model: KeypointsTransformer,
        device: torch.device,
        tokenizer: WordLevelTokenizer,
        translator: Translator,
        lr: float,
        sample_input: tuple[Tensor, Tensor, Tensor, Tensor],
        class_weights: Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = cross_entropy
        self.lr = lr
        self.running_device = device
        self.tokenizer = tokenizer
        self.translator = translator
        self.example_input_array = sample_input
        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=tokenizer.pad_token_id,
        )
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.running_device)

        self.ys_step: list[str] = []
        self.beam_translations_step: list[str] = []
        self.greedy_translations_step: list[str] = []

    def forward(
        self, src: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ):
        return self.model(src, tgt, tgt_mask, tgt_padding_mask)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def run_on_batch(self, batch):
        src, tgt = batch
        # tgt_input and tgt_ouptut are displaced by one position, so tgt_input[i] is the input to the model and tgt_output[i] is the expected output
        tgt_input = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = create_target_mask(
            tgt_input, self.tokenizer.pad_token_id, self.running_device
        )
        logits = self.model(src, tgt_input, tgt_mask, tgt_padding_mask)
        tgt_output = tgt[:, 1:]
        loss = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            tgt_output.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            weight=self.class_weights,
        )
        accuracy = self.accuracy(
            logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)
        )
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.run_on_batch(batch)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.run_on_batch(batch)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True, batch_size=len(batch))
        return loss

    def test_step(self, batch, batch_idx):
        self.model.eval()
        loss, accuracy = self.run_on_batch(batch)
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        ys, preds_greedy, preds_beam = self.get_translations(batch, batch_idx)
        self.ys_step.extend(ys)
        self.greedy_translations_step.extend(preds_greedy)
        self.beam_translations_step.extend(preds_beam)
        return loss, accuracy

    def on_test_epoch_end(self):
        translation_results = [
            (y, trans_greedy, trans_beam)
            + tuple(bleu_score(trans_greedy, [y], n_gram=n).item() for n in range(1, 5))
            + tuple(bleu_score(trans_beam, [y], n_gram=n).item() for n in range(1, 5))
            for y, trans_greedy, trans_beam in zip(
                self.ys_step, self.greedy_translations_step, self.beam_translations_step
            )
        ]
        self.ys_step = []
        self.greedy_translations_step = []
        self.beam_translations_step = []
        translation_results_df = pd.DataFrame(
            translation_results,
            columns=[
                "y",
                "trans_greedy",
                "trans_beam",
                "bleu_1_greedy",
                "bleu_2_greedy",
                "bleu_3_greedy",
                "bleu_4_greedy",
                "bleu_1_beam",
                "bleu_2_beam",
                "bleu_3_beam",
                "bleu_4_beam",
            ],
        )
        self.logger.log_table(key="translation-results", columns=list(translation_results_df.columns), data=translation_results)  # type: ignore
        self.log("bleu_1_greedy", translation_results_df["bleu_1_greedy"].mean())
        self.log("bleu_2_greedy", translation_results_df["bleu_2_greedy"].mean())
        self.log("bleu_3_greedy", translation_results_df["bleu_3_greedy"].mean())
        self.log("bleu_4_greedy", translation_results_df["bleu_4_greedy"].mean())
        self.log("bleu_1_beam", translation_results_df["bleu_1_beam"].mean())
        self.log("bleu_2_beam", translation_results_df["bleu_2_beam"].mean())
        self.log("bleu_3_beam", translation_results_df["bleu_3_beam"].mean())
        self.log("bleu_4_beam", translation_results_df["bleu_4_beam"].mean())
        translation_results_df.to_csv(f"results/translation{self.logger.experiment.name}.csv", index=False)  # type: ignore

    def get_translations(
        self, batch: Tensor, batch_idx: int
    ) -> tuple[list[str], list[str], list[str]]:
        src, tgt = batch
        preds_greedy = self.translator.translate(
            src, self.model, "greedy", self.tokenizer
        )
        preds_beam = self.translator.translate(
            src, self.model, "beam", self.tokenizer, k=32
        )
        ys: list[str] = []
        for i in range(len(src)):
            src_0 = src[i]
            ys.append(
                self.tokenizer.decode(
                    [int(x) for x in tgt[i].tolist()],
                    skip_special_tokens=True,
                )
            )
        return ys, preds_greedy, preds_beam
