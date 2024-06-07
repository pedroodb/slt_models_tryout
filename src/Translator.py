from typing import Literal
import torch
from torch import Tensor
from torch.nn.functional import softmax

from WordLevelTokenizer import WordLevelTokenizer
from KeypointsTransformer import KeypointsTransformer
from helpers import generate_square_subsequent_mask


class Translator:
    # TODO: implement batch_greedy_decode and batch_beam_decode

    def __init__(
        self,
        model: KeypointsTransformer,
        tokenizer: WordLevelTokenizer,
        max_tokens: int,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens

    def translate(self, src, method: Literal["greedy", "beam"], k: int = 5) -> str:
        self.model.eval()
        with torch.no_grad():
            if method == "greedy":
                out = self.greedy_decode(src)
            elif method == "beam":
                out = self.beam_decode(src, k)
            else:
                raise ValueError("Invalid method. Choose between 'greedy' and 'beam'.")
        return self.tokenizer.decode(
            [int(x) for x in out.tolist()], skip_special_tokens=True
        )

    def greedy_decode(self, src: Tensor) -> Tensor:
        memory = self.model.encode(src)
        ys = torch.ones(1, 1).fill_(self.tokenizer.cls_token_id).to(self.device)
        for _ in range(self.max_tokens - 1):
            tgt_mask = generate_square_subsequent_mask(
                ys.size(1), self.device
            )  # should work with zeros
            out = self.model.decode(ys, memory, tgt_mask, None)
            prob = self.model.generator(out)[:, -1]
            _, next_word_tensor = torch.max(prob, dim=1)
            next_word = next_word_tensor.item()
            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
            if next_word == self.tokenizer.sep_token_id:
                break
        return ys.squeeze()

    def batched_greedy_decode(self, src: Tensor) -> Tensor:
        self.model.eval()
        memory = self.model.encode(src)
        ys = torch.ones(1, 1).fill_(self.tokenizer.cls_token_id).to(self.device)
        for i in range(self.max_tokens - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), self.device)
            tgt_padding_mask = ys == self.tokenizer.pad_token_id
            out = self.model.decode(ys, memory, tgt_mask, tgt_padding_mask)
            prob = self.model.generator(out)[:, -1]
            _, next_word_tensor = torch.max(prob, dim=1)
            next_word = next_word_tensor.item()
            ys = torch.cat(
                [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
            if next_word == self.tokenizer.sep_token_id:
                break
        return ys.squeeze()

    def beam_decode(self, src: Tensor, k: int) -> Tensor:
        # first dimension corresponding to the batch will be used to predict over the k posible beams
        memory = self.model.encode(src).repeat(k, 1, 1)
        # ys is a Tensor of shape (k, T), initialized with the start token and beams_probs is a Tensor of shape (k) with the probabilities of each beam
        ys = torch.ones(k, 1).fill_(self.tokenizer.cls_token_id).to(self.device)
        beams_probs = torch.ones(k).to(self.device)

        for _ in range(self.max_tokens - 1):
            tgt_mask = torch.zeros(ys.size(1), ys.size(1)).to(self.device)
            next_word_probs = softmax(
                self.model.generator(self.model.decode(ys, memory, tgt_mask, None))[
                    :, -1
                ],
                dim=1,
            )

            # repeat the possible ys and the probabilities k times to later generate k combinations for each beam
            # repeated_ys is a Tensor of shape (k*k, T), repeated_beams_probs is a Tensor of shape (k*k)
            repeated_ys = ys.repeat_interleave(k, dim=0)
            repeated_beams_probs = beams_probs.repeat_interleave(k, dim=0).view(-1)

            # next_words_probs and next_words are Tensors of shape (k,k), the first is per each beam, the second the top k words
            # they are shaped as (k*k) to combine the possible ys with the top k words and their probabilities
            next_words_probs, next_words = torch.topk(next_word_probs, k=k)
            next_words = next_words.view(-1)
            next_words_probs = next_words_probs.view(-1)

            # if any possible y is the end token, change the following token by another end token and keep the original probabilties
            next_words_probs = torch.where(
                repeated_ys[:, -1] == self.tokenizer.sep_token_id,
                torch.ones_like(next_words_probs),
                next_words_probs,
            )
            next_words = torch.where(
                repeated_ys[:, -1] == self.tokenizer.sep_token_id,
                torch.ones_like(next_words) * self.tokenizer.sep_token_id,
                next_words,
            )
            # next_possible_ys is a Tensor of shape (k*k, T+1) with the next possible sequences and next_beam_probs is a Tensor of shape (k*k) with the probabilities for each sequence
            next_possible_ys = torch.cat((repeated_ys, next_words.unsqueeze(1)), dim=1)
            next_beam_probs = next_words_probs * repeated_beams_probs

            # remove repeated sequences from the possible_ys and probs
            l = [str(x) for x in next_possible_ys.tolist()]  # refactor this
            unique_indices = [l.index(x) for x in set(l)]
            next_possible_ys = next_possible_ys[unique_indices]
            next_beam_probs = next_beam_probs[unique_indices]

            # sort the sequences by their probabilities and keep the top k
            sorted_indices = torch.argsort(next_beam_probs, descending=True)
            beams_probs = torch.index_select(next_beam_probs, 0, sorted_indices)[:k]
            ys = torch.index_select(next_possible_ys, 0, sorted_indices)[:k]

            if (ys[:, -1] == self.tokenizer.sep_token_id).all():
                break
        return ys[0].squeeze()
