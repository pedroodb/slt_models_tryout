from typing import Literal
import torch
from torch import Tensor

from WordLevelTokenizer import WordLevelTokenizer
from KeypointsTransformer import KeypointsTransformer
from helpers import generate_square_subsequent_mask

class Translator:
    # TODO: implement batch_greedy_decode and batch_beam_decode

    def __init__(self, model: KeypointsTransformer, tokenizer: WordLevelTokenizer, max_tokens: int, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens

    def translate(self, src, method: Literal["greedy", "beam"], k: int = 5) -> str:
        with torch.no_grad():
            if method == "greedy":
                out = self.greedy_decode(src)
            elif method == "beam":
                out = self.beam_decode(src, k)
            else:
                raise ValueError("Invalid method. Choose between 'greedy' and 'beam'.")
        return self.tokenizer.decode([int(x) for x in out.tolist()], skip_special_tokens=True)

    def greedy_decode(self, src: Tensor) -> Tensor:
        memory = self.model.encode(src)
        ys = torch.ones(1, 1).fill_(self.tokenizer.cls_token_id).to(self.device)
        for i in range(self.max_tokens-1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            prob = self.model.generator(out[:, -1])
            _, next_word_tensor = torch.max(prob, dim=1)
            next_word = next_word_tensor.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == self.tokenizer.sep_token_id:
                break
        return ys.squeeze()

    def beam_decode(self, src: Tensor, k: int) -> Tensor:
        # We use first dimension corresponding to the batch to predict over the k posible beams
        memory = self.model.encode(src).repeat(k, 1, 1)
        ys = torch.ones(k, 1).fill_(self.max_tokens).to(self.device)
        probs = torch.ones(k, 1).to(self.device)
        for i in range(self.max_tokens-1):
            tgt_mask = torch.zeros(ys.size(1), ys.size(1)).to(self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            prob = self.model.generator(out[:, -1])

            next_words_probs, next_words = torch.topk(prob, k=k)

            next_words_joint_probs = (next_words_probs * probs).view(-1)
            next_words_probs = next_words_probs.view(-1)
            next_words = next_words.view(-1)

            sorted_indices = torch.argsort(next_words_joint_probs, descending=True)

            next_words_probs = torch.index_select(next_words_probs, 0, sorted_indices)[:k]
            next_words = torch.index_select(next_words, 0, sorted_indices)[:k]

            probs = next_words_probs.clone()
            ys = torch.cat([ys, next_words.unsqueeze(1)], dim=1)
            if (next_words == self.tokenizer.sep_token_id).all():
                break
        return ys[0].squeeze()