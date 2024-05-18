import torch


class WordLevelTokenizer:

    def __init__(self, texts: list[str] | None = None):
        self.token_to_idx: dict[str,int] = {
            "PAD": 0,
            "UNK": 1,
            "BOS": 2,
            "EOS": 3,
        }
        self.idx_to_token: dict[int,str] = {
            0: "PAD",
            1: "UNK",
            2: "BOS",
            3: "EOS",
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.vocab_size = len(self.token_to_idx)
        if texts is not None:
            self.fit(texts)

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for word in text.split():
                if word not in self.token_to_idx:
                    self.token_to_idx[word] = len(self.token_to_idx)
                    self.idx_to_token[len(self.idx_to_token)] = word
                    self.vocab_size += 1

    def tokenize(self, text: str) -> list[int]:
        tokens = []
        for word in text.split():
            if word in self.token_to_idx:
                tokens.append(self.token_to_idx[word])
            else:
                tokens.append(self.unk_token_id)
        return tokens

    def __call__(self, texts: list[str], return_tensors=False, padding='max_length', max_length=None) -> dict[str,torch.Tensor | list[list[int]]]:
        tokenized_texts: list[list[int]] = []
        for text in texts:
            tokenized_text = self.tokenize(text)
            if padding == 'max_length':
                tokenized_text = [self.cls_token_id] + tokenized_text + [self.sep_token_id] + (([self.pad_token_id] * (max_length - (len(tokenized_text) + 2))) if max_length is not None else [])
                if max_length is not None and len(tokenized_text) > max_length:
                    tokenized_text = tokenized_text[:max_length]
            tokenized_texts.append(tokenized_text)
        return {
            "input_ids": (torch.tensor(tokenized_texts) if return_tensors == 'pt' else tokenized_texts),
        }

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.idx_to_token[id] for id in ids]

    def decode(self, ids: list[int], skip_special_tokens=False, clean_up_tokenization_spaces=None) -> str:
        if skip_special_tokens:
            ids = [id for id in ids if id not in [self.cls_token_id, self.sep_token_id, self.pad_token_id]]
        return " ".join(self.convert_ids_to_tokens(ids))
