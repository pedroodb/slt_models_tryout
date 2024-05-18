import math
import torch
from torch import Tensor, nn
from torch.nn.functional import relu, softmax
from transformers import AutoModel # type: ignore


class Conv1DEmbedder(nn.Module):

	def __init__(self, in_channels: int, out_channels: int):
		super(Conv1DEmbedder, self).__init__()
		self.conv1d_1 = nn.Conv1d(in_channels, 128, 1)
		# self.conv1d_2 = nn.Conv1d(512, 256, 1)
		# self.conv1d_3 = nn.Conv1d(256, 128, 1)
		self.conv1d_4 = nn.Conv1d(128, out_channels, 1)

	def forward(self, x: Tensor) -> Tensor:
		'''
			Args:
				x: (N, S, E) where N is the batch size, S is the sequence length and E is the embedding size
			Returns:
				(N, S, E) where E is the embedding size
		'''
		x = x.permute(0, 2, 1)
		x = relu(self.conv1d_1(x))
		# x = relu(self.conv1d_2(x))
		# x = relu(self.conv1d_3(x))
		x = relu(self.conv1d_4(x))
		return x.permute(0, 2, 1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        '''
        Apply positional encoding to the input tensor.
        Args:
            x: (N, S, E)
        Returns:
            Tensor of shape (N, S, E)
        '''
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    '''Code taken from https://pytorch.org/tutorials/beginner/translation_transformer.html'''

    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        '''
            Applies token embedding to the target tensor.
            Args:
                tokens: (N, T)
            Returns:
                Tensor of shape (N, T, E)
        '''
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class KeypointsTransformer(nn.Module):
    '''
        Transformer model for sign language translation. It uses a 1D convolutional layer to embed the keypoints and a transformer to translate the sequence.
        S refers to the source sequence length, T to the target sequence length, N to the batch size, and E is the features number.
    '''

    def __init__(self,
                src_max_len: int,
                tgt_max_len: int,
                in_features: int,
                tgt_vocab_size: int,
                d_model: int = 64,
                num_encoder_layers: int = 6,
                num_decoder_layers: int = 6,
                dropout: float = 0.1,
                use_bert_embeddings = False,
                text_model: str | None = None
                ):
        '''
            Args:
                src_max_len: max length of the source sequence
                tgt_max_len: max length of the target sequence
                in_features: number of features of the input (amount of keypoints * amount of coordinates)
                tgt_vocab_size: size of the target vocabulary
                d_model: number of dimensions of the encoding vectors (default=64). Must be even so the positional encoding works.
                kernel_size: the size of the 1D convolution window (default=5)
                keys_initial_emb_size: the size of the keys embedding (default=128)
        '''
        super(KeypointsTransformer, self).__init__()

        self.batch_norm = nn.BatchNorm1d(in_features)
        self.src_keyp_emb = Conv1DEmbedder(in_channels=in_features, out_channels=d_model)
        self.src_pe = PositionalEncoding(d_model=d_model, max_len=src_max_len)
        self.use_bert_embeddings = use_bert_embeddings
        if self.use_bert_embeddings and text_model is not None:
            self.tgt_tok_emb = AutoModel.from_pretrained(text_model)
            self.tgt_tok_emb.requires_grad_(False)
            self.tgt_tok_conv_emb = Conv1DEmbedder(in_channels=768, out_channels=d_model)
        else:
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.tgt_pe = PositionalEncoding(d_model=d_model, max_len=tgt_max_len)
        self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout, batch_first=True)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def embed_tgt(self, tgt: Tensor, pad_idx: int = 0) -> Tensor:
        if self.use_bert_embeddings:
            tgt_emb = self.tgt_tok_emb(tgt, attention_mask=(tgt == pad_idx)).last_hidden_state
            tgt_emb = self.tgt_tok_conv_emb(tgt_emb)
        else:
            tgt_emb = self.tgt_tok_emb(tgt)
        return tgt_emb


    def forward(self,
                src: Tensor,
                tgt: Tensor,
                tgt_mask: Tensor,
                tgt_padding_mask: Tensor
    ):
        '''
            Forward pass of the model.
            Args:
                src: (N, S, E)
                tgt: (N, T, E)
                tgt_mask: (T, T)
                tgt_padding_mask: (N, T)
            Returns:
                Tensor of shape (N, T, tgt_vocab_size)
        '''
        src = src.permute(0, 2, 1)
        src = self.batch_norm(src)
        src = src.permute(0, 2, 1)

        src_emb = self.src_keyp_emb(src)
        src_emb = self.src_pe(src_emb)
        tgt_emb = self.embed_tgt(tgt)
        tgt_emb = self.tgt_pe(tgt_emb)
        # src_mask and src_key_padding_mask are set to none as we use the whole input at every timestep
        outs = self.transformer(
            src = src_emb,
            tgt = tgt_emb,
            src_mask = None,
            tgt_mask = tgt_mask,
            src_key_padding_mask = None,
            tgt_key_padding_mask = tgt_padding_mask)
        # return softmax(self.generator(outs), dim=0)
        return self.generator(outs)

    def encode(self, src: Tensor):
        src_emb = self.src_pe(self.src_keyp_emb(src))
        return self.transformer.encoder(src_emb, None)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        tgt = tgt.to(torch.int64)
        tgt_emb = self.embed_tgt(tgt)
        tgt_emb = self.tgt_pe(tgt_emb)
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)
