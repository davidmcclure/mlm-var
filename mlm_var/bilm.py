

import torch
import numpy as np
import string
import pickle

from tqdm import tqdm
from dataclasses import dataclass
from typing import List
from boltons.iterutils import chunked_iter
from collections import Counter
from itertools import chain, islice

from torch import nn
from torchtext.vocab import Vocab, Vectors
from torch.nn.utils import rnn
from torch.utils.data import random_split
from torch.nn import functional as F

from . import utils, logger


@dataclass
class Line:

    domain: str
    tokens: List[str]
    clf_tokens: List[str]

    @classmethod
    def from_dict(cls, row):
        """Map in a raw dictionary.
        """
        field_names = cls.__dataclass_fields__.keys()
        return cls(**{fn: row.get(fn) for fn in field_names})

    @classmethod
    def read_spark_lines(cls, root):
        """Parse JSON lines, build match objects.
        """
        for row in utils.read_json_gz_lines(root):
            yield cls.from_dict(row)

    def __len__(self):
        return len(self.clf_tokens)


class Corpus:

    @classmethod
    def from_spark_lines(cls, path, skim=None, **kwargs):
        """Read JSON gz lines.
        """
        lines_iter = tqdm(islice(Line.read_spark_lines(path), skim))

        return cls(list(lines_iter), **kwargs)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    def __init__(self, lines, test_frac=0.1):
        self.lines = lines
        self.test_frac = test_frac
        self.set_splits()

    def __len__(self):
        return len(self.lines)

    def token_counts(self):
        """Collect all token -> count.
        """
        logger.info('Gathering token counts.')

        counts = Counter()
        for line in tqdm(self.lines):
            counts.update(line.clf_tokens)

        return counts

    def set_splits(self):
        """Fix train/val/test splits.
        """
        test_size = round(len(self) * self.test_frac)
        train_size = len(self) - (test_size * 2)

        sizes = (train_size, test_size, test_size)
        self.train, self.val, self.test = random_split(self.lines, sizes)


DEVICE = (torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu'))


# TODO: Which pre-trained embeds, if any?
class PretrainedTokenEmbedding(nn.Module):

    def __init__(self, token_counts, vector_file='glove.840B.300d.txt',
        vocab_size=10000, freeze=False):
        """Load pretrained embeddings.
        """
        super().__init__()

        self.vocab = Vocab(
            token_counts,
            vectors=Vectors(vector_file),
            max_size=vocab_size,
        )

        self.embed = nn.Embedding.from_pretrained(self.vocab.vectors, freeze)

        self.out_size = self.embed.weight.shape[1]

    def forward(self, tokens):
        """Map to token embeddings.
        """
        x = [self.vocab.stoi[t] for t in tokens]
        x = torch.LongTensor(x).to(DEVICE)

        return self.embed(x)


# CharCNN params from https://arxiv.org/abs/1508.06615

class CharEmbedding(nn.Embedding):

    def __init__(self, embed_size=15):
        """Set vocab, map s->i.
        """
        self.vocab = (
            string.ascii_letters +
            string.digits +
            string.punctuation
        )

        # <PAD> -> 0, <UNK> -> 1
        self._ctoi = {s: i+2 for i, s in enumerate(self.vocab)}

        super().__init__(len(self.vocab)+2, embed_size)

    def ctoi(self, c):
        return self._ctoi.get(c, 1)

    def chars_to_idxs(self, chars, max_size=20):
        """Map characters to embedding indexes.
        """
        # Truncate super long tokens, to prevent CUDA OOMs.
        chars = chars[:max_size]

        idxs = [self.ctoi(c) for c in chars]

        return torch.LongTensor(idxs).to(DEVICE)

    def forward(self, tokens, min_size=7):
        """Batch-embed token chars.

        Args:
            tokens (list<str>)
        """
        # Map chars -> indexes.
        xs = [self.chars_to_idxs(t) for t in tokens]

        pad_size = max(min_size, max(map(len, xs)))

        # Pad + stack index tensors.
        x = torch.stack([
            F.pad(x, (0, pad_size-len(x)))
            for x in xs
        ])

        return super().forward(x)


class CharCNN(nn.Module):

    def __init__(self, widths=range(1, 7), fpn=25, out_size=512):
        """Conv layers + linear projection.
        """
        super().__init__()

        self.embed = CharEmbedding()

        self.widths = widths

        self.convs = nn.ModuleList([
            nn.Conv2d(1, w*fpn, (w, self.embed.weight.shape[1]))
            for w in self.widths
        ])

        conv_sizes = sum([c.out_channels for c in self.convs])

        self.out = nn.Linear(conv_sizes, out_size)

        self.out_size = out_size

    def forward(self, tokens):
        """Convolve, max pool, linear projection.
        """
        x = self.embed(tokens, max(self.widths))

        # 1x input channel.
        x = x.unsqueeze(1)

        # Convolve, max pool.
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]

        # Cat filter maps.
        x = torch.cat(x, 1)

        return self.out(x)


class TokenEmbedding(nn.Module):

    def __init__(self, token_counts):
        """Initialize token + char embeddings
        """
        super().__init__()

        self.embed_t = PretrainedTokenEmbedding(token_counts)
        self.embed_c = CharCNN()

        self.out_size = self.embed_t.out_size + self.embed_c.out_size

    def forward(self, tokens):
        """Map to token embeddings, cat with character convolutions.
        """
        # Token embeddings.
        xt = self.embed_t(tokens)

        # Char embeddings.
        xc = self.embed_c(tokens)
        x = torch.cat([xt, xc], dim=1)

        return x


LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 1


class TokenLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS):
        """Initialize LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.out_size = self.lstm.hidden_size * 2

    def forward(self, xs):
        """Sort, pack, encode, reorder.

        Args:
            xs (list<Tensor>): Variable-length embedding tensors.

        Returns:
            x (Tensor)
            states (list<Tensor): LSTM states per input.
        """
        sizes = list(map(len, xs))

        # Indexes to sort descending.
        sort_idxs = np.argsort(sizes)[::-1]

        # Indexes to restore original order.
        unsort_idxs = torch.from_numpy(np.argsort(sort_idxs)).to(DEVICE)

        # Sort by size descending.
        xs = [xs[i] for i in sort_idxs]

        # Pad + pack, LSTM.
        x = rnn.pack_sequence(xs)
        _, (hn, _) = self.lstm(x)

        # Cat forward + backward hidden layers.
        x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        x = x[unsort_idxs]

        return x


MAX_VOCAB_SIZE = 10000
CLF_EMBED_SIZE = 512
CLF_HIDDEN_SIZE = 256


class MLM(nn.Module):

    def __init__(self, token_counts, max_vocab_size=MAX_VOCAB_SIZE,
        embed_size=CLF_EMBED_SIZE, hidden_size=CLF_HIDDEN_SIZE):
        """Initialize encoders + clf.
        """
        super().__init__()

        self.vocab = Vocab(token_counts, max_size=max_vocab_size)

        self.embed_tokens = TokenEmbedding(token_counts)
        self.encode_lines = TokenLSTM(self.embed_tokens.out_size)

        self.merge = nn.Sequential(
            nn.Linear(self.encode_lines.out_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size),
        )

        self.dropout = nn.Dropout()

        self.predict = nn.Sequential(
            nn.Linear(embed_size, len(self.vocab)),
            nn.LogSoftmax(1),
        )

    def embed(self, lines, ctx):
        """Embed lines.
        """
        # Line lengths.
        sizes = [len(line) for line in lines]

        # Embed tokens, regroup by line.
        x = self.embed_tokens(list(chain(*lines)))
        xs = utils.group_by_sizes(x, sizes)

        # Embed lines.
        x = self.encode_lines(xs)

        # Blend encoder outputs, dropout.
        x = self.merge(x)
        x = self.dropout(x)

        return x

    def forward(self, lines, ctx=False):
        x = self.embed(lines, ctx)
        return self.predict(x)

    def collate_batch(self, batch):
        """Labels -> indexes.
        """
        lines, labels = list(zip(*batch))

        yt_idx = [self.vocab.stoi[label] for label in labels]
        yt = torch.LongTensor(yt_idx).to(DEVICE)

        return lines, yt


START_TOKEN = '[START]'
END_TOKEN = '[END]'
MASK_TOKEN = '[MASK]'


class MLMGenerator:

    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return sum(map(len, self.lines))

    def __iter__(self):
        """Generate (masked tokens, target) pairs.
        """
        for line in self.lines:
            for i, target in enumerate(line.clf_tokens):

                masked = [
                    token if j != i else MASK_TOKEN
                    for j, token in enumerate(line.clf_tokens)
                ]

                yield [START_TOKEN] + masked + [END_TOKEN], target

    def batches_iter(self, batch_size):
        return chunked_iter(iter(self), batch_size)
