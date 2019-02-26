

from typing import List
from dataclasses import dataclass
from itertools import islice
from collections import Counter
from tqdm import tqdm

from torch.utils.data import random_split

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
