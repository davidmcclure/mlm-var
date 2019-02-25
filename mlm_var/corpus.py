

from typing import List
from dataclasses import dataclass
from itertools import islice

from . import utils


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


class Corpus:

    @classmethod
    def from_spark_lines(cls, path, skim=None, **kwargs):
        """Read JSON gz lines.
        """
        lines_iter = islice(Line.read_spark_lines(path), skim)
        return cls(list(lines_iter), **kwargs)

    def __init__(self, lines, test_frac=0.1):
        self.lines = lines
        self.test_frac = test_frac
        # self.set_splits()
