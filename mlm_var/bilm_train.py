

import click
import torch

from torch import optim, nn
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import functional as F

from mlm_var import logger
from mlm_var.bilm import Corpus, BiLM, DEVICE


def train_epoch(model, optimizer, loss_func, split, batch_size=50):
    """Train a single epoch, return batch losses.
    """
    loader = DataLoader(split, collate_fn=model.collate_batch,
        batch_size=batch_size)

    losses = []
    with tqdm(total=len(split)) as bar:
        for lines, yt in loader:

            model.train()
            optimizer.zero_grad()

            yp = model(lines)

            loss = loss_func(yp, yt)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            bar.update(len(lines))

            yield len(lines)


def evaluate(model, loss_func, split, batch_size=200):
    """Compute loss for split.
    """
    model.eval()

    loader = DataLoader(split, collate_fn=model.collate_batch,
        batch_size=batch_size)

    loss, size = 0, 0
    with tqdm(total=len(split)) as bar:
        for lines, yt in loader:
            loss += F.nll_loss(model(lines), yt, reduction='sum').item()
            size += len(yt)
            bar.update(len(lines))

    return loss / size


def train_model(model, optimizer, loss_func, corpus,
    max_epochs, es_wait, eval_every):
    """Train for N epochs, or stop early.
    """
    losses = []
    eval_n, total_n = 0, 0
    for i in range(max_epochs):

        logger.info(f'Epoch {i+1}')

        for n in train_epoch(model, optimizer, loss_func, corpus.train):

            total_n += n
            if total_n - eval_n >= eval_every:

                loss = evaluate(model, loss_func, corpus.val)
                losses.append(loss)

                logger.info('Val loss: %f' % loss)

                # Stop early.
                if len(losses) > es_wait and losses[-1] > losses[-es_wait]:
                    # TODO: Eval test.
                    break

                eval_n = total_n

    return model


@click.group()
def cli():
    pass


@cli.command()
@click.argument('src', type=click.Path())
@click.argument('dst', type=click.Path())
@click.option('--test_frac', type=float, default=0.1)
@click.option('--skim', type=int, default=None)
def build_corpus(src, dst, skim, test_frac):
    """Freeze off train/dev/test splits.
    """
    corpus = Corpus.from_spark_lines(src, skim, test_frac=test_frac)
    corpus.save(dst)


@cli.command()
@click.argument('src', type=click.Path())
@click.argument('dst', type=click.Path())
@click.option('--max_epochs', type=int, default=100)
@click.option('--es_wait', type=int, default=5)
@click.option('--eval_every', type=int, default=1000000)
def train(src, dst, max_epochs, es_wait, eval_every):
    """Train, dump model.
    """
    corpus = Corpus.load(src)
    token_counts = corpus.token_counts()

    model = BiLM(token_counts).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.NLLLoss()

    model = train_model(model, optimizer, loss_func, corpus,
        max_epochs, es_wait, eval_every)

    torch.save(model, dst)


if __name__ == '__main__':
    cli()
