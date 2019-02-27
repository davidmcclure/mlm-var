

import click
import torch
import math

from torch import optim, nn
from tqdm import tqdm

from mlm_var import logger
from mlm_var.model import MLM, DEVICE, MLMGenerator
from mlm_var.corpus import Corpus


def train_epoch(model, optimizer, loss_func, split):
    """Train a single epoch, return batch losses.
    """
    generator = MLMGenerator(split)
    batches = generator.batches_iter(50)

    losses = []
    with tqdm(total=len(generator)) as bar:
        for batch in batches:

            model.train()
            optimizer.zero_grad()

            lines, yt = model.collate_batch(batch)
            yp = model(lines)

            loss = loss_func(yp, yt)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            bar.update(len(lines))

            yield len(lines)


def predict(model, split):
    """Predict inputs in a split.
    """
    model.eval()

    generator = MLMGenerator(split)
    batches = generator.batches_iter(50)

    yt, yp = [], []
    with tqdm(total=len(generator)) as bar:
        for batch in batches:
            lines, yti = model.collate_batch(batch)
            yp += model(lines).tolist()
            yt += yti.tolist()
            bar.update(len(lines))

    yt = torch.LongTensor(yt)
    yp = torch.FloatTensor(yp)

    return yt, yp


def evaluate(model, loss_func, split):
    """Predict matches in split, log accuracy, return loss.
    """
    yt, yp = predict(model, split)
    return loss_func(yp, yt)


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

                logger.info('Val loss: %f' % loss.item())

                # Stop early.
                if len(losses) > es_wait and losses[-1] > losses[-es_wait]:
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
@click.option('--max_epochs', type=int, default=100)
@click.option('--es_wait', type=int, default=5)
@click.option('--eval_every', type=int, default=10000)
def train(src, max_epochs, es_wait, eval_every):
    """Train, dump model.
    """
    corpus = Corpus.load(src)
    token_counts = corpus.token_counts()

    model = MLM(token_counts).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.NLLLoss()

    model = train_model(model, optimizer, loss_func, corpus,
        max_epochs, es_wait, eval_every)


if __name__ == '__main__':
    cli()
