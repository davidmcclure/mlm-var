
# Linguistic variation via masked language models (WIP)

Can we use something like the MLM (masked language model) objective from BERT to model linguistic variation across different corpora? In theory, this might make it possible to inductively identify differences in *how* words are being used, instead of just differences in *how frequently* words are used; and to identify specific instances of words/phrases that most typify those differences.

- Say we've got two corpora, A and B. Sample N sentences from each corpus.
- Train a MLM on these sentences, where the model just predicts masked tokens, and knows nothing about the underlying A/B labels on the training sentences.
- Train this to completion, and then treat it as a fixed encoder that produces both independent and contextual representations of tokens.
- Then, train a classifier that predicts the A/B sentence labels but sees as input *only individual (contextual) token embeddings*. Specifically, for each token in the corpus, produce training pairs:
  - [token, zeros] -> A/B
  - [token, contextual token] -> A/B
  - So, one pair where the classifier has access to both the independent token embedding and the sentence context; and another where the model just sees the isolated token embedding, and knows nothing about the context.
- Then, find cases where the addition of the sentence context (second case) makes the model significantly more accurate than just the isolated token (first case). Or, cases where `(word+context)` provides more information about the sentence label than just `(word)`.
