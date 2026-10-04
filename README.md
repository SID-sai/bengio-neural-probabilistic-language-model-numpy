# Neural Probabilistic Language Model in NumPy

A from-scratch implementation of the **neural probabilistic language model** from
Bengio, Ducharme, Vincent & Jauvin (2003), written in pure NumPy.
No PyTorch, no autograd: every forward computation has a hand-derived backward pass,
and the parameters are updated with SGD written by hand.

This is a learning project: I implemented the paper to understand how word embeddings
and neural language models work under the hood. It is **not** a benchmark or a
reproduction of the paper's results.

## The model

The model predicts the next word from the previous `n` words:

```
x = [C(w_{t-n}), ..., C(w_{t-1})]     concatenated context embeddings
h = tanh(H x + d)                     hidden layer
y = b + U h + W x                     output scores (W x = direct connections)
p = softmax(y)                        P(w_t | previous n words)
```

- `C` is the word-embedding table, learned jointly with everything else.
- `W x` is the paper's optional "direct connections" from input to output (can be turned off).
- Training minimizes the average negative log-likelihood with mini-batch SGD.
- Weight decay is applied to the weights (`H`, `U`, `W`) and to `C`, but not to the biases (`d`, `b`), as in the paper.

## What's in the code

| Part | What it does |
|---|---|
| `build_vocab`, `build_examples` | Builds the vocabulary and the (context, next-word) training pairs, with `<s>`, `</s>` and `<unk>` tokens |
| `NeuralProbabilisticLM` | The model: `forward`, hand-written `backward`, and `step` (SGD + weight decay) |
| `train` | Mini-batch training with validation perplexity and early stopping (keeps the best weights) |
| `gradient_check` | Compares the hand-written gradients against finite differences to verify `backward` |
| `nearest_neighbors`, `predict_next` | Inspect the learned embeddings and next-word predictions |

## Quick start

```bash
git clone <your-repo-url>
cd bengio-nplm-numpy
pip install -r requirements.txt
python extra.py
```

Tested with Python 3.12 and NumPy 2.4. Training takes a few seconds on a CPU.

## Example output

The gradient check runs first. All relative errors are tiny, so the backward pass is correct:

```
gradient check (finite differences vs. backward()):
  C: relative error = 7.88e-10
  H: relative error = 5.81e-08
  d: relative error = 5.19e-10
  U: relative error = 8.32e-08
  b: relative error = 8.22e-11
  W: relative error = 1.46e-08
  -> PASS (tolerance 1e-06)
```

Then training (seed 42), on 28 toy sentences with a vocabulary of 56 tokens and 18,416 parameters:

```
epoch   1 | train ppl    49.94 | val ppl    43.30
epoch  10 | train ppl     4.53 | val ppl    10.24
epoch  20 | train ppl     3.29 | val ppl    10.21
early stopping at epoch 24 (best val ppl 9.36)

test set perplexity: 5.57  (avg NLL 1.7168)
```

Example next-word predictions after training:

```
the cat is       -> walking (0.38), sleeping (0.10), shining (0.10), running (0.07), driving (0.07)
a dog was        -> walking (0.25), running (0.16), driving (0.08), reading (0.08), sleeping (0.07)
```

Exact numbers can vary slightly across NumPy versions and platforms.

## Limitations (please read)

- **The dataset is tiny.** 28 hand-written sentences, split into 24 train / 2 validation / 2 test. The test perplexity above is based on two sentences and is **noise**, not a meaningful result.
- **The vocabulary is built from all sentences**, including the validation and test ones, so there are no unseen words. A real setup would build it from the training split only.
- **The embedding neighbours are mostly noise at this scale.** With so little data, similarity results are unreliable (for example, "walking" and "book" come out close).
- **There is no baseline.** I have not compared this against a smoothed n-gram model.
- **Differences from the paper:** a toy dataset instead of real corpora, a constant learning rate, plain uniform initialization, and no mixing with an n-gram model.

## Possible next steps

- Train on a real corpus (for example Tiny Shakespeare) with a proper train/val/test split.
- Add a smoothed n-gram baseline and compare perplexities.
- Compare `direct_connections` on and off, and different context sizes `n`.

## Reference

Y. Bengio, R. Ducharme, P. Vincent, C. Jauvin.
*A Neural Probabilistic Language Model.*
Journal of Machine Learning Research 3 (2003), 1137-1155.

## Notes on how this was made

I wrote the first version of the implementation after reading the paper. I used Claude (an AI assistant) to review it, which led to two fixes: weight decay now matches the paper (it had been applied to the biases and not to `C`), and the gradient check was added. Changes made with AI help are marked with `# [CLAUDE ...]` comments in `extra.py`.
