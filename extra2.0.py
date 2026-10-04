# [CLAUDE FIX] Docstring below corrected: weight decay goes on the weights AND C, not the biases (see step()).
"""
A Neural Probabilistic Language Model
Bengio, Ducharme, Vincent (NIPS 2000 / JMLR 2003)

Pure NumPy re-implementation of the "Direct Architecture" -- no PyTorch, no
autograd. Every forward computation has a matching hand-derived backward
(gradient) computation, and parameters are updated with plain SGD that we
write ourselves.

    f(i, w_{t-1}, ..., w_{t-n}) = g(i, C(w_{t-1}), ..., C(w_{t-n}))

  x = [C(w_{t-n}), ..., C(w_{t-1})]     concatenated context feature vectors
  a = H x + d                           hidden pre-activation
  h = tanh(a)                           hidden layer
  y = b + U h + W x                     output scores  (W x = optional
                                          "direct connection" from Figure 1)
  p = softmax(y)                        P(w_t = i | context)

Training maximizes penalized log-likelihood via mini-batch SGD with weight
decay (L2 penalty on the weights {H, U, W} and on C, but NOT on the biases
{d, b}, as in the paper's regularizer R).
"""

import math
import random
from collections import Counter

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
RAW_TEXT = """
the cat is walking in the bedroom
a dog was running in a room
the dog is walking in the room
a cat was running in the bedroom
the cat is sleeping on the mat
a dog was sleeping on the rug
the dog is running in the garden
a cat was walking in the garden
the boy is reading a book in the room
a girl was reading a book in the bedroom
the boy is playing with the dog in the garden
a girl was playing with the cat in the room
the man is driving a car on the road
a woman was driving a car on the highway
the man is walking on the road
a woman was walking on the street
the children are playing in the garden
the children were sleeping in the bedroom
the teacher is reading a book to the children
a teacher was writing on the board
the dog is chasing the cat in the garden
a cat was chasing a mouse in the room
the mouse is hiding under the bed
a mouse was hiding in the wall
the sun is shining over the garden
the moon was shining over the lake
the wind is blowing over the field
a storm was coming over the sea
""".strip().split("\n")

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def tokenize(line):
    return line.strip().split()


def build_vocab(lines, min_freq=1):
    counter = Counter()
    for line in lines:
        counter.update(tokenize(line))
    itos = [BOS, EOS, UNK] + sorted(w for w, c in counter.items() if c >= min_freq)
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def build_examples(lines, stoi, n):
    # [CLAUDE FIX] removed the leftover "like NGramDataset did with PyTorch" phrase from this docstring
    """
    Slide an n-word window across every (padded) sentence and collect
    (context, target) pairs -- here we just build two plain NumPy integer arrays.
    """
    unk = stoi[UNK]
    contexts, targets = [], []
    for line in lines:
        tokens = [BOS] * n + tokenize(line) + [EOS]
        ids = [stoi.get(w, unk) for w in tokens]
        for t in range(n, len(ids)):
            contexts.append(ids[t - n:t])
            targets.append(ids[t])
    return np.array(contexts, dtype=np.int64), np.array(targets, dtype=np.int64)


def iterate_minibatches(contexts, targets, batch_size, shuffle=True):
    n_examples = contexts.shape[0]
    idx = np.arange(n_examples)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n_examples, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield contexts[batch_idx], targets[batch_idx]


# --------------------------------------------------------------------------- #
# 2. Model: parameters, forward pass, backward pass -- all manual
# --------------------------------------------------------------------------- #
class NeuralProbabilisticLM:
    """
    Parameters (all plain NumPy arrays):
        C : (V, m)          word feature vectors (the shared lookup table)
        H : (hidden, n*m)   hidden layer weights
        d : (hidden,)       hidden layer bias
        U : (V, hidden)     output weights (from hidden layer)
        b : (V,)            output bias
        W : (V, n*m)        output weights (direct connection), or None
    """

    def __init__(self, vocab_size, embed_dim=30, context_size=5,
                 hidden_dim=40, direct_connections=True):
        self.V = vocab_size
        self.m = embed_dim
        self.n = context_size
        self.hidden_dim = hidden_dim
        self.direct_connections = direct_connections
        concat_dim = context_size * embed_dim
        self.concat_dim = concat_dim

        # --- Part 1: lookup table C, |V| x m, random init in [-0.01, 0.01]
        self.C = np.random.uniform(-0.01, 0.01, size=(vocab_size, embed_dim))

        # --- Part 2: function g -----------------------------------------
        # [CLAUDE NOTE] The init below is plain uniform(-0.1, 0.1), not true Xavier/Glorot scaling,
        # so the "Xavier-ish" wording in the next lines is loose. Code is unchanged.
        # Standard "Xavier-ish" scaling so activations don't explode/vanish
        # at the start of training (the paper doesn't specify this in
        # detail; any small random init works, this just trains faster).
        self.H = np.random.uniform(-0.1, 0.1, size=(hidden_dim, concat_dim))
        self.d = np.zeros(hidden_dim)

        self.U = np.random.uniform(-0.1, 0.1, size=(vocab_size, hidden_dim))
        self.b = np.zeros(vocab_size)

        if direct_connections:
            self.W = np.random.uniform(-0.1, 0.1, size=(vocab_size, concat_dim))
        else:
            self.W = None

    # ---- helper: softmax over the last axis, numerically stable --------
    @staticmethod
    def _softmax(y):
        y = y - y.max(axis=1, keepdims=True)
        e = np.exp(y)
        return e / e.sum(axis=1, keepdims=True)

    # ---- forward pass ----------------------------------------------------
    def forward(self, context_ids):
        """
        context_ids: (batch, n) integer array of word IDs.
        Returns (probs, cache). `cache` holds every intermediate value we'll
        need again during backward() -- this replaces what autograd would
        normally track for us.
        """
        batch = context_ids.shape[0]

        # Embedding lookup + concatenation:
        #   C[context_ids] -> (batch, n, m)  -> reshape -> (batch, n*m)
        x = self.C[context_ids].reshape(batch, -1)

        a = x @ self.H.T + self.d          # (batch, hidden)
        h = np.tanh(a)                      # (batch, hidden)

        y = h @ self.U.T + self.b           # (batch, V)
        if self.direct_connections:
            y = y + x @ self.W.T            # + direct input->output term

        p = self._softmax(y)                # (batch, V) probabilities

        cache = {"context_ids": context_ids, "x": x, "a": a, "h": h, "p": p}
        return p, cache

    # ---- backward pass (hand-derived gradients) ---------------------------
    def backward(self, cache, targets):
        """
        Computes dL/d(param) for every parameter, where L is the *mean*
        negative log-likelihood over the batch:  L = -mean(log p[target]).

        Derivation:
          dL/dy_i = p_i - 1{i == target}     (standard softmax+NLL gradient)
          dL/dU   = dy^T h
          dL/db   = sum(dy)
          dL/dW   = dy^T x                    (only if direct_connections)
          dL/dh   = dy U
          dL/da   = dL/dh * (1 - h^2)          (tanh derivative)
          dL/dH   = da^T x
          dL/dd   = sum(da)
          dL/dx   = da H + dy W                (only add dy W if direct_connections)
          dL/dC[word] = sum of dL/dx at every context slot that used `word`
        """
        context_ids = cache["context_ids"]
        x, h, p = cache["x"], cache["h"], cache["p"]
        batch = context_ids.shape[0]

        # dL/dy: softmax + cross-entropy gradient, averaged over the batch
        dy = p.copy()
        dy[np.arange(batch), targets] -= 1.0
        dy /= batch                                    # mean, not sum

        grads = {}
        grads["U"] = dy.T @ h                            # (V, hidden)
        grads["b"] = dy.sum(axis=0)                       # (V,)

        dh = dy @ self.U                                   # (batch, hidden)
        da = dh * (1.0 - h ** 2)                             # tanh'(a) = 1 - tanh(a)^2

        grads["H"] = da.T @ x                               # (hidden, n*m)
        grads["d"] = da.sum(axis=0)                          # (hidden,)

        dx = da @ self.H                                     # (batch, n*m)

        if self.direct_connections:
            grads["W"] = dy.T @ x                              # (V, n*m)
            dx = dx + dy @ self.W                               # add direct-path contribution

        # Scatter dx (batch, n*m) back into per-word gradients on C.
        # Reshape to (batch, n, m): dx_reshaped[b, j, :] is the gradient
        # flowing into whichever word occupied context slot j of example b.
        dx_reshaped = dx.reshape(batch, self.n, self.m)
        grads["C"] = np.zeros_like(self.C)
        for j in range(self.n):
            # np.add.at does an "accumulate" scatter-add: if the same word
            # appears in multiple context slots/examples, its gradients sum.
            np.add.at(grads["C"], context_ids[:, j], dx_reshaped[:, j, :])

        return grads

    # ---- SGD update with weight decay (L2 penalty on theta, not C) -------
    # [CLAUDE FIX] the header above is outdated: the paper penalizes the weights AND C, not the biases.
    def step(self, grads, lr, weight_decay):
        # [CLAUDE FIX] original version, kept for reference (it decayed d and b, and skipped C):
        # theta_names = ["H", "d", "U", "b"] + (["W"] if self.direct_connections else [])
        # for name in theta_names:
        #     param = getattr(self, name)
        #     grad = grads[name] + weight_decay * param       # R(theta) = weight_decay * ||theta||^2
        #     setattr(self, name, param - lr * grad)
        # self.C = self.C - lr * grads["C"]
        # [CLAUDE FIX] corrected version: decay H, U, W and C; plain SGD for the biases d, b
        decayed_names = ["H", "U", "C"] + (["W"] if self.direct_connections else [])
        for name in decayed_names:
            param = getattr(self, name)
            grad = grads[name] + weight_decay * param
            setattr(self, name, param - lr * grad)
        for name in ["d", "b"]:
            setattr(self, name, getattr(self, name) - lr * grads[name])


# --------------------------------------------------------------------------- #
# 3. Training / evaluation loops
# --------------------------------------------------------------------------- #
def run_epoch(model, contexts, targets, batch_size, lr=None, weight_decay=1e-4):
    """
    One pass over the data. If `lr` is given, this is a training epoch
    (parameters get updated); otherwise it's pure evaluation.
    Returns (avg_nll, perplexity).
    """
    training = lr is not None
    total_nll, total_count = 0.0, 0

    for ctx_batch, tgt_batch in iterate_minibatches(contexts, targets, batch_size, shuffle=training):
        probs, cache = model.forward(ctx_batch)

        # Negative log-likelihood of the true next word, summed over the batch
        true_probs = probs[np.arange(len(tgt_batch)), tgt_batch]
        nll = -np.log(true_probs + 1e-12).sum()

        if training:
            grads = model.backward(cache, tgt_batch)
            model.step(grads, lr, weight_decay)

        total_nll += nll
        total_count += len(tgt_batch)

    avg_nll = total_nll / total_count
    perplexity = math.exp(avg_nll)
    return avg_nll, perplexity


def train(model, train_data, val_data, batch_size, epochs=80, lr=0.1,
          weight_decay=1e-4, patience=8, verbose=True):
    train_ctx, train_tgt = train_data
    val_ctx, val_tgt = val_data

    best_val_ppl = float("inf")
    best_params = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_nll, train_ppl = run_epoch(model, train_ctx, train_tgt, batch_size,
                                          lr=lr, weight_decay=weight_decay)
        val_nll, val_ppl = run_epoch(model, val_ctx, val_tgt, batch_size, lr=None)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"epoch {epoch:3d} | train ppl {train_ppl:8.2f} | val ppl {val_ppl:8.2f}")

        if val_ppl < best_val_ppl - 1e-3:
            best_val_ppl = val_ppl
            # Snapshot every parameter array (early stopping: keep the best
            # weights seen so far, not necessarily the final ones).
            best_params = {name: getattr(model, name).copy()
                            for name in ["C", "H", "d", "U", "b"]
                            if getattr(model, name) is not None}
            if model.direct_connections:
                best_params["W"] = model.W.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best val ppl {best_val_ppl:.2f})")
                break

    if best_params is not None:
        for name, value in best_params.items():
            setattr(model, name, value)
    return best_val_ppl


# --------------------------------------------------------------------------- #
# 4. Qualitative checks: nearest neighbours in C-space, next-word prediction
# --------------------------------------------------------------------------- #
def nearest_neighbors(model, itos, stoi, word, k=5):
    if word not in stoi:
        return []
    idx = stoi[word]
    C = model.C
    vec = C[idx]

    # Cosine similarity computed by hand: dot product / (norm * norm)
    dots = C @ vec
    norms = np.linalg.norm(C, axis=1) * np.linalg.norm(vec) + 1e-12
    sims = dots / norms

    order = np.argsort(-sims)                    # sort descending
    results = [(itos[i], float(sims[i])) for i in order if i != idx]
    return results[:k]


def predict_next(model, itos, stoi, context_words, n, k=5):
    ids = [stoi.get(w, stoi[UNK]) for w in ([BOS] * n + context_words)][-n:]
    context_ids = np.array([ids], dtype=np.int64)
    probs, _ = model.forward(context_ids)
    probs = probs[0]
    order = np.argsort(-probs)[:k]
    return [(itos[i], float(probs[i])) for i in order]


# --------------------------------------------------------------------------- #
# [CLAUDE ADDED] Gradient check: proves the hand-derived backward() is correct
# --------------------------------------------------------------------------- #
def gradient_check(eps=1e-5, tol=1e-6):
    """Compare backward() against numerical (finite-difference) gradients on a tiny model."""
    rng_state = np.random.get_state()      # restore afterwards so your seeded training run is unaffected
    model = NeuralProbabilisticLM(vocab_size=15, embed_dim=4, context_size=3,
                                  hidden_dim=5, direct_connections=True)
    ctx = np.random.randint(0, 15, size=(6, 3))
    tgt = np.random.randint(0, 15, size=6)

    def loss():
        probs, _ = model.forward(ctx)
        return -np.log(probs[np.arange(len(tgt)), tgt]).mean()

    _, cache = model.forward(ctx)
    grads = model.backward(cache, tgt)

    print("gradient check (finite differences vs. backward()):")
    worst = 0.0
    for name in ["C", "H", "d", "U", "b", "W"]:
        param = getattr(model, name)
        numeric = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = param[idx]
            param[idx] = old + eps
            loss_plus = loss()
            param[idx] = old - eps
            loss_minus = loss()
            param[idx] = old
            numeric[idx] = (loss_plus - loss_minus) / (2 * eps)
        # norm-based relative error (robust when individual gradient entries are ~0)
        err = np.linalg.norm(numeric - grads[name]) / (
            np.linalg.norm(numeric) + np.linalg.norm(grads[name]) + 1e-12)
        worst = max(worst, err)
        print(f"  {name}: relative error = {err:.2e}")
    print(f"  -> {'PASS' if worst < tol else 'FAIL'} (tolerance {tol:g})\n")
    np.random.set_state(rng_state)


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    # [CLAUDE ADDED] verify the backward pass before training
    gradient_check()

    n = 5              # context size
    m = 30             # word feature dimension
    hidden_dim = 40    # hidden units
    batch_size = 16
    epochs = 80
    lr = 0.3
    weight_decay = 1e-4

    itos, stoi = build_vocab(RAW_TEXT)
    vocab_size = len(itos)
    print(f"vocab size |V| = {vocab_size}")

    lines = RAW_TEXT[:]
    random.shuffle(lines)
    n_val = max(1, len(lines) // 10)
    n_test = max(1, len(lines) // 10)
    val_lines = lines[:n_val]
    test_lines = lines[n_val:n_val + n_test]
    train_lines = lines[n_val + n_test:]

    train_ctx, train_tgt = build_examples(train_lines, stoi, n)
    val_ctx, val_tgt = build_examples(val_lines, stoi, n)
    test_ctx, test_tgt = build_examples(test_lines, stoi, n)

    model = NeuralProbabilisticLM(
        vocab_size=vocab_size,
        embed_dim=m,
        context_size=n,
        hidden_dim=hidden_dim,
        direct_connections=True,
    )

    n_params = (model.C.size + model.H.size + model.d.size +
                model.U.size + model.b.size + (model.W.size if model.W is not None else 0))
    print(f"\nmodel: n={n}, m={m}, hidden_dim={hidden_dim}, params={n_params:,}\n")

    train(model, (train_ctx, train_tgt), (val_ctx, val_tgt),
          batch_size=batch_size, epochs=epochs, lr=lr, weight_decay=weight_decay)

    test_nll, test_ppl = run_epoch(model, test_ctx, test_tgt, batch_size, lr=None)
    print(f"\ntest set perplexity: {test_ppl:.2f}  (avg NLL {test_nll:.4f})")

    print("\nnearest neighbours in learned feature space C:")
    for w in ["dog", "cat", "walking", "running", "the", "a"]:
        neighbors = nearest_neighbors(model, itos, stoi, w)
        print(f"  {w:10s} -> {neighbors}")

    print("\nnext-word predictions:")
    for ctx in [["the", "cat", "is"], ["a", "dog", "was"], ["the", "children", "are"]]:
        preds = predict_next(model, itos, stoi, ctx, n)
        print(f"  {' '.join(ctx):25s} -> {preds}")


if __name__ == "__main__":
    main()
