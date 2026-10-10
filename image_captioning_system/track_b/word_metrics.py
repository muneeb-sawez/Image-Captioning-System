"""Word-presence classification metrics (pure Python, no PyTorch needed).

For each test image we ask, for every word in the vocabulary: "does this word belong in the caption?"
  truth      = the words that appear in ANY of the 5 human captions for the image
  prediction = the words the model used in its generated caption
Per image and vocabulary word:
  TP: in prediction and in truth      FP: in prediction, not in truth
  FN: in truth, not in prediction     TN: in neither
Counts are pooled over all test images (micro-average).
"""

STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "with", "at", "to", "is", "are", "and", "his", "her", "their", "its",
    "by", "for", "from", "up", "into", "over", "near", "while", "as", "it", "he", "she", "they", "that", "this",
    "be", "has", "have", "some", "out", "off", "down", "through", "around", "behind", "next", "other", "one",
}


def evaluate_sets(pred_tokens, truth_tokens, universe, drop=frozenset()):
    """pred_tokens / truth_tokens: one token list per image. universe: set of vocabulary words.
    drop: words to leave out entirely (e.g. STOPWORDS for the content-word version)."""
    uni = set(universe) - set(drop)
    tp = fp = fn = 0
    for pred, truth in zip(pred_tokens, truth_tokens):
        p, t = set(pred) & uni, set(truth) & uni
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    tn = len(pred_tokens) * len(uni) - tp - fp - fn
    return scores(tp, fp, fn, tn)


def scores(tp, fp, fn, tn):
    def div(a, b):
        return a / b if b else 0.0
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * precision * recall, precision + recall),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


if __name__ == "__main__":  # tiny hand-checked test: expect acc 0.8, precision 0.6, recall 0.6, specificity 0.8667, f1 0.6
    uni = {"a", "dog", "runs", "grass", "cat", "sits", "mat", "ball", "tree", "sky"}
    preds = [["a", "dog", "runs"], ["cat", "sits"]]
    truths = [["a", "dog", "grass"], ["cat", "mat"]]
    r = evaluate_sets(preds, truths, uni)
    print({k: round(v, 4) for k, v in r.items()})
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (3, 2, 2, 13)
    assert abs(r["accuracy"] - 0.8) < 1e-9 and abs(r["specificity"] - 13 / 15) < 1e-9 and abs(r["f1"] - 0.6) < 1e-9
    print("word_metrics self-test passed")
