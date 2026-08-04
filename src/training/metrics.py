from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundaryMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    exact: int = 0
    examples: int = 0
    word_errors: int = 0
    reference_words: int = 0

    def update(self, predicted: list[int], gold: list[int], predicted_text: str, gold_text: str) -> None:
        self.true_positive += sum(p == 1 and g == 1 for p, g in zip(predicted, gold))
        self.false_positive += sum(p == 1 and g == 0 for p, g in zip(predicted, gold))
        self.false_negative += sum(p == 0 and g == 1 for p, g in zip(predicted, gold))
        self.exact += int(predicted_text == gold_text)
        self.examples += 1
        reference = gold_text.split()
        hypothesis = predicted_text.split()
        self.word_errors += levenshtein(reference, hypothesis)
        self.reference_words += len(reference)

    def as_dict(self) -> dict[str, float | int]:
        precision_den = self.true_positive + self.false_positive
        recall_den = self.true_positive + self.false_negative
        precision = self.true_positive / precision_den if precision_den else 0.0
        recall = self.true_positive / recall_den if recall_den else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            "exact_sentence_accuracy": self.exact / self.examples if self.examples else 0.0,
            "word_error_rate": self.word_errors / self.reference_words if self.reference_words else 0.0,
            "examples": self.examples,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


def levenshtein(a: list[str], b: list[str]) -> int:
    if len(b) > len(a):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]

