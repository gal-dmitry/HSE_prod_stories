import os.path as osp
import sys
from typing import Callable, List, Optional, Sequence

import numpy as np

from hunspell import HunSpell
from textdistance import levenshtein, jaro_winkler, needleman_wunsch


class SpellChecker:
    def __init__(
            self,
            dict_path: str = './dictionaries/en',
            num_suggestions: int = 5,
            distances: Optional[Sequence[Callable[[str, str], float]]] = None
    ) -> None:
        self._hunspell = HunSpell(
            osp.join(dict_path, 'index.dic'), osp.join(dict_path, 'index.aff')
        )
        self._num_suggestions = num_suggestions
        self._distances = distances or [
            levenshtein.normalized_distance,
            jaro_winkler.normalized_distance,
            needleman_wunsch.normalized_distance
        ]

    def _compute_distances(self, word: str, candidates: str) -> List[float]:
        return [sum(d(word, c) for d in self._distances) for c in candidates]

    def suggest_if_needed(self, word: str) -> List[str]:
        if self._hunspell.spell(word):
            return []
        candidates = self._hunspell.suggest(word)
        distances = self._compute_distances(word, candidates)
        if self._num_suggestions < len(candidates):
            topk_idx = np.argpartition(
                distances, self._num_suggestions
            )[:self._num_suggestions]

            candidates = [candidates[i] for i in topk_idx]
            distances = [distances[i] for i in topk_idx]

        return [x[1] for x in sorted(zip(distances, candidates))]


def spell_check():
    words = sys.argv[1:]
    spell_checker = SpellChecker()

    for word in words:
        suggestions = spell_checker.suggest_if_needed(word)
        if len(suggestions) != 0:
            print(word.ljust(max(map(len, words))), '| Suggestions:', suggestions)
        else:
            print(
                word.ljust(max(map(len, words))),
                '| The word is correct or nothing to suggest!'
            )
