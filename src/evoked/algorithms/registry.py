from __future__ import annotations

from evoked.algorithms.linear import Peak, MatchedFilter
from evoked.algorithms.probabilistic import GLRT
from evoked.algorithms.nonlinear import DTW
# from evoked.algorithms.other import OtherAlgo  # add each new algorithm here only


ALGORITHMS = {
    "Peak": Peak,
    "MatchedFilter": MatchedFilter,
    "GLRT": GLRT,
    "DTW": DTW,
}


def parse_algorithm(data):
    if not isinstance(data, dict):
        return data

    data = dict(data)

    algorithm_type = data.pop("type")

    try:
        algorithm_class = ALGORITHMS[algorithm_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown algorithm type: {algorithm_type}"
        ) from exc

    return algorithm_class(**data)