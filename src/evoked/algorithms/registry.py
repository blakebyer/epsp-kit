from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Annotated, Union
from evoked.algorithms.ddt import DDT
from evoked.algorithms.dtw import DTW
from evoked.algorithms.matched_filter import MatchedFilter
from evoked.algorithms.peak import Peak
from evoked.algorithms.rms import RMS

# from evoked.algorithms.other import OtherAlgo  # add each new algorithm here only

AlgorithmType = Annotated[
    Union[DDT, DTW, MatchedFilter, Peak, RMS],
    Field(discriminator="method"),
]

