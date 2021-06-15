#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################


from typing import List
from sddp.utilities import *

T = TypeVar('T')



class NoiseRealization(Generic[T]):
    def __init__(self, observation: T, probability: float):
        # 具体的值
        self.observation = observation
        self.probability = probability


class DiscreteDistribution(Generic[T]):
    def __init__(self, noises):
        self.noises = noises  # type:List[NoiseRealization[T]]

    def create(self, observations: List, probabilities: List[float] = None):
        if probabilities is None:
            probabilities = [1 / len(observations)] * len(observations)

        if not isapprox(sum(probabilities), 1.0, atol=1e-6):
            raise RuntimeError("Finite discrete distribution must sum to 1.0")
        y = [NoiseRealization(xi, pi) for xi, pi in zip(observations, probabilities)]
        return DiscreteDistribution(y)

    @property
    def probabilities(self):
        return [n.probability for n in self.noises]

    def sample(self):
        nidx = sample(self.probabilities)
        return self.noises[nidx]

    def __getitem__(self, item):
        return self.noises[item]

    def __len__(self):
        return len(self.noises)
    #
    # def __next__(self):
    #     return self.noises.__next__()
