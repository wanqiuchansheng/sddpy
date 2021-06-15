#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

import math
from enum import Enum
from typing import List, Tuple
from sddp.typedefinitions import SDDPModel, Subproblem, AbstractRiskMeasure, Sense
import numpy as np


#
# def track_data(sourc: List, dest: List):
#     dest.clear()
from sddp.utilities import dominates


class Expectation(AbstractRiskMeasure):
    def __init__(self):
        pass

    def modifyprobability(self, riskadjusted_distribution: List[float], original_distribution: List[float],
                          observations: List[float], m: SDDPModel, sp: Subproblem):
        riskadjusted_distribution = [d for d in original_distribution]
        # riskadjusted_distribution.clear()
        # riskadjusted_distribution.extend(original_distribution)
        return riskadjusted_distribution


class WorstCase(AbstractRiskMeasure):
    def __init__(self):
        pass

    def modifyprobability(self, riskadjusted_distribution: List[float], original_distribution: List[float],
                          observations: List[float], m: 'SDDPModel', sp: 'Subproblem'):
        riskadjusted_distribution = [0.0] * len(riskadjusted_distribution)
        worst_idx = 0
        worst_observation = float("inf") if m.sense == Sense.Max else -float("inf")
        for idx, (probability, observation) in enumerate(zip(original_distribution, observations)):
            if probability > 0 and dominates(m.sense, observation, worst_observation):
                worst_idx = idx
                worst_observation = observation
        riskadjusted_distribution[worst_idx] = 1
        return riskadjusted_distribution


class RiskMeasures(Enum):
    Expectation = Expectation()
    WorstCase = WorstCase()


class AVaR(AbstractRiskMeasure):
    def __init__(self, beta: float):
        if beta > 1.0 or beta < 0:
            raise RuntimeError(
                "Beta must be in the range [0, 1]. Increasing values of beta are less risk averse. beta=1 is identical to taking the expectation.")
        self.beta = beta

    def modifyprobability(self, riskadjusted_distribution: List[float], original_distribution: List[float],
                          observations: List[float], m: 'SDDPModel', sp: 'Subproblem'):
        if self.beta < 1e-8:
            return RiskMeasures.WorstCase.value.modifyprobability(riskadjusted_distribution, original_distribution,
                                                                  observations, m, sp)
        elif self.beta > 1.0 - 1e-8:
            return RiskMeasures.Expectation.value.modifyprobability(riskadjusted_distribution, original_distribution,
                                                                    observations, m, sp)
        else:
            ismax = sp.sense == Sense.Max
            riskadjusted_distribution = [0.0] * len(riskadjusted_distribution)
            q = 0.0
            idx = np.argsort(observations)
            if not ismax:
                idx = reversed(idx)
            for i in idx:
                if q >= self.beta:
                    break
                avar_prob = min(original_distribution[i], self.beta - q) / self.beta
                riskadjusted_distribution[i] = avar_prob
                q += avar_prob * self.beta
            return riskadjusted_distribution





class ConvexCombination(AbstractRiskMeasure):
    def __init__(self, riskmeasures: List[Tuple[float, AbstractRiskMeasure]]):
        self.measures = riskmeasures  # type:List[Tuple[float,AbstractRiskMeasure]]

    def modifyprobability(self, riskadjusted_distribution: List[float], original_distribution: List[float],
                          observations: List[float], m: 'SDDPModel', sp: 'Subproblem'):
        riskadjusted_distribution = np.array([0.0] * len(riskadjusted_distribution))
        for wight, measure in self.measures:
            y = [0.0] * len(original_distribution)
            y = measure.modifyprobability(y, original_distribution, observations, m, sp)
            riskadjusted_distribution = np.multiply(wight, y) + riskadjusted_distribution
        return riskadjusted_distribution.tolist()


class EAVaR(ConvexCombination):
    def __init__(self, lamb: float = 1, beta: float = 0):
        if lamb > 1.0 or lamb < 0.0:
            raise RuntimeError(
                "Lambda must be in the range [0, 1]. Increasing values of lambda are less risk averse. lambda=1 is identical to taking the expectation.")
        if beta > 1.0 or beta < 0.0:
            raise RuntimeError(
                "Beta must be in the range [0, 1]. Increasing values of beta are less risk averse. beta=1 is identical to taking the expectation.")
        self.measures = [
            (lamb, Expectation()),
            (1 - lamb, AVaR(beta))
        ]


class DRO(AbstractRiskMeasure):
    def __init__(self, radius):
        self.radius = radius  # type:float

    def popvar(self, x: List[float]) -> float:
        """
        方差
        """
        ninv = 1 / len(x)
        return ninv * sum(_x ** 2 for _x in x) - (ninv * sum(x)) ** 2

    def popstd(self, x):
        """
        标准差
        """
        return math.sqrt(self.popvar(x))

    def is_dro_applicable(self, radius: float, observations: List[float]):
        if abs(radius) < 1e-9:
            return False
        elif abs(self.popstd(observations)) < 1e-9:
            return False
        return True

    def getconstfactor(self, S: int, k: int, radius: float, permuted_observations: List[float]):
        stdz = self.popstd(permuted_observations[k :S])
        return math.sqrt((S - k) * radius ** 2 - k / S) / (stdz * (S - k))

    def getconstadditive(self, S: int, k: int, const_factor: float, permuted_observations: List[float]):
        avgz = np.mean(permuted_observations[k:S ])
        return 1 / (S - k) + const_factor * avgz

    def modifyprobability(self, riskadjusted_distribution: List[float],
                          original_distribution: List[float], observations: List[float], m: 'SDDPModel',
                          sp: 'Subproblem'):
        S = len(observations)
        r = self.radius
        if not self.is_dro_applicable(r, observations):
            riskadjusted_distribution = [1 / S] * S
            return riskadjusted_distribution

        if sp.sense == Sense.Min:
            perm = np.argsort(observations)
            permuted_observations = (-np.sort(observations)).tolist()
        else:
            perm = (np.argsort(observations)[::-1]).tolist()
            permuted_observations = (np.sort(observations)[::-1]).tolist()

        for k in range(S - 1):
            if k > 0:
                riskadjusted_distribution[perm[k - 1]] = 0.0
            const_factor = self.getconstfactor(S, k, r, permuted_observations)
            const_additive = self.getconstadditive(S, k, const_factor, permuted_observations)
            for i in range(k+1, S+1):
                riskadjusted_distribution[perm[i-1]] = const_additive - const_factor * permuted_observations[i-1]

            if riskadjusted_distribution[perm[k]] >= 0.0:
                break
        return riskadjusted_distribution
