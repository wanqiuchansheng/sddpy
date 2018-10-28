from typing import List
import numpy as np

from sddp.typedefinitions import AbstractCutOracle, Cut
from sddp.utilities import dominates


class StoredCut:
    def __init__(self, cut, non_dominated_count):
        self.non_dominated_count = non_dominated_count  # type:int
        self.cut = cut  # type:Cut


class SampledState:
    def __init__(self, state: List[float], best_object: float, best_cut_index: int):
        self.best_cut_index = best_cut_index
        self.best_object = best_object
        self.state = state


class LevelOneCutOracle(AbstractCutOracle):
    def __init__(self, cuts=None, states=None, sampled_states=None):
        if cuts is None: cuts = []
        if states is None: states = []
        if sampled_states is None: []  # TODO
        self.cuts = cuts  # type:List[StoredCut]
        self.states = states  # type:List[SampledState]
        self.sampled_states = sampled_states  # type:List[List[float]]

    def storecut(self, m: 'SDDPModel', sp: 'Subproblem', cut: 'Cut'):
        sense = sp.sense
        self.cuts.append(StoredCut(cut, 0))
        cut_index = len(self.cuts) - 1
        for state in self.states:
            y = cut.intercept + np.dot(cut.coefficients, state.state).__float__()
            if dominates(sense, y, state.best_object):
                self.cuts[state.best_cut_index].non_dominated_count -= 1
                self.cuts[cut_index].non_dominated_count += 1
                state.best_cut_index = cut_index
                state.best_object = y

        current_state = [f for f in m.stages[sp.stage].state]
        if len(current_state) == 0:
            return

        if current_state in self.sampled_states:
            return

        self.sampled_states.append(current_state)

        sampled_state = SampledState(current_state,
                                     cut.intercept + np.dot(cut.coefficients, current_state).__float__(),
                                     cut_index  # assume that the new cut is the best
                                     )
        self.states.append(sampled_state)
        self.cuts[cut_index].non_dominated_count += 1
        for (i, stored_cut) in enumerate(self.cuts):
            y = stored_cut.cut.intercept + np.dot(stored_cut.cut.coefficients, sampled_state.state)
            if dominates(sense, y, sampled_state.best_objective):
                # if new cut is strictly better
                # decrement the counter at the old cut
                self.cuts[sampled_state.best_cut_index].non_dominated_count -= 1
                # increment the counter at the new cut
                self.cuts[i].non_dominated_count += 1
                sampled_state.best_cut_index = i
                sampled_state.best_objective = y

    def validcuts(self):
        return [stored_cut.cut for stored_cut in self.cuts
                if stored_cut.non_dominated_count > 0]

    def allcuts(self):
        return [stored_cut.cut for stored_cut in self.cuts]
