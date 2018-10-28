import math
import unittest

from sddp.SDDP import solve_default, Sense, MonteCarloSimulation
from sddp.example.HydroValley.hydro_valley import hydrovalleymodel
from sddp.riskmeasures import EAVaR, DRO
from sddp.utilities import isapprox


class MyTestCase(unittest.TestCase):
    def test_deterministic(self):
        deterministic_model = hydrovalleymodel(hasmarkovprice=False, hasstagewiseinflows=False)
        status = solve_default(deterministic_model, iteration_limit=10, cut_selection_frequency=1, print_level=0)
        self.assertTrue(isapprox(deterministic_model.getbound(), 835.0, atol=1e-3))

    def test_stagewise(self):
        stagewise_model = hydrovalleymodel(hasmarkovprice=False)
        solve_default(stagewise_model, iteration_limit=20, print_level=0)
        self.assertTrue(isapprox(stagewise_model.getbound(), 838.33, atol=1e-2))

    def test_markov_prices(self):
        markov_model = hydrovalleymodel(hasstagewiseinflows=False)
        status = solve_default(markov_model, iteration_limit=10, print_level=0)
        self.assertTrue(isapprox(markov_model.getbound(), 851.8, atol=1e-3))

    def test_stagewise_inflows_and_markov_prices(self):
        markov_stagewise_model = hydrovalleymodel(hasstagewiseinflows=True, hasmarkovprice=True)
        solve_default(markov_stagewise_model, iteration_limit=10, print_level=0)
        self.assertTrue(isapprox(markov_stagewise_model.getbound(), 855.0, atol=1e-3))

    def test_riskaverse(self):
        """
        风险厌恶者
        """
        riskaverse_model = hydrovalleymodel(riskmeasure=EAVaR(lamb=0.5, beta=0.66))
        solve_default(riskaverse_model, iteration_limit=10, print_level=0)
        self.assertTrue(isapprox(riskaverse_model.getbound(), 828.157, atol=1e-3))

    def test_worst_case(self):
        worst_case_model = hydrovalleymodel(
            riskmeasure=EAVaR(lamb=0.5, beta=0.0), sense=Sense.Min)
        solve_default(worst_case_model,
                      iteration_limit=10,
                      simulation=MonteCarloSimulation(
                          frequency=2,
                          steps=list(range(20, 51, 10))
                      ))
        self.assertTrue(isapprox(worst_case_model.getbound(),  -780.867, atol=1e-3))

    def test_DRO(self):
        dro_model = hydrovalleymodel(hasmarkovprice=False, riskmeasure=DRO(math.sqrt(2 / 3) - 1e-6))
        solve_default(dro_model, iteration_limit=10, print_level=0)
        self.assertTrue(isapprox(dro_model.getbound(),  835.0, atol=1e-3))

    def test_DRO2(self):
        dro_model = hydrovalleymodel(hasmarkovprice=False, riskmeasure=DRO(1/6))
        solve_default(dro_model, iteration_limit=20, print_level=0)
        self.assertTrue(isapprox(dro_model.getbound(),  836.695, atol=1e-3))





if __name__ == '__main__':
    unittest.main()
