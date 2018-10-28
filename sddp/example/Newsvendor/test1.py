import unittest
# from sddp.SDDP import *
from sddp.utilities import isapprox
from .newsvendor import *





class MyTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_converge(self):
        self.news = newsvendormodel()
        status = solve_default(self.news,
                               iteration_limit=20,
                               cut_selection_frequency=10,
                               simulation=MonteCarloSimulation(
                                   frequency=10,
                                   steps=list(range(10, 501, 10))

                               ),
                               bound_stalling=BoundStalling(
                                   iterations=5,
                                   atol=1e-3
                               )
                               )
        self.assertEqual(status, Staus.stalling_convergence)
        self.assertTrue(isapprox(self.news.getbound(), -97.9, 1e-3))

    def test_risk(self):
        pass





if __name__ == '__main__':
    unittest.main()
