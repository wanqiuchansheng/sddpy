import unittest
from sddp.typedefinitions import CachedVector


class MyTestCase(unittest.TestCase):
    def test_something(self):
        ls = [1, 2, 3, 4, 5]
        cl = CachedVector(ls)
        self.assertEqual(cl[1], ls[1])
        cl[1] = 10
        self.assertEqual(10, ls[1])
        self.assertEqual(len(cl), len(ls))
        cl[1] *= 2
        self.assertEqual(cl[1], ls[1])
        print("c1[1]=%f" % cl[1])
        self.assertEqual(cl[1], 20)
        print(cl.range(list(range(3))))


if __name__ == '__main__':
    unittest.main()
