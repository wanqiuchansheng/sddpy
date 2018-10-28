# @version : python3.5
# @Time    : 2018/7/8 22:20
# @Author  : zzp
# @FileName: test1.py
from sddp.SDDP import solve_default
from sddp.example.HydroValley.hydro_valley import hydrovalleymodel
if __name__ == '__main__':
    deterministic_model = hydrovalleymodel(hasmarkovprice=False, hasstagewiseinflows=False)
    status = solve_default(deterministic_model, iteration_limit=100, cut_selection_frequency=1, print_level=0)