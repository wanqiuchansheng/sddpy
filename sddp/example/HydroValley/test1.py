#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################


from sddp.SDDP import solve_default
from sddp.example.HydroValley.hydro_valley import hydrovalleymodel
if __name__ == '__main__':
    deterministic_model = hydrovalleymodel(hasmarkovprice=False, hasstagewiseinflows=False)
    status = solve_default(deterministic_model, iteration_limit=100, cut_selection_frequency=1, print_level=0)