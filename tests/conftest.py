# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# The tests pin coniZpM's device="auto" to the CPU (many xdist workers would
# otherwise share one GPU); test_gpu.py exercises the GPU backend explicitly.
# PFVS_TEST_DEVICE=auto|gpu runs everything else on that device instead.

import os

os.environ["PFVS_DEVICE"] = os.environ.get("PFVS_TEST_DEVICE", "cpu")
