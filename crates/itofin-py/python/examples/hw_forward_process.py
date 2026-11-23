"""Standalone forward-measure process, not a calibrated Hull-White model."""

# itofin library
from itofin.processes import HullWhiteForwardProcess
from itofin.termstructures import FlatForward
from itofin.time import Date, DayCounter

curve = FlatForward(Date(23, 10, 2026), 0.03, DayCounter.actual365_fixed())
process = HullWhiteForwardProcess(curve, 0.1, 0.02)
process.set_forward_measure_time(5.0)
print(process.expectation(1.0, 0.03, 0.8), process.variance(1.0, 0.03, 0.8))
