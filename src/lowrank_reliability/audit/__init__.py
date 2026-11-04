"""Mechanism audit of the ST decoder.

* ``basis``: training second moment, fixed and random bases, spectra and
  donor directions (NumPy).
* ``readout``: ridge read-out of Eq. 2 (NumPy).
* ``hooks``: keep-k projections, direction patching and activation recording
  as PyTorch forward hooks (import explicitly; requires torch).
* ``state``: loading arc-state checkpoints and running arms over a data
  loader (import explicitly; requires torch).
"""
from . import basis, readout

__all__ = ["basis", "readout"]
