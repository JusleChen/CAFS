# CAFS / Free-BiSC-HyperSBL core

This repository provides reusable numerical methods for complex EEG
source inversion.

Python 3.11 is required.

## Install

```powershell
python -m pip install -e .
```

## Core modules

- `src/free_bisc_hypersbl/calibration/`: bilevel and nonlinear physical calibration.
- `src/free_bisc_hypersbl/forward/`: reference transforms, spherical-harmonic forward operators, and BEM geometry utilities.
- `src/free_bisc_hypersbl/freesigm/`: stability-aware Free-SIGM weighting and safe evidence solvers.
- `src/free_bisc_hypersbl/frequency/`: frequency-aware closed-form weighting utilities.
- `src/free_bisc_hypersbl/inverse/`: complex SBL, Champagne, uncertainty-aware, structured, and HQSL inverse solvers.
- `src/free_bisc_hypersbl/utils/`: small shared reproducibility helpers.

Example imports:

```python
from free_bisc_hypersbl.forward import standard_eeg_leadfield_family
from free_bisc_hypersbl.inverse import ComplexGroupSBL
from free_bisc_hypersbl.freesigm import solve_safe_freesigm
```
