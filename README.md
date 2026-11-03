# Code for "Inferred precision as the carrier of conscious content"

Python 3 with numpy >= 2, scipy, matplotlib; python-docx, lxml and pandoc/xelatex only for building the manuscript.

- `wmp.py`: the model.
- `run_v2.py`: reproduces every number; writes `results.json` and `arrays.npz` (about 30 min on two cores).
- `figs2.py`: makes Fig1-Fig7.png from those files.
- `build_ms.py`: fills the numbers into `ms_*.md` and builds the PDF and Word files.

`results.json` comes from a full run of `run_v2.py`, except `pulse_amp`, whose block was added afterwards and run on its own with the same `perturb` function and states re-relaxed from the same values (the check values matched the full run to at least four digits).
