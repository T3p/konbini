# konbini

Combinatorial bandits with a Gymnasium interface and optional winner
feedback.

## Installation

Python 3.10 or newer is required. Create a virtual environment and install the
package with its Python dependencies:

On Debian or Ubuntu, the system prerequisites can be installed with:

```bash
sudo apt-get update
sudo apt-get install python3 python3-venv texlive-latex-base \
  texlive-latex-extra texlive-pictures
```

Then install the Python package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Plot compilation also requires a TeX Live installation that provides
`pdflatex`, `standalone`, and `pgfplots`. If TeX Live is already installed but
the two packages are missing, install them with:

```bash
tlmgr install standalone pgfplots
```

## Reproducing all experiments

From the repository root, run:

```bash
python scripts/compare_settings.py --all
```

This regenerates the CSV data in `scripts/results/`, then writes standalone
LaTeX sources and compiled PDFs to `plots/`. The complete run is
computationally expensive, especially the 100-instance correlated experiment.

The command runs four algorithms (OSMA, OSMA-W, EXP2, and K-Metaplayer) on:

- the correlated-reward instance used in the main paper;
- the correlated action-size experiment, with action sizes 2 through 10;
- the corrupted-reward instance;
- 100 independently generated correlated-reward sequences;
- the stationary uniform-reward instance;
- the nonstationary four-block instance; and
- the Bernoulli-reward instance.

The command uses the arm counts, horizons, good-arm counts, and
random seeds defined in `scripts/compare_settings.py`.
