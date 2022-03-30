# vivarium-ReaDDy

A Vivarium wrapper for [ReaDDy](https://readdy.github.io/)

---

# Installation

**Stable Release:** `pip install vivarium_readdy` (coming soon)<br>
**Development Head:** `pip install git+https://github.com/vivarium-collective/vivarium-ReaDDy.git`

## Local editable installation with pyenv + conda

To see all pyenv versions:

```
pyenv install list
```

To install a particular version of python (or conda):

```
pyenv install anaconda3-5.3.1
```

Install dependencies using pyenv + conda:

```
pyenv local anaconda3-5.3.1 # or whatever version you have installed
pyenv virtualenv vivarium-models
pyenv local vivarium-models
conda env update -f env.yml
```

## Local editable installation with conda alone

Install conda: https://docs.conda.io/en/latest/miniconda.html

Using conda, you can run

```
conda env create -f env.yml
conda activate vivarium-models
```

which will create and then activate a conda environment called `vivarium-models` with all the required dependencies (including ReaDDy) installed.

To update:

```
conda env update -f env.yml
```

# Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## Commands You Need To Know

1. `black vivarium_readdy`

    This will fix lint issues.

2. `make build`

    This will run `tox` which will run all your tests as well as lint your code.

3. `make clean`

    This will clean up various Python and build generated files so that you can ensure that you are working in a clean environment.

