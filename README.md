## Installation Instructions

Install the conda environment using the provided `environment.yml` file. This will set up all the necessary dependencies for the project.
```bash
conda env create -f environment.yml
```

Then install the local environment package using pip:
```bash
cd posggym-main/
pip install -e .
```

## Run
```bash
PYTHONPATH="." python run/racecar_warm_hj.py
```