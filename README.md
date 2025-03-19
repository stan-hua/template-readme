
<!-- OPTIONAL: Project Logo
<p align="center">
  <img src="[OPT FILL: Path/link to logo image]" alt="Logo" style="width: 15%; display: block; margin: auto;">
</p>
-->

<h1 align="center"> [MUST FILL: Project Name] </h1>


<!-- OPTIONAL: Badges with Hyperlinks
<p align="center">
  <a href="[OPT FILL: Path/link to paper]"><img src="https://img.shields.io/badge/arXiv-2405.01535-b31b1b.svg" alt="arXiv"></a>
  <a href="[OPT FILL: Path/link to HuggingFace]"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
  <a href="[OPT FILL: Path/link to LICENSE]"><img src="https://img.shields.io/license-MIT-blue/license-MIT-blue.svg" alt="License"></a>
  <a href="[OPT FILL: Path/link to PyPI project]"><img src="https://img.shields.io/pypi/v/[OPT FILL: Name of PyPI package].svg" alt="PyPI version"></a>
</p>
-->

<p align="center">
  ⚡ A repository for [MUST FILL: 1-line description] 🚀 ⚡ <br>
</p>

---

## 🌲 About the Repo

<!-- OPTIONAL: Create Repository Structure Automatically
pip install rptree
rptree -d .
[OPT FILL: Copy structure to this README]
-->

```shell
my_project/
├── data/                   # Data directory
│   ├── datasets/               # Contains raw/processed datasets
│   │   ├── metadata/               # Metadata files 
│   │   ├── dataset_A               # Files for dataset A
│   │   └── dataset_B               # Files for dataset B
│   └── save_data/              # Saved artifacts from training/inference
│       ├── train_runs/             # Contains training runs (model checkpoints)
│       ├── predictions/            # Contains model predictions masks
│       ├── findings/               # Contains artifacts related to findings (e.g., tables)
│       └── figures/                # Contains figures
├── slurm/                  # Contains sbatch scripts for running on SLURM server
│   └── logs/                   # Stores logs from SLURM jobs
├── src/
│   ├── models/             # Contains model code
│   ├── data/               # Contains data code
│   └── utils/              # Contains utility funcitons
│       ├── data/
│       └── misc/
└── scripts/                # Contains scripts to run
   ├── data/                   # Data processing scripts
   └── model/                  # Model training/evaluation scripts
```



## 💴 About the Data

**Data Description**: [MUST FILL: 1-line description of data]

**Data Provider**: [MUST FILL: Remove after adding data providers]
* [Andrei Turinsky](mailto:andrei.turinsky@sickkids.ca)
    *  (Feb. 19, 2025) [MUST FILL: 1-line description of which data]


---

## 🔧 Installation

**(Automatic) Installation:**

```shell
# Option 1. Available on PyPI
pip install [OPT FILL: PyPI package name]

# Option 2. Local Pip Install
git clone https://github.com/[OPT FILL: Path to Repo]
cd [OPT FILL: Repository Name]
pip install -e .
```

**(Manual) Installation:**
```shell
# Get repository
git clone https://github.com/[OPT FILL: Path to Repo]
cd [OPT FILL: Repository Name]

# Install dependencies
# Option 1. Pip
pip install -r requirements.txt
# Option 2. Conda
conda env create -f environment.yaml
```

## 🏃 How to Run

**(Shell) [MUST FILL: Example of Command-Line Action]**

```shell
# Bash
bash script.sh

# Python (src/scripts/code.py)
python -m src.scripts.code

# Slurm Batch Job (slurm/job.sh)
sbatch slurm/job.sh
```

**(Python) [MUST FILL: Example of Python Import]**
```python
import [MUST FILL: Package Name]

args = [...]
kwargs = {...}
data = pkg.dummy_command(*args, **kwargs)
```

## 👏 Acknowledgements

**CCM Members**: [MUST FILL: Name of CCM Members with Emails]
1. [Andrei Turinsky](mailto:andrei.turinsky@sickkids.ca)

**Collaborators**: [MUST FILL: Name of PI(s) & External Collaborators]
1. [Anon PI](mailto:andrei.turinsky@sickkids.ca)


## Citation

If you find our work useful, please consider citing our paper!

```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
