# Guitar Mix Information Retrieval Using Machine Learning

This project aims to extract the mix information from the mixed guitar and use the mixing information to reconstruct the signal chain.

## Installation

Installation uses [Anaconda](https://www.anaconda.com/) for package management:

```bash
conda env create -f environment.yaml
```

Then activate the environment you've created with

```bash
conda activate gt_mir
```

## Check logs

```bash
tensorboard --logdir _log/Legacy/c53_classify --port 6066 
```

or

```bash
tensorboard --logdir _log/Legacy/c53_parameter --port 6067 
```

## Dataset

Download [IDMT-SMT-Audio-Effects Dataset](https://zenodo.org/record/7544032) and extract the unprocessed guitar signle notes and polyphonic sounds to `_assets/DATASET/IDMT-SMT-AUDIO-EFFECTS/`.

## Paper

The thesis "Guitar Mix Information Retrieval Using Machine Learning" and is available [here](https://github.com/DCisHurt/Guitar-MIR-Thesis).
