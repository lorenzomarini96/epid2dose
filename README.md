# epid2dose
Deep learning-based portal dose prediction from transit EPID images.


# epid2dose

**epid2dose** is an open-source Python package for portal dose prediction from transit EPID images using the deep learning model developed in this work.

The software reproduces the inference pipeline presented in the accompanying PhD thesis, allowing users to generate predicted portal dose (PD) distributions from measured transit EPID images.

---

## Features

- Portal dose prediction from transit EPID images
- Fully automated preprocessing
- Deep learning inference using a trained U-Net model
- Batch processing of multiple EPID images
- Visualization and export of predicted portal dose distributions

---

## Repository structure

```
epid2dose/
│
├── pd_prediction.py        # Main inference script
├── demo.ipynb              # Interactive notebook
├── requirements.txt
│
├── models/                 # Neural network architecture
├── utils/                  # Utility functions
├── examples/               # Example input/output
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/USERNAME/epid2dose.git

cd epid2dose
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Linux / macOS

```bash
source .venv/bin/activate
```

Windows

```cmd
.venv\Scripts\activate
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Download trained weights

The trained model weights are not distributed with this repository because of GitHub file size limitations.

Download the weights from:

**TODO**

and place them inside

```
models/weights/
```

---

## Usage

Run the inference pipeline with

```bash
python pd_prediction.py \
    -i /path/to/input_folder \
    -o /path/to/output_folder
```

where

- `-i` : folder containing the input EPID images
- `-o` : output folder where the predicted portal dose images will be saved

---

## Demo

An interactive demonstration of the complete workflow is available in

```
demo.ipynb
```

---

## Citation

If you use this software in your research, please cite

```bibtex
@article{MARINI2026100966,
title = {Improving patient treatment accuracy using transit dosimetry with Electronic Portal Imaging Device images and deep learning},
journal = {Physics and Imaging in Radiation Oncology},
volume = {39},
pages = {100966},
year = {2026},
issn = {2405-6316},
doi = {10.1016/j.phro.2026.100966},
author = {Lorenzo Marini and Carlotta Mozzi and Michele Avanzo and Francesca Lizzi and Icro Meattini and Giovanni Pirrone and Alessandra Retico and Emmanuel Uwitonze and Aafke Christine Kraan and Cinzia Talamonti}
}
```

or cite the published article directly:

Marini L, Mozzi C, Avanzo M, Lizzi F, Meattini I, Pirrone G, Retico A, Uwitonze E, Kraan AC, Talamonti C. *Improving patient treatment accuracy using transit dosimetry with Electronic Portal Imaging Device images and deep learning*. **Physics and Imaging in Radiation Oncology**. 2026;39:100966. https://doi.org/10.1016/j.phro.2026.100966

---

## License

This project is distributed under the MIT License.
