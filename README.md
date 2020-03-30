# Pneumothorax segmentation

Project for the MVA Deep Learning in Medical Imaging 2019 / 2020 course.

**Authors :** Théo Viel, Mathieu Orhan

## Install

You should start by cloning the repository:
```
git clone https://github.com/TheoViel/pneumothorax_segmentation.git
```

The project was tested on Ubuntu 18.04, CUDA 10.0 and Python 3.6.
To install the dependencies in a virtual environment using pip, run:
```
pip install -r requirements.txt
```

## Data preparation

In order to run the code, you will need to download and prepare the data.
The raw dataset (about 2 Go) can be downloaded [here](https://www.kaggle.com/seesee/siim-train-test). It requires a Kaggle account.
You should get a unique zipped file called `siim-train-test.zip`. Run these commands at the root of the project:
```
mkdir input
cd input
mv path/to/siim-train-test.zip .
unzip siim-train-test.zip -d .
```

To preprocess the data, simply run the notebook `notebooks/Data Preparation.ipynb`.

## Experiments

To run experiments, use the `Segmentation.ipynb` notebook in the `notebook` directory. The associated code is in the `src` directory. 
