# Active Learning
**Author:** Luiz Matias</br>
2020-10-29

## About
This repository aims to demonstrate the active learning technique at the machine learning experience meetup

## Basic usage:

Fist, make sure you are in a virtual environment. If not (and you don't have virtualenv installed), please follow this commands:

```
pip3 install virtualenv 
virtualenv ../venv -p python3
source ../venv/bin/activate 
```
after that, use makefile command:
* `make deps`: It's gonna install all dependencies you need to run the experiment.

Then, you have to install a jupyter kernel. This command will install a kernel inside the environment, to use to run in the Jupyter notebook there:

```
python -m ipykernel install --user --name=active_learning
```
After this, run Jupyter and select the active_learning kernel to run the notebook


## Project Organization
------------

    ├── README.md                       <- The top-level README for developers using this project.
    │
    │   
    │
    ├── docs                            <- a HTML document about the experiment.
    │
    ├── notebooks                       <- Jupyter notebooks.
    │   ├── active_learning_simulation.py <- active learning with a synthetic data XOR
    │   ├── active_learning_text_classification.ipynb  <- active learning with a text classification problem
    │   
    │
    │   
    │
    │── src                             <- custom source code.
    │   ├── plot_functions.py           <- functions to plot graphs
    │   ├── preprocessing_functions.py  <- clean text function
    │
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis 
    │

--------


## Sources:


<a href="http://burrsettles.com/pub/settles.activelearning.pdf">Active Learning Paper</a>

<a href="https://modal-python.readthedocs.io/en/latest/">modAL framework</a>

<a href="https://medium.com/towards-artificial-intelligence/how-to-use-active-learning-to-iteratively-improve-your-machine-learning-models-1c6164bdab99">Active Learning - How to use</a>

<a href="https://medium.com/@tivadar.danka/how-to-use-unlabelled-data-to-get-more-training-data-218f300fffe4">How to use unlabelled data to get more training data</a>

<a href="https://www.youtube.com/watch?v=0efyjq5rWS4">PyCon.DE 2018: Building Semi-supervised Classifiers When Labeled Data Is Unavailable (YouTube video)</a>

<a href="https://arxiv.org/pdf/1805.00979.pdf">modAL paper</a>

<a href="https://towardsdatascience.com/active-learning-tutorial-57c3398e34d">active learning tutorial</a>

<a href="https://github.com/modAL-python/modAL">modAL github</a>
