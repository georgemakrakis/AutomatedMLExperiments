# AutomatedMLExperiments
A Python program that lets you perform ML experiments in a convinient and automated way. This created as part of the semester project for the Advanced Security Incident Identification course (CS504-43) at University of Idaho 2019.

# Scope

The goal of this project is to create a tool for automating the process of conducting ML experiments upon time-series data of a specific format. The data correspond to EM emanations obtained by IoT devices when running in a normal or anomalous mode. 

The tool must provide a GUI/CLI for creating a configuration file. The program will rely on the configuration file to automatically create and run an experiment. The experiment may include steps like:

1. Load the dataset.

2. Plot sample observations.

3. Apply normalization/standardization or preprocessing/transformations upon the data like PCA or FFT.

4. Create models based on classifiers or clustering methods.

5. Split the datasets in certain ways e.g., training-test splits, k-folds.

6. Evaluate the models based on different metrics.

Note: You are required to make use of **sklearn pipelines**.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes

### Prerequisites

Python 3 must be installed into your system.

Inside the projects main directory create a folder called **dataset** and place the **.npy** files inside

Then, create a virtual enviroment to run the app by running the following commands through a CLI
```
python3 -m venv env

source env/bin/activate
```

Install all the required packages using **pip**
```
pip install -r requirements.txt
```


### Run

In order to run the app, from a CLI that has a virtual enviroment enabled, the following command can be used:

```
python src/main.py
```

When reach the "Select dataset files" question, select one file from data and one from labels. For example:
```
data_0inches.npy
labels_0inches.npy
```

Each one of the plots created from the app, will be placed inside the plots folder.

## Running the tests

The tests that exist in the **tests** folder are described below

### How to run tests

Each test file starts with the *test_* prefix, follwed by the file or the functions that it tests. No 100% code coverage has been achieved.

In order to run tests that test code in the **main.py** and **experiment.py** files, some lines must change to include the relative path of each module. 

In **main.py** change the line:

```
from experiment import run_experiment
```

to:

```
from src.experiment import run_experiment
```

and comment the last line 

```
# parse_arguments()
```

In **experiment.py** change:


```
from ploting import plot_observations, plot_precision_recall_curves
```

to:

```
from src.ploting import plot_observations, plot_precision_recall_curves
```

The tests can be run using the following command for the whole file:
```
python -m unittest -v tests/<file_name>
```

e.g. 

```
python -m unittest -v tests/test_experiment.py
```

For individual functions you can use the following syntax
```
python -m unittest -v tests.<file_name>.<class_name>.<function_name>
```

e.g.

```
python -m unittest -v tests.test_plots.TestPlot.test_plot_observations
```

### Coding style tests

The tests follow the style described in: [Getting Started With Testing in Python](https://realpython.com/python-testing/)

## Deployment

No live system deployment options for now.

## Not working

1. Check on the number of files choosen in the menu


## Authors

* **Georgios Michail Makrakis** - makr7178@vandals.uidaho.edu



