To set up the environment, follow these steps:

1. **Create and activate the conda environment**
```sh
conda create -n IT3190E_Group_4 python=3.10.12
conda activate IT3190E_Group_4
```

2. **Install the necessary packages**
```sh
pip install -r requirements.txt
```
- data_processing.ipynb is used to classify prices into labels

- The hyperparameter tuning folder is used to find the best hyperparameters of the models

- model.ipynb aggregates hyperparameters and evaluates models

- demo.ipynb is used for demo
