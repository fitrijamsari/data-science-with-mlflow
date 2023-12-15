# Project Title: End-to-End MLOps for Data Science Projects

## Overview:

This repository showcases a comprehensive implementation of MLOps practices for a Data Science Regression Model Project, covering every stage from data ingestion to model deployment using MLflow. It demonstrates a structured workflow, emphasizing reproducibility, scalability, and automation in machine learning operations.

Integration with MLflow to tracking and monitor model training experiments using various model architecture. The repository can be use to compare which model architecture perform the best in the use case. Supported model architecture as the following:

- ElasticNet
- RandomForestRegression
- KNNRegression
- XGboostRegression

Upon comparing which model perform the best, we can proceed with GridSearchCV or RandomizeSearchCV to choose which hyperparamters is the best for the selected model.

## Key Features:

**Data Ingestion:** Download .csv dataset from URL.
**Data Validation:** Validate dataset including checking on dataset size and distribution. Validate if the dataset meet the schema requirements.
**Data Transformation:** Efficient robust data transformation strategies are implemented to ensure data quality including but not limited to data analytics, handling duplicates,handling imbalance dataset, imputing missing data, identify & handling outliers, data correlation data encoding, split dataset to train.csv & test.csv.
**Model Training:** Utilizes various machine learning algorithms and frameworks for model development, ensuring flexibility and experimentation with different models.
**Model Validation and Tracking:** Implements rigorous model validation techniques and leverages MLflow to track experiments, hyperparameters, metrics, and model versions.
**MLOps Workflow:** Illustrates an end-to-end MLOps workflow, incorporating CI/CD pipelines, version control, and model deployment strategies.
**Documentation and Best Practices:** Includes comprehensive documentation on project structure, code organization, and best practices in MLOps for enhanced collaboration and knowledge sharing.
**Technologies Used:**

- Python, Pandas, Scikit-learn
- MLflow for experiment tracking and model management
- Docker for containerization and orchestration
- Git for version control and collaboration

### BEST PRACTICE:

1. Run each model with default hyperparamter first.
2. Compare with model perform the best.
3. Tune the hyperparameter for the selected model to increase model performance before deployment.

### NOTES:

1. Any experiments or test code shall be run on "research" folder.

### STEPS:

Clone the repository

```bash
git clone https://github.com/fitrijamsari/data-science-with-mlflow.git
```

### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```

### STEP 02 - Install the requirements

```bash
pip install -r requirements.txt
```

### STEP 03 - Registration on dagshub

1. Register on dagshub, create a project and link to your github project repositories.
2. Execute your MLFlow Tracking from dagshub "Remote - Experiments" menu on your project environment.

```bash
export MLFLOW_TRACKING_URI=
export MLFLOW_TRACKING_USERNAME=
export MLFLOW_TRACKING_PASSWORD=
```

### STEP 04 - Update .env file

update the following secret parameter with your credential:

```bash
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```

### STEP 04 - Change config.yaml

Open the file config/config.yaml and change the data source url accordingly

Selected section which need to be changed:

```bash
data_ingestion:
    source_URL:

data_validation:
    unzip_data_dir: artifacts/01_data_ingestion/dataset.csv

data_transformation:
    test_split_percentage: 0.3

model_trainer:
    model_name: XGboostRegression #choose from params.yaml
```

### STEP 05 - Change params.yaml

Open the file params.yaml and change the model hyperparamter accordingly

### STEP 06 - Update schema.yaml

Update the schema.yaml according to the dataset.csv

```bash
source_URL: https://
```

### STEP 07 - Change params.yaml

Edit the params based on the selected models.

### STEP 05 - Run the Application

```bash
# Finally run the following command
python main.py
```

### STEP 08 - Model Performance Tracking.

You may view through daghubs - MLflowUI to compare various model performance based on evaluation metrics set.

## License:

This project is licensed under [MIT]. Please refer to the LICENSE file for details.

## References:

[MLflow Documentation](https://mlflow.org/docs/latest/index.html)
[dagshub Documentation](https://dagshub.com/)
