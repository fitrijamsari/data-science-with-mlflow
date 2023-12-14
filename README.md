# Project Title: End-to-End MLOps for Data Science Projects

## Overview:

This repository showcases a comprehensive implementation of MLOps practices for a Data Science project, covering every stage from data ingestion to model deployment using MLflow. It demonstrates a structured workflow, emphasizing reproducibility, scalability, and automation in machine learning operations.

## Key Features:

**Data Ingestion:** Download .csv dataset from URL.
**Data Validation:** Validate dataset including checking on dataset size and distribution. Validate if the dataset meet the schema requirements.
**Data Transformation:** Efficient robust data transformation strategies are implemented to ensure data quality including but not limited to data analytics, handling imbalance dataset, imputing missing data, identify & handling outliers, data correlation data encoding, split dataset to train.csv & test.csv.
**Model Training:** Utilizes various machine learning algorithms and frameworks for model development, ensuring flexibility and experimentation with different models.
**Model Validation and Tracking:** Implements rigorous model validation techniques and leverages MLflow to track experiments, hyperparameters, metrics, and model versions.
**MLOps Workflow:** Illustrates an end-to-end MLOps workflow, incorporating CI/CD pipelines, version control, and model deployment strategies.
**Documentation and Best Practices:** Includes comprehensive documentation on project structure, code organization, and best practices in MLOps for enhanced collaboration and knowledge sharing.
**Technologies Used:**

- Python, Pandas, Scikit-learn
- MLflow for experiment tracking and model management
- Docker for containerization and orchestration
- Git for version control and collaboration

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

```bash
source_URL: https://
```

### STEP 04 - Change params.yaml

Edit the params based on the selected models.

### STEP 05 - Run the Application

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up you local host and port
```

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the app.py

## License:

This project is licensed under [MIT]. Please refer to the LICENSE file for details.

## References:

[MLflow Documentation](https://mlflow.org/docs/latest/index.html)
[dagshub Documentation](https://dagshub.com/)
