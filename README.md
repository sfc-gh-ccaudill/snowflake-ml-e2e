# End-to-End ML Workflow on Snowflake

A step-by-step demo of building a complete machine learning pipeline on Snowflake, using a **patient risk stratification** use case. The pipeline covers infrastructure setup, synthetic data generation, feature engineering, model training, evaluation, deployment, monitoring, and streaming inference.

## Prerequisites

- A Snowflake account with access to [Snowpark Container Services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview) (for remote training and REST inference)
- A [named Snowflake connection](https://docs.snowflake.com/en/developer-guide/snowflake-cli/connecting/specify-credentials) configured locally (update `connection_name` in `source/config.yaml`)
- Python 3.9+
- The packages listed in `requirements.txt`

## Configuration

All pipeline settings are in **`source/config.yaml`**. At a minimum, update:

```yaml
snowflake:
  connection_name: DEMO  # Your Snowflake connection name
  database: ML_DEMO_PIPELINE_DB
  schema: HEALTHCARE
  warehouse: ML_DEMO_WAREHOUSE

compute:
  compute_pool: ML_DEMO_COMPUTE_POOL
```

## Pipeline Steps

Run the notebooks in order from the `notebooks/` directory. Each notebook begins by loading the shared config.

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_setup_infrastructure.ipynb` | Creates the database, schema, warehouse, tables, stages, and compute pool in Snowflake. |
| 2 | `02_data_generation.ipynb` | Generates synthetic patient encounter data and loads it into `RAW_PATIENT_DATA`. |
| 3 | `03_preprocessing.ipynb` | Reads raw data, engineers features (shock index, pulse pressure, BMI category, vital signs severity), creates a Snowflake Feature Store entity and feature view, and produces train/test splits. |
| 4 | `04_model_training.ipynb` | Trains a Random Forest classifier locally and remotely via ML Jobs on a Snowpark Container Services compute pool. Registers the model in the Snowflake Model Registry. |
| 5 | `05_model_evaluation.ipynb` | Runs inference on the test set using the registered model and computes classification metrics (accuracy, precision, recall, F1, confusion matrix). |
| 6 | `06_model_deployment.ipynb` | Deploys the model for SQL inference (warehouse) and REST inference (SPCS service endpoint). |
| 7 | `07_model_monitoring.ipynb` | Sets up a Model Monitor with a baseline, creates an inference log view, and configures drift detection alerts. |
| 8 | `08_streaming_inference.ipynb` | Simulates streaming patient data, calls the REST endpoint for real-time predictions, and writes results for the monitor to track. |
| 9 | `09_cleanup.ipynb` | Tears down all created objects (alerts, tasks, monitors, services, models, feature store, tables, stages). |

## Project Structure

```
.
├── notebooks/              # Ordered pipeline notebooks (01-09)
├── source/
│   ├── config.yaml         # Pipeline configuration
│   ├── configs.py          # Configuration dataclasses and loader
│   ├── train.py            # Model training logic (used by ML Jobs)
│   └── utils.py            # Session management and feature config helpers
├── data/
│   ├── historical.py       # Synthetic historical data generator
│   └── simulator.py        # Streaming data simulator with drift injection
├── setup/
│   ├── database_setup.py   # Database and schema creation
│   ├── tables_setup.py     # Table DDL
│   ├── stages_setup.py     # Stage creation
│   └── compute_pool_setup.py  # Compute pool provisioning
├── requirements.txt
└── README.md
```

## Snowflake Features Demonstrated

- **Snowpark** - Python DataFrames and session management
- **Feature Store** - Entity registration, feature views, and feature retrieval
- **Model Registry** - Model versioning, logging, and metadata tracking
- **Experiment Tracking** - Run logging with parameters and metrics
- **ML Jobs** - Remote model training on Snowpark Container Services
- **Model Deployment** - SQL inference (warehouse) and REST inference (SPCS)
- **Model Monitoring** - Drift detection, baseline comparison, and alerting

## Disclaimers

- **Synthetic data only.** All patient data is randomly generated and does not represent real individuals or medical records. It is not suitable for clinical use or medical decision-making.
- **Demo purposes.** This pipeline is designed to illustrate Snowflake ML capabilities and is not production-ready. Error handling, security hardening, CI/CD, and automated testing should be added for production deployments.
- **Cost awareness.** Running this demo creates Snowflake objects (warehouses, compute pools, SPCS services) that consume credits. Run the cleanup notebook (`09_cleanup.ipynb`) when finished to avoid unnecessary charges.
- **Preview features.** Some Snowflake ML features used in this demo (e.g., Model Monitoring, ML Jobs, Experiment Tracking) may be in public preview and subject to change.
