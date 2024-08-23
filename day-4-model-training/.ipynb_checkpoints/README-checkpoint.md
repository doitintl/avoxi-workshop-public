# Vertex AI Pipelines for Anomaly Detection: Workshop Day 4

This repository contains a Vertex AI Pipeline designed for anomaly detection using tabular data. It leverages Google Cloud services like Dataproc Serverless, BigQuery ML, Vertex AI Endpoints, and Vertex AI Pipelines for a seamless and automated workflow.

## Table of Contents

- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Components](#components)
- [Pipeline Definition](#pipeline-definition)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Deployment and Validation](#model-deployment-and-validation)
- [Configuration File](#configuration-file)
- [Running the Pipeline](#running-the-pipeline)
- [Cost Analysis](#cost-analysis)
- [Pipeline Graph via Google Cloud Console](#pipeline-graph-via-google-cloud-console)

## Overview

This pipeline automates the following steps:

1. **Data Preprocessing:** Using Dataproc Serverless with a PySpark script to clean, normalize, and identify anomalies in the input data.
2. **Data Loading:** Loading preprocessed data into BigQuery tables.
3. **Model Training:** Training an anomaly detection model using BigQuery ML.
4. **Model Evaluation:** Evaluating the trained model using relevant metrics.
5. **Model Selection:** Choosing the best performing model based on evaluation metrics.
6. **Model Deployment:** Deploying the selected model to a Vertex AI Endpoint for serving.
7. **Model Validation:** Validating the deployed model by sending a prediction request.

## Dataset Information

The dataset used for this anomaly detection pipeline contains information about phone calls and their quality metrics.  It has the following characteristics:

* **Table Size:** 1.59 GB
* **Number of Rows:** 9,872,691

**Schema:**

| Field Name          | Type     | Mode      | Description                                                                    |
|----------------------|----------|-----------|--------------------------------------------------------------------------------|
| caller_id           | STRING   | NULLABLE   | GUID, unique identifier for a phone call                                     |
| organization_id     | STRING   | NULLABLE   | GUID, customer identifier                                                     |
| e164_from_caller_id | STRING   | NULLABLE   | Caller ID for the phone call, in international format including country code |
| e164_to_caller_id   | STRING   | NULLABLE   | Dialed number for the phone call, in international format including country code |
| status              | STRING   | NULLABLE   | What happened on the call: answered, unanswered, voicemail                     |
| start_time          | TIMESTAMP | NULLABLE   | Time the call began                                                          |
| duration            | INTEGER  | NULLABLE   | Length of the call in integer seconds                                          |
| data_center         | STRING   | NULLABLE   | Data Center AVOXI processed the call in                                      |
| carrier_id          | INTEGER  | NULLABLE   | Numeric identifier of the AVOXI vendor that processed the call               |
| packet_loss         | FLOAT    | NULLABLE   | Percent of audio packets that were not delivered                             |
| mos                 | FLOAT    | NULLABLE   | Mean Opinion Score of call, a measure of quality derived from packet loss and jitter. |
| jitter              | FLOAT    | NULLABLE   | Average inter-packet arrival time deviated from expected. Larger absolute values are worse, buffers in the system typically hide anything under 100ms. |

## Prerequisites

* **Google Cloud Project:** A GCP project with billing enabled.
* **Permissions:** Necessary permissions to create and manage resources in Vertex AI, BigQuery, Dataproc, and GCS.
* **Python:** Python 3.7+ installed.
* **Libraries:** Install required libraries:
  ```bash
  pip install google-cloud-aiplatform kfp google-cloud-bigquery phonenumbers
  ```

## Architecture

This pipeline utilizes a serverless architecture on Google Cloud, leveraging managed services for each stage of the workflow.

* **Data Ingestion & Preprocessing:** Raw data is stored in Google Cloud Storage, and Dataproc Serverless is used for efficient and scalable data preprocessing with PySpark.
* **Model Training & Evaluation:** BigQuery ML provides a convenient way to train and evaluate machine learning models directly within BigQuery, using SQL.
* **Model Deployment & Serving:** Vertex AI Endpoints provide a scalable and managed environment for deploying and serving the trained models.
* **Orchestration:** Vertex AI Pipelines orchestrates the entire workflow, managing dependencies and ensuring the seamless execution of each step.

## Components

The pipeline is built using custom and pre-built components:

* **`get_config`:** Fetches configuration settings from a GCS bucket.
* **`load_bigquery_tables`:** Loads data into BigQuery tables from GCS.
* **`interpret_bqml_evaluation_metrics`:** Analyzes BQML model evaluation metrics.
* **`load_bigquery_model`:** Loads a BQML model into the pipeline.
* **`select_best_model`:** Compares model performance metrics and selects the best model.
* **`validate_infrastructure`:** Validates the deployed model infrastructure by sending a prediction request.

## Pipeline Definition

The `vertex_ai_pipeline` function defines the workflow using the Kubeflow Pipelines (KFP) DSL. It orchestrates the components in a sequence, passing outputs from one component as inputs to the next.

## Data Preprocessing

The `preprocessing_v2.py` script performs data preprocessing using Dataproc Serverless and PySpark:

* **Data Loading:** Reads CSV data from Google Cloud Storage.
* **Column Renaming:** Standardizes column names.
* **Data Cleaning:** Handles missing values and removes unnecessary columns.
* **Phone Number Parsing:** Extracts country information from phone numbers.
* **Data Normalization:** Normalizes numeric features using StandardScaler.
* **Anomaly Detection:** Identifies anomalies based on predefined criteria.
* **Output Writing:** Writes processed data back to Google Cloud Storage.

## Model Training and Evaluation

BigQuery ML is used to train and evaluate the anomaly detection model. The `configuration.yaml` file contains the SQL query for model creation. The pipeline uses predefined metrics for evaluating the model's performance.

## Model Deployment and Validation

The best performing model is deployed to a Vertex AI Endpoint. The pipeline then validates the deployed model by sending a prediction request and verifying the response.

## Configuration File

The `configuration.yaml` file contains essential parameters for the pipeline, such as project ID, bucket name, dataset ID, model name, and more. It also includes configurations for Dataproc Serverless and the BigQuery ML model.

## Running the Pipeline

1. Clone the repository.
2. Update the `configuration.yaml` file with your project details.
3. Upload `configuration.yaml` to GCS
3. Specify GCS path within `workshop.ipynb`
3. Run the pipeline script:

   ```bash
   python pipeline_v2.py
   ```

## Cost Analysis

* Table Size: 1.59 GB
* Number of Rows: 9,872,691

| Service          | Cost ($) |
|------------------|----------|
| Vertex AI        | 0.64     |
| Dataproc          | 0.33     |
| Networking       | 0.30     |
| Compute Engine    | 0.01     |
| Cloud DNS        | 0.01     |
| Artifact Registry | 0.01     |
| **Total**         | **1.30**   |

***Note:*** the costs presented in this table are estimates and are intended to provide an approximate upper-bound figure. There is potential for significant cost reduction through the optimization of each stage within our operational pipeline. We encourage stakeholders to consider these figures as a starting point for further refinement and cost-saving initiatives.

## Pipeline Graph via Google Cloud Console

You can visualize the pipeline's execution graph in the [Google Cloud Console](https://console.cloud.google.com/vertex-ai/pipelines/locations/us-central1/runs/avox-pipeline-20240823014848?project=gurkomal-playground). This provides a clear representation of the workflow and the status of each component.