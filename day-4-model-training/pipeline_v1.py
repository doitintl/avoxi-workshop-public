import yaml
import json
from typing import NamedTuple
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1 import bigquery as bq_components
from google_cloud_pipeline_components.types import artifact_types
import kfp
from kfp import dsl, compiler
from kfp.dsl import Artifact, Input, Metrics, Output, component, placeholders

def get_config(config_gcs_path : str) -> dict:
    import json
    from google.cloud import storage
    from datetime import datetime
    # Initialize GCS client
    storage_client = storage.Client()
    
    # Extract the bucket name and blob (file) name from the config GCS path
    bucket_name, blob_name = config_gcs_path.replace("gs://", "").split("/", 1)
    
    # Download the YAML file from GCS
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    config_data = blob.download_as_text()
    config_data = yaml.safe_load(config_data)
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    BATCH_ID = "avoxi-workshop-" + TIMESTAMP
    config_data['batch_id'] = BATCH_ID
    return config_data    


@component(packages_to_install=["google-cloud-bigquery==3.24.0"], base_image="gcr.io/ml-pipeline/google-cloud-pipeline-components:2.15.0")
def load_bigquery_tables(anomaly_data_path: str, no_anomaly_data_path: str, anomaly_table_name: str, no_anomaly_table_name: str, anomaly_table : Output[artifact_types.BQTable], normal_table: Output[artifact_types.BQTable]):
    from google.cloud import bigquery
    # Initialize BigQuery client
    client = bigquery.Client()

    # Define the external connection for anomaly data
    external_config_anomaly = bigquery.ExternalConfig('CSV')
    external_config_anomaly.source_uris = [f"{anomaly_data_path}*.csv"]
    external_config_anomaly.autodetect = True

    # Define the external connection for non-anomaly data
    external_config_no_anomaly = bigquery.ExternalConfig('CSV')
    external_config_no_anomaly.source_uris = [f"{no_anomaly_data_path}*.csv"]
    external_config_no_anomaly.autodetect = True

    # Create or replace the anomaly table
    anomaly_table = artifact_types.BQTable.create(name="anomaly_table", project_id="gurkomal-playground", dataset_id="avoxi_workshop",table_id="anomaly_table")
    table_anomaly = bigquery.Table(anomaly_table_name)
    table_anomaly.external_data_configuration = external_config_anomaly
    client.create_table(table_anomaly, exists_ok=True)

    # Create or replace the non-anomaly table
    normal_table = artifact_types.BQTable.create(name="normal_table", project_id="gurkomal-playground", dataset_id="avoxi_workshop",table_id="no_anomaly_table")
    table_no_anomaly = bigquery.Table(no_anomaly_table_name)
    table_no_anomaly.external_data_configuration = external_config_no_anomaly
    client.create_table(table_no_anomaly, exists_ok=True)

@component()
def interpret_bqml_evaluation_metrics(
    bqml_evaluation_metrics: Input[Artifact], 
    metrics: Output[Metrics]
) -> dict:
    import math

    metadata = bqml_evaluation_metrics.metadata
    for r in metadata["rows"]:

        rows = r["f"]
        schema = metadata["schema"]["fields"]

        output = {}
        for metric, value in zip(schema, rows):
            metric_name = metric["name"]
            val = float(value["v"])
            output[metric_name] = val
            metrics.log_metric(metric_name, val)
            if metric_name == "mean_squared_error":
                rmse = math.sqrt(val)
                metrics.log_metric("root_mean_squared_error", rmse)

    metrics.log_metric("framework", "BQML")
    print(output)

@component
def load_bigquery_model(project_id : str, dataset_id : str, model_id : str, model_registry : str, region : str, model: Output[Artifact]):
    project_id = project_id
    dataset_id = dataset_id
    model_id = model_id
    model_registry_id = model_registry
    region=region

    model.metadata['projectId'] = project_id
    model.metadata['datasetId'] = dataset_id
    model.metadata['modelId'] = model_id
    model.uri = f'projects/{project_id}/datasets/{dataset_id}/models/{model_id}'
    model.metadata['resourceName'] = f'projects/{project_id}/locations/{region}/models/{model_registry_id}'

@component(base_image="python:3.9")
def select_best_model(
    metrics_bqml_challenger: Input[Metrics],
    metrics_bqml_blessed: Input[Metrics],
    thresholds_dict_str: str,
    best_metrics: Output[Metrics],
    reference_metric_name: str = "mean_squared_error",
) -> NamedTuple(
    "Outputs",
    [
        ("deploy_decision", str),
        ("best_model", str),
        ("metric", float),
        ("metric_name", str),
    ],
):
    import json
    from collections import namedtuple

    best_metric = float("inf")
    best_model = None

    metric_bqml_challenger = float("inf")
    metric_bqml_blessed = float("inf")

    try:
        metric_bqml_challenger = metrics_bqml_challenger.metadata[reference_metric_name]
        print(f"Challenger Metric bqml: {metric_bqml_challenger}")
    except:
        print(f"{reference_metric_name} doesn't exist in the BQML dictionary")

    try:
        metric_bqml_blessed = metrics_bqml_blessed.metadata[reference_metric_name]
        print(f"Blessed Metric bqml: {metric_bqml_blessed}")
    except:
        print(f"{reference_metric_name} doesn't exist in the BQML dictionary")

    # Change condition if higher is better.
    print(f"Comparing Challenger BQML ({metric_bqml_challenger}) vs Blessed BQML ({metric_bqml_blessed})")
    if metric_bqml_challenger <= metric_bqml_blessed:
        best_model = "blessed bqml"
        best_metric = metric_bqml_blessed
        best_metrics.metadata = metrics_bqml_blessed.metadata
    else:
        best_model = "challenger bqml"
        best_metric = metric_bqml_challenger
        best_metrics.metadata = metrics_bqml_challenger.metadata

    thresholds_dict = json.loads(thresholds_dict_str)
    deploy = False

    # Change condition if higher is better.
    if best_metric < thresholds_dict[reference_metric_name]:
        deploy = True

    if deploy:
        deploy_decision = "true"
    else:
        deploy_decision = "false"

    print(f"Which model is best? {best_model}")
    print(f"What metric is being used? {reference_metric_name}")
    print(f"What is the best metric? {best_metric}")
    print(f"What is the threshold to deploy? {thresholds_dict_str}")
    print(f"Deploy decision: {deploy_decision}")

    Outputs = namedtuple(
        "Outputs", ["deploy_decision", "best_model", "metric", "metric_name"]
    )

    return Outputs(
        deploy_decision=deploy_decision,
        best_model=best_model,
        metric=best_metric,
        metric_name=reference_metric_name,
    )
@component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform==1.64.0"])
def validate_infrastructure(
    endpoint: Input[Artifact],
) -> NamedTuple(
    "validate_infrastructure_output", [("instance", str), ("prediction", float)]
):
    import json
    from collections import namedtuple

    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value

    def treat_uri(uri):
        return uri[uri.find("projects/") :]

    def request_prediction(endp, instance):
        instance = json_format.ParseDict(instance, Value())
        instances = [instance]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        response = endp.predict(instances=instances, parameters=parameters)
        print("deployed_model_id:", response.deployed_model_id)
        print("predictions: ", response.predictions)
        return response.predictions[0]
            

    endpoint_uri = endpoint.uri
    treated_uri = treat_uri(endpoint_uri)

    instance = {
        "packet_loss" : 0.6,
        "duration" : 100,
        "jitter" : 1,
        "mean_opinion_score" : 4.5
    }
    instance_json = json.dumps(instance)
    print("Using the following instance: " + instance_json)

    endpoint = aiplatform.Endpoint(treated_uri)
    prediction = request_prediction(endpoint, instance)
    result_tuple = namedtuple(
        "validate_infrastructure_output", ["instance", "prediction"]
    )
    
    return result_tuple(instance=str(instance_json), prediction=float(prediction['mean_squared_error']))


@dsl.pipeline(name='avox-pipeline')
def vertex_ai_pipeline():    
    # Preprocess and normalize data using DataprocPySparkBatchOp
    from google_cloud_pipeline_components.v1.bigquery import (
        BigqueryCreateModelJobOp, BigqueryEvaluateModelJobOp,
        BigqueryExportModelJobOp, BigqueryPredictModelJobOp,
        BigqueryQueryJobOp)
    from google_cloud_pipeline_components.v1.endpoint import (EndpointCreateOp,
                                                              ModelDeployOp)
    from google_cloud_pipeline_components.v1.model import ModelUploadOp
    from google_cloud_pipeline_components.v1.dataproc import DataprocPySparkBatchOp
    
    
    config = get_config(config_gcs_path='gs://avoxi_workshop_bucket/data_pipeline/configuration.yaml')

    preprocess_job = DataprocPySparkBatchOp(
        project=config['project_id'],
        location=config['location'],
        main_python_file_uri=config['main_python_file'],
        #service_account='your-service-account@your-project-id.iam.gserviceaccount.com',
        args=[
            '--input', config['dataproc_args']['input'],
            '--output', config['dataproc_args']['output'],
            '--anomaly_output', config['dataproc_args']['anomaly_output'],
            '--no_anomaly_output', config['dataproc_args']['no_anomaly_output'],
            '--anomaly_normalized_output', config['dataproc_args']['anomaly_normalized_output'],
            '--no_anomaly_normalized_output', config['dataproc_args']['no_anomaly_normalized_output']
        ],
        runtime_config_properties=config['runtime_config_properties'],
        batch_id=config['batch_id'],
        container_image=config['image_uri']).set_caching_options(config['testing'])
    
    # Create BigQuery tables
    import_data = load_bigquery_tables(
        anomaly_data_path=config['dataproc_args']['anomaly_normalized_output'],
        no_anomaly_data_path=config['dataproc_args']['no_anomaly_normalized_output'],
        anomaly_table_name=config['anomaly_table_name'],
        no_anomaly_table_name=config['no_anomaly_table_name']
    ).after(preprocess_job)

    create_train_op = BigqueryCreateModelJobOp(
        project=config['project_id'],
        location=config['bq_location'],
        query=config['create_model_query'].format(**config)
    ).set_caching_options(config['testing']).after(import_data)
    
    bqml_model = load_bigquery_model(project_id=config['project_id'], dataset_id=config['dataset_id'],
                                     model_id=config['model_id'], model_registry=config['model_registry_name'], region=config['location']).after(create_train_op)

    # Evaluate model
    bqml_evaluate_op = bq_components.BigqueryEvaluateModelJobOp(
        project=config['project_id'], 
        location=config['bq_location'],
        model=bqml_model.outputs['model'],
        table_name=f"{config['dataset_id']}.{config['evaluation_table']}",
    ).after(bqml_model)
    
    bqml_eval_metrics_raw = bqml_evaluate_op.outputs["evaluation_metrics"]
    
    # Analyzes evaluation BQML metrics using a custom component.
    interpret_bqml_evaluation_metrics_op = interpret_bqml_evaluation_metrics(
        bqml_evaluation_metrics=bqml_eval_metrics_raw
    ).after(bqml_evaluate_op)
    
    bqml_eval_metrics = interpret_bqml_evaluation_metrics_op.outputs["metrics"]
    
    # Compare metrics
    best_model_task = select_best_model(
        metrics_bqml_challenger=bqml_eval_metrics,
        metrics_bqml_blessed=bqml_eval_metrics,
        thresholds_dict_str=json.dumps(config['model_thresholds']),
        reference_metric_name=config['reference_metric_name']
    ).after(interpret_bqml_evaluation_metrics_op)    
    
    # Deploy model
    with dsl.If(
        ((best_model_task.outputs["deploy_decision"] == "true")),
        name="deploy_decision",
    ):
        # Creates a Vertex AI endpoint using a prebuilt component.
        endpoint_create_op = EndpointCreateOp(
            project=config['project_id'],
            location=config['location'],
            display_name=config['endpoint_name'],
        ).after(best_model_task)
        
        # Deploys the BQML model (now on Vertex AI) to the recently created endpoint using a prebuilt component.
        model_deploy_bqml_op = ModelDeployOp(  # noqa: F841
            endpoint=endpoint_create_op.outputs["endpoint"],
            model=bqml_model.outputs['model'],
            deployed_model_display_name=config['deployed_model_display_name'],
            dedicated_resources_machine_type=config['dedicated_resources_machine_type'],
            dedicated_resources_min_replica_count=config['dedicated_resources_min_replica_count'],
            dedicated_resources_max_replica_count=config['dedicated_resources_max_replica_count'],
        ).set_caching_options(config['testing']).after(endpoint_create_op)

        # Sends an online prediction request to the recently deployed model using a custom component.
        validate_infrastructure(
            endpoint=endpoint_create_op.outputs["endpoint"]
        ).set_caching_options(config['testing']).after(model_deploy_bqml_op)

def run_pipeline(config_file: str = "", project_id : str = 'gurkomal-playground', location : str = 'us-central1', staging_bucket : str = "gs://avoxi_workshop_bucket/staging_pipeline"):
    
    experiment_name="avoxi-experiment-2"
    experiment_description="Avoxi Workshop"
    
    aiplatform.init(
        project=project_id, 
        location=location,
        experiment=experiment_name,
        experiment_description=experiment_description,
        experiment_tensorboard=False,
    )
    
    compiler.Compiler().compile(
        pipeline_func=vertex_ai_pipeline,
        package_path="vertex-ai-pipeline.json"
    )
    
    aiplatform.PipelineJob(
        display_name='vertex-ai-pipeline',
        template_path="vertex-ai-pipeline.json",
        pipeline_root=f"{staging_bucket}/pipeline_root/vertex-ai-pipeline",
        enable_caching=False
    ).submit(
        experiment=experiment_name
    )

if __name__ == '__main__':
    import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python pipeline.py <config_file>")
    #     sys.exit(1)   
    # else:
    #     config_file = sys.argv[1]
        #config = set_config(config_file)
    run_pipeline()
