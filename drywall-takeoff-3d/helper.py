import json

from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel


def load_vertex_ai_client(credentials, region="us-central1"):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = GenerativeModel(credentials["VertexAI"]["llm"]["model_name"])
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config

def bigquery_run(credentials, GBQ_query, job_config=dict()):
    bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
    job_config = bigquery.QueryJobConfig(
        destination_encryption_configuration=bigquery.EncryptionConfiguration(
            kms_key_name=credentials["GBQServer"]["KMS_key"]
        ),
        **job_config
    )
    query_output = bigquery_client.query(GBQ_query, job_config=job_config)
    return query_output