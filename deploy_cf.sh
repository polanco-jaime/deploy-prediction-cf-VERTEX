#!/bin/bash

#####################################################################################################
# Script Name: deploy_cf.sh
# Date of Creation: 05/18/2022
# Author: Jaime Polanco
# Updated: 04/18/2022
#####################################################################################################

source ./config.sh

project_id=${PROJECT_ID}
cf_causal_salida_pred="causal_salida_pred-gen2"



echo "Deploying causal salida"
cd ~/earth-engine-on-bigquery/src/cloud-functions/causal_salida_pred

gcloud functions deploy ${cf_hansen} --entry-point causal_salida --runtime python39 --trigger-http --allow-unauthenticated --set-env-vars SERVICE_ACCOUNT=${ee_sa} --project ${project_id} --service-account ${ee_sa} --gen2 --region ${REGION} --run-service-account ${ee_sa}  --memory 16GB --timeout 500s 



endpoint_causal_salida_pred=$(gcloud functions describe ${cf_causal_salida_pred} --region=${REGION} --gen2 --format=json | jq -r '.serviceConfig.uri')
 

bq mk -d udfs_eda


build_sql="CREATE OR REPLACE FUNCTION udfs_eda.predict_cf_causal_salida_pred(text STRING )  RETURNS STRING REMOTE WITH CONNECTION \`${project_id}.us.gcf-ee-conn\` OPTIONS ( endpoint = '${endpoint_hansen}')"
