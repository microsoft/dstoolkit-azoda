organization_name="<input organization>"
project_name="<input project name>"
subscription_name="<Subscription name>"

az login &&
subscription_id=$(az account list --query "[?isDefault].id | [0]" | jq -r .) &&
sp_name=azoda_sp &&
sc_name=ARMSC &&
az devops configure --defaults organization=https://dev.azure.com/$organization_name project=$project_name &&
sp_details=$(az ad sp create-for-rbac --name $sp_name --role Contributor --scopes /subscriptions/$subscription_id) &&
export AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_ID=$(echo $sp_details | jq -r ".appId") &&
export AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_KEY=$(echo $sp_details | jq -r ".password") &&
export AZURE_DEVOPS_EXT_AZURE_RM_TENANT_ID=$(echo $sp_details | jq -r ".tenant") &&
az devops service-endpoint azurerm create --azure-rm-service-principal-id $AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_ID --azure-rm-subscription-id $subscription_id --azure-rm-subscription-name "$subscription_name" --azure-rm-tenant-id $AZURE_DEVOPS_EXT_AZURE_RM_TENANT_ID --name $sc_name &&
endpoint_id=$(az devops service-endpoint list --query "[].{id:id, name:name} | [? contains(name, '$sc_name')]".id | jq -r ".[0]") &&
az devops service-endpoint update --id $endpoint_id --enable-for-all &&
vargroup_id=$(az pipelines variable-group create --name vars --variables dataset=synthetic_dataset service_principal_id=$AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_ID --authorize true | jq -r ".id") &&
az pipelines variable-group variable create --group-id $vargroup_id --name service_principal_password --secret true --value $AZURE_DEVOPS_EXT_AZURE_RM_SERVICE_PRINCIPAL_KEY &&
az pipelines variable-group variable create --group-id $vargroup_id --name vargroup_id --secret false --value $vargroup_id &&
az pipelines variable-group variable create --group-id $vargroup_id --name organization_name --secret false --value $organization_name &&
az pipelines variable-group variable create --group-id $vargroup_id --name project_name --secret false --value $project_name
