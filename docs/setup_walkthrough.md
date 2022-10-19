# Walkthrough

## Introduction

This walkthrough aims to set up your own object detection project as quickly and easily as possible. We will cover model training, deployment, testing and labelling integration. We will use a code generated dataset, but you can later replace the starter dataset with your own. The setup consists of the following steps:
- Setup your Azure DevOps project
- Connect to your Azure Subscription
- Import the code
- Run the starter pipeline

## Setup

### Setup your Azure DevOps project

Azure DevOps (ADO) is a product for managing projects.

Start a new project [here](https://dev.azure.com) and set it to Private. You may need to create a **New organisation** first if you don't have one yet. The Project name doesn't matter.

### Connect to your Azure Subscription

In a bash terminal, run the script below this paragraph and answer the three prompts. The organisation and project names can be seen in the AzureDevOps URL in the format: https://dev.azure.com/<organization_name>/<project_name>/.../.
To determine your subscription name, go to your [Azure portal](https://portal.azure.com) and search for **Subscriptions**. Here you will see of subscription names that you have access to, choose the one you want to use. The script will then take you to a webpage for authentication in order to connect your Azure DevOps account to your Azure subscription and store relevant output variables to the DevOps project.

```
read -p "Enter organization name: " organization_name &&
read -p "Enter project name: " project_name &&
read -p "Enter subscription name: " subscription_name &&
az login &&
subscription_id=$(az account list --query "[?name=='$subscription_name'].id | [0]" | jq . -r) &&
sp_name=azoda_sp_$subscription_id &&
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
az pipelines variable-group variable create --group-id $vargroup_id --name organization_name --secret false --value "$organization_name" &&
az pipelines variable-group variable create --group-id $vargroup_id --name project_name --secret false --value "$project_name" &&
az pipelines variable-group variable create --group-id $vargroup_id --name subscription_name --secret false --value "$subscription_name"

```

If you get any errors that state the jq cannot be found, install it with this bash command: ` sudo apt-get install jq `

### Import the code

Next we need to get the code.

On the left side of ADO, click **Repos**, then the **Import** Button, inside the **Clone URL** field enter in the HTTPS github URL https://github.com/microsoft/dstoolkit-azoda, while this repo is in Private mode you will need to check the **Requires Authentication** box, then enter the Username and PAT fields (from the owner Daniel Baumann), then finally the button **Import**.

### Run the starter pipeline

Now we can import and run pipelines from the connected repo to create resources on the Azure subscription. This will run the demo pipeline that does all the steps combined from setup to deployment.

In ADO, select the following:
- **Pipelines**
- **Create Pipeline** (**New pipeline** if one already exists)
- **Azure Repos Git**
- Select your project
- **Existing Azure Pipelines YAML file**
- Under **Path** select **/azure-pipelines/demo.yml** then **Continue**
- **Run**

That's it, you've set up the project. The pipeline should take about an hour to complete all the steps.

### Next Steps

To use the endpoint, follow the steps [here](use_endpoint.md).

To use your own dataset, follow the steps [here](use_your_dataset.md).

To clear up all the resources created, follow the steps [here](delete_project_resources.md).

Feel free to add more pipelines and make a pull request, we are planning to add more soon.
