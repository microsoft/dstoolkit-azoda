# Walkthrough

## Introduction

This walkthrough aims to setup your own object detection project as quickly and easily as possible. We will cover model training, deployment, testing and labelling integration. We will use a code generated dataset, but you can use the pattern to use your own dataset later.

## Setup

### Setup your Azure DevOps project

Azure DevOps (ADO) is a product for managing projects.

Start a new project [here](https://dev.azure.com) and set it to Private. The Project name doesn't matter.

### Get the code

Next we need to get the code.

On the left side of ADO, click **Repos**, then the **Import** Button, inside the **Clone URL** field enter in the github URL https://github.com/microsoft/dstoolkit-objectdetection-tensorflow-azureml, then the button **Import**.

### Connect to your Azure subscription

Next we will need to connect it to an Azure Subscription to access our Azure resources, this is called a service connection.

In ADO, select the following:
- **Project Settings**
- **Service connections**
- **Create service connection**
- **Azure Resource Manager** then **Next** at the bottom of the window
- **Service principal (automatic)** then **Next**,
-  Under Scope level, choose **Subscription**, choose your subscription and a Resource group, then set Service connection name to **ARMSC** and check **Grant access permission to all pipelines**, then **Save**
- Once complete, select the service connection called **ARMSC**, click **Manage Service Principal**, then copy the **Display name**
- Open your [Azure portal](https://portal.azure.com)
- In the Azure search bar type Subscriptions and select **Subscriptions** from the listed Services
- Select the subscription from the list used above
- Select **Access control (IAM)** from the menu on the left
- Click **+ Add**, then **Add role assignment**
- Select **Contributor** then **Next**
- Select **+Select members** and paste the copied **Display name** from earlier, select it from the list, then click **Select**
- Click **Next**, then **Review + assign**

### Run the Infra-setup pipeline (~9 minutes)

Now we can import pipelines from the connected repo to create resources on our Azure subscription

In ADO, select the following:
- **Pipelines**
- **Create Pipeline** (**New pipeline** if one already exists)
- **Azure Repos Git**
- Select your project
- **Existing Azure Pipelines YAML file**
- Under **Branch** select **feature/auto_setup**
- Under **Path** select **/azure-pipelines/PIPELINE-auto-setup.yml** then **Continue**
- **Run**

### Submit a model training job (~2 minutes)

Now we can train a first model, it was be with a generated dataset. Later you can replace it with your own.

Repeat the same process as above, but set the path to **/azure-pipelines/PIPELINE-auto-training.yml**. This pipeline will ask you to authenicate interactively about one minute after the pipeline starts. Watch the pipeline output to see it. The default training config uses very few steps, so it shoud take about 5 minutes to finish.

### Deploy the model (~9 minutes)

Repeat the same process as above, but set the path to **/azure-pipelines/PIPELINE-auto-deployment.yml**. This will also need interactive authentication after about a minute. Make sure the model has finished training before deploying. 

### Test the deployment (~4 minutes)

Repeat the same process as above, but set the path to **/azure-pipelines/PIPELINE-auto-testing.yml**. This will require some variables to be set before pressing Run. 

First you will need the endpoint name and key:
- Go to your [Azure portal](https://portal.azure.com)
- Search and select **Machine learning** from the search bar
- Select the one you created from the list (the default is called tfod-dev-amlw, in resource group tfod-dev-rg-demo)
- **Launch studio**
- **Endpoints** on the left
- **syntheticdataset** (the name of the dataset)
- **Consume**

Here you can see the **REST endpoint** and **Primary key**. In the next step, we will give these to the pipeline.

On the Edit page of the pipeline, select the following:
- **Variables**
- **New variable**
- In the **Name** field enter **dataset**, in the **Value** field enter **synthetic_dataset**, then **OK**
- When entering both \<REST endpoint\> and \<Primary key\>, surround them with double quotes **" "** so that the characters don't lead to misinterpretation. Also remember to check the secret option, so that the information is not visible.
- **+** , then as above, **Name**: **endpoint**, **Value**: \<"REST endpoint"\>, select **Keep this value secret**, then **OK**
- **+** , then as above, **Name**: **key**, **Value**: \<"Primary key"\>, select **Keep this value secret**, then **OK**
- **Save**
- **Run**

Once the pipeline is done, the confusion matrix will be shown in the logs.

### Export labels to Pixie (~6 minutes)

Pixie is an interactive labelling tool that has built-in model training to accelerate the labelling process.

Repeat the pipeline setup process with the path **/azure-pipelines/PIPELINE-auto-pixie-upload.yml**. Replace \<"Pixie endpoint"\> and \<"Pixie key"\> with your Pixie connection information. This will also require variables added before running, as above:

- **Name**: **PIXIE_API**, **Value**: \<"Pixie endpoint"\>, select **Keep this value secret**, then **OK**
- **Name**: **PIXIE_KEY**, **Value**: \<"Pixie key"\>, select **Keep this value secret**, then **OK**
- **Name**: **dataset**, **Value**: **synthetic_dataset**, then **OK**

You will need to get pixie authentication or start your own deployment. More info on this soon.

### Import labels from Pixie (~5 minutes)

After adding more labels Pixie, you can run this pipeline to load the new labels to your storage account.

Repeat the pipeline setup process with the path **/azure-pipelines/PIPELINE-auto-pixie-import-labels.yml** with the following variables:

- **Name**: **PIXIE_API**, **Value**: \<"Pixie endpoint"\>, select **Keep this value secret**, then **OK**
- **Name**: **PIXIE_KEY**, **Value**: \<"Pixie key"\>, select **Keep this value secret**, then **OK**
- **Name**: **dataset**, **Value**: **synthetic_dataset**, then **OK**
- **Name**: **project_id**, **Value**: **<project_id>**, then **OK** (usually five letters)

### Import inferences from Pixie (~3.5 minutes)

After inferencing with Pixie, you can run this pipeline to load the inferences to your storage account.

Repeat the pipeline setup process with the path **/azure-pipelines/PIPELINE-auto-pixie-import-inferences.yml** with the following variables:

- **Name**: **PIXIE_API**, **Value**: \<"Pixie endpoint"\>, select **Keep this value secret**, then **OK**
- **Name**: **PIXIE_KEY**, **Value**: \<"Pixie key"\>, select **Keep this value secret**, then **OK**
- **Name**: **dataset**, **Value**: **synthetic_dataset**, then **OK**
- **Name**: **project_id**, **Value**: **<project_id>**, then **OK** (usually five letters)
- **Name**: **model_id**, **Value**: **<model_id>**, then **OK** (visible from Models tab)
