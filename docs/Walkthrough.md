# Walkthrough

## Introduction

This walkthrough aims to setup your own object detection project as quickly and easily as possible. We will cover model training, deployment, testing and labelling integration. We will use a code generated dataset, but you can use the pattern to use your own dataset later.

## Setup

### Setup your Azure DevOps project

Azure DevOps (ADO) is a product for managing projects.

Start a new project [here](https://dev.azure.com) and set it to Private. The Project name doesn't matter.

### Get the code

Next we need to get the code.

On the left side of ADO, click **Repos**, then the **Import** Button, inside the **Clone URL** field enter in the HTTPS github URL https://github.com/microsoft/dstoolkit-objectdetection-tensorflow-azureml, then the button **Import**.

### Connect to your Azure subscription

Next we will need to connect it to an Azure Subscription to access our Azure resources, this is called a service connection.

In ADO, select the following:
- **Project Settings**
- **Service connections**
- **Create service connection**
- **Azure Resource Manager** then **Next** at the bottom of the window
- **Service principal (automatic)** then **Next**,
-  Under Scope level, choose **Subscription**, choose your subscription and a Resource group, then set Service connection name to **ARMSC** and check **Grant access permission to all pipelines**, then **Save**
- Once complete, select the service connection called **ARMSC**, click **Manage Service Principal**, then copy the **Display name**. Keep this tab open for later.
- Open your [Azure portal](https://portal.azure.com)
- In the Azure search bar type Subscriptions and select **Subscriptions** from the listed Services
- Select the subscription from the list used above
- Select **Access control (IAM)** from the menu on the left
- Click **+ Add**, then **Add role assignment**
- Select **Contributor** then **Next**
- Select **+Select members** and paste the copied **Display name** from earlier, select it from the list, then click **Select**
- Click **Next**, then **Review + assign**

### Run the Infra-setup pipeline

Now we can import pipelines from the connected repo to create resources on our Azure subscription

In ADO, select the following:
- **Pipelines**
- **Create Pipeline** (**New pipeline** if one already exists)
- **Azure Repos Git**
- Select your project
- **Existing Azure Pipelines YAML file**
- Under **Branch** select **feature/yolo**
- Under **Path** select **/azure-pipelines/setup.yml** then **Continue**
- **Run**

### Get service principal id and password

A service principal acts like a user account that the code can use to authenticate. We will use this to authenticate instead of using our own accounts to interactively authenticate.

In the open service principal tab from above:
- Store the **Application (client) ID** value
- Select **Certificates & secrets**
- **+New client secret**
- Change the **Expires** options if you would like the password to last something other than 6 months
- **Add**
- Store the new **Value**, not the Secret ID.

This is the username and password which we will store in the variable group next.
 
### Setup a variable group

A variable group is a collection of variables which can be used across multiple pipelines. We will store the dataset and service principal information here.

In ADO, select the following:
- **Pipelines**
- **Library**
- **+Variable group**
- Under Variable group name: **vars**
- Below under Variables, select **+Add**
- Under name enter: **dataset**, under value: **synthetic_dataset**
- Select **+Add**
- Under name enter: **service_principal_id**, under value: **Application (client) ID from above**
- Select **+Add**
- Under name enter: **service_principal_password**, under value: **Application (client) Secret Value from above**

### Submit a model training job

Now we can train a first model, it will be with a generated dataset. Later you can replace it with your own.

Repeat the same process as above, but set the path to **/azure-pipelines/remote-train.yml**. Each of these pipelines using the variable group will need permission. After starting the pipeline, click **View** at the top right, then **Permit**, then **Permit** again.

This will train a model and store the weights in AML datastore under yolov5_models. Each model will be stored in its own folder named by its timestamp.

To explore the datastore, select **Datastores** in your AML Studio resource, then **workspaceblobstore (Default)**, then **Browse (preview)**. Otherwise you can download **Azure Storage Explorer** which has a drag and drop interface for navigating datastores.

### Submit a model training job 

### Deploy the model 

Repeat the same process as above, but set the path to **/azure-pipelines/PIPELINE-auto-deployment.yml**. This will also need interactive authentication after about a minute. Make sure the model has finished training before deploying. 

### Test the deployment

Repeat the same process as above, but set the path to **/azure-pipelines/PIPELINE-auto-testing.yml**. This will require some variables to be set before pressing Run. 

First you will need the endpoint name and key:
- Go to your [Azure portal](https://portal.azure.com)
- Search and select **Machine learning** from the search bar
- Select the one you created from the list (the default is called azoda-amlw, in resource group azoda-rg)
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
