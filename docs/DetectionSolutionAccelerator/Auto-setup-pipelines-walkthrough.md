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

### Run the Infra-setup pipeline

Now we can import pipelines from the connected repo to create resources on our Azure subscription

In ADO, select the following:
- **Pipelines**
- **Create Pipeline**
- **Azure Repos Git**
- Select your project
- **Existing Azure Pipelines YAML file**
- Under **Branch** select **feature/auto_setup**
- Under **Path** select **/azure-pipelines/PIPELINE-auto-setup.yml** then **Continue**
