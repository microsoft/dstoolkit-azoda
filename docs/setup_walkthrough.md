# Walkthrough

## Introduction

This walkthrough aims to set up your own object detection project as quickly and easily as possible. We will cover model training, deployment, testing and labelling integration. We will use a code generated dataset, but you can use the pattern to use your own dataset later.

## Setup

### Setup your Azure DevOps project

Azure DevOps (ADO) is a product for managing projects.

Start a new project [here](https://dev.azure.com) and set it to Private. You may need to create a **New organisation** first if you don't have one yet. The Project name doesn't matter.

### Get the code

Next we need to get the code.

On the left side of ADO, click **Repos**, then the **Import** Button, inside the **Clone URL** field enter in the HTTPS github URL https://github.com/microsoft/dstoolkit-objectdetection-tensorflow-azureml, then the button **Import**.

### Connect to your Azure subscription

Next we will need to connect it to an Azure Subscription to access our Azure resources, this is called a service connection.

First to create a resource group for the service connection:
- Open your [Azure portal](https://portal.azure.com)
- In the Azure search bar type **Resource groups** and select it from the list
- Select **Create**
- Fill in a name for your resource group
- Choose a different region for your resource group, if you want
- **Review + create**
- **Create**

A resource group is a collection of resources, it is used for finding, tracking and deleting groups of resources.

In ADO, select the following:
- **Project Settings** (bottom left)
- **Service connections** (center left)
- **Create service connection**
- **Azure Resource Manager** then **Next** at the bottom of the window
- **Service principal (automatic)** then **Next**,
-  Under Scope level, choose **Subscription**, choose your subscription and the resource group you made earlier, then set Service connection name to **ARMSC** and check **Grant access permission to all pipelines**, then **Save**
- Once complete, select the service connection called **ARMSC**, click **Manage Service Principal**, then copy the **Display name**. Keep this tab open for later.
- Open your [Azure portal](https://portal.azure.com)
- In the Azure search bar type **Subscriptions** and select it from the list
- Select the subscription from the list used above
- Select **Access control (IAM)** from the menu on the left
- Click **+ Add**, then **Add role assignment**
- Select **Contributor** then **Next**
- Select **+Select members** and paste the copied **Display name** from earlier, select it from the list, then click **Select**
- Click **Next**, then **Review + assign**

### Run the Infra-setup pipeline (~3 minutes)

Now we can import pipelines from the connected repo to create resources on our Azure subscription

In ADO, select the following:
- **Pipelines**
- **Create Pipeline** (**New pipeline** if one already exists)
- **Azure Repos Git**
- Select your project
- **Existing Azure Pipelines YAML file**
- Select the intended branch
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
- Click the lock icon next to passwords, keys and secrets to set the variable type to secret
- Select **Save**

### Submit a model training job (~25 minutes)

Now we can train a first model, it will be with a generated dataset. Later you can replace it with your own.

Repeat the same process as above to start a pipeline, but set the path to **/azure-pipelines/remote-train.yml**. Each of these pipelines using the variable group will need permission. After starting the pipeline, select the **Job**, click **View** at the top right, then **Permit**, then **Permit** again.

This will train a model and store the weights in AML datastore under **yolov5_models**. Each model will be stored in its own folder named by its timestamp.

To explore the datastore, select **Datastores** in your AML Studio resource, then **workspaceblobstore (Default)**, then **Browse (preview)**. Otherwise you can download **Azure Storage Explorer** which has a drag and drop interface for navigating datastores.

### Submit a model inference job (~10 minutes)

Once the model training is complete, we can perform inference on a directory of images.

Repeat the same process as above to start a pipeline, but set the path to **/azure-pipelines/remote-infer.yml**. After starting the pipeline, click **View** at the top right, then **Permit**, then **Permit** again.

This will perform inference on all the datasets images. You can change the directory in the script **start_yolo_aml_run.py**. The results will be stored in **yolov5_inferences** in the datastore.

### Submit a model test job (~20 minutes)

Once the model training is complete, we can test the model on a dataset.

Repeat the same process as above to start a pipeline, but set the path to **/azure-pipelines/remote-test.yml**. After starting the pipeline, click **View** at the top right, then **Permit**, then **Permit** again.

This will evaluate the model on the test labels. You can change adjust the tested dataset in the configuration set in **test_coordinator.py**. The results will be stored in **yolov5_tests** in the datastore.

### Deploy the model (~15 minutes)

After the model has been trained, we can deploy an inference webservice which return the inference results given an image.

Repeat the same process as above to start a pipeline, but set the path to **/azure-pipelines/deploy.yml**. After starting the pipeline, click **View** at the top right, then **Permit**, then **Permit** again.

Once the deployment is complete, you can fetch the endpoint name and key:
- Go to your [Azure portal](https://portal.azure.com)
- Search and select **Machine learning** from the search bar
- Select the one you created from the list (the default is called azoda-amlw, in resource group azoda-rg)
- **Launch studio**
- **Endpoints** on the left
- Select your endpoint
- **Consume**

You can use **consume_endpoint.py** as an example to use the endpoint. 

### Use your own dataset

To use your own dataset add another folder with the same structure as **synthetic_dataset** in the datastore. If you don't have annotations, you can just put in the images folder and use the pixie pipelines to get annotations.

### Other notes

You can rename pipelines by three dots on right side of the pipeline in the main Pipielines view, then select **Rename/move**.

### Extensions

Feel free to add more pipelines and make a pull request, we are planning to add more soon.