## Pixie 

Pixie is a tool developed by Microsoft Research to assist with labelling. It has a variety of features to accelerate the labelling process. The features include ingesting different types of data, training models and sorting and filtering images based on model results. This repo provides three pipelines to interact with this platform. 

In order to interact with Pixie you will need an endpoint and key. You will need to deploy your own or get the details of an existing deployment. More details to follow in the next update.

### Export labels to Pixie

Repeat the pipeline setup process with the path **/azure-pipelines/pixie-export-dataset.yml**. Add the following variables to your variable group:

- **pixie_api**, **Value**: \<"Pixie endpoint"\>, select **Keep this value secret**, then **OK**
- **Name**: **PIXIE_KEY**, **Value**: \<"Pixie key"\>, select **Keep this value secret**, then **OK**
- **Name**: **dataset**, **Value**: **synthetic_dataset**, then **OK**

You will need to get pixie authentication or start your own deployment. More info on this soon.

### Import labels from Pixie

After adding more labels Pixie, you can run this pipeline to load the new labels to your storage account.

Repeat the pipeline setup process with the path **/azure-pipelines/PIPELINE-auto-pixie-import-labels.yml** with the following variables:

- **Name**: **PIXIE_API**, **Value**: \<"Pixie endpoint"\>, select **Keep this value secret**, then **OK**
- **Name**: **PIXIE_KEY**, **Value**: \<"Pixie key"\>, select **Keep this value secret**, then **OK**
- **Name**: **dataset**, **Value**: **synthetic_dataset**, then **OK**
- **Name**: **project_id**, **Value**: **<project_id>**, then **OK** (usually five letters)

### Import inferences from Pixie

After inferencing with Pixie, you can run this pipeline to load the inferences to your storage account.

Repeat the pipeline setup process with the path **/azure-pipelines/PIPELINE-auto-pixie-import-inferences.yml** with the following variables:

- **Name**: **PIXIE_API**, **Value**: \<"Pixie endpoint"\>, select **Keep this value secret**, then **OK**
- **Name**: **PIXIE_KEY**, **Value**: \<"Pixie key"\>, select **Keep this value secret**, then **OK**
- **Name**: **dataset**, **Value**: **synthetic_dataset**, then **OK**
- **Name**: **project_id**, **Value**: **<project_id>**, then **OK** (usually five letters)
- **Name**: **model_id**, **Value**: **<model_id>**, then **OK** (visible from Models tab)
