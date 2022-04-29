## Pixie 

Pixie is a tool developed by Microsoft Research to assist with labelling. It has a variety of features to accelerate the labelling process. The features include ingesting different types of data, training models and sorting and filtering images based on model results. This repo provides three pipelines to interact with this platform. 

In order to interact with Pixie you will need an endpoint and key. You will need to deploy your own or get the details of an existing deployment. More details to follow in the next update.

### Export labels to Pixie

Repeat the pipeline setup process with the path **/azure-pipelines/pixie-export-dataset.yml**. Add the following variables to your variable group:

- **pixie_api**: **Your pixie api**
- **pixie_key**, **Your pixie key**
- **dataset**, **synthetic_dataset** or the name of your dataset. This will be the name of your pixie project.

### Import labels from Pixie

After adding labels on Pixie, you can run this pipeline to load the new labels to your storage account.

Repeat the pipeline setup process with the path **/azure-pipelines/pixie-import-labels.yml**, make sure the following variables are in your variable group:

- **pixie_api**: **Your pixie api**
- **pixie_key**, **Your pixie key**
- **dataset**, **synthetic_dataset** or the name of your dataset. This will be the name of your pixie project.
- **project_id**, **your five letter project id**, shown next to your dataset name on the Pixie Projects page.

This will import the inferences to your storage account under the name **inferences_pixie.csv**.

### Import inferences from Pixie

After inferencing with Pixie, you can run this pipeline to load the inferences to your storage account.

Repeat the pipeline setup process with the path **/azure-pipelines/pixie-import-inferences.yml**, make sure the following variables are in your variable group:

- **pixie_api**: **Your pixie api**
- **pixie_key**, **Your pixie key**
- **dataset**, **synthetic_dataset** or the name of your dataset. This will be the name of your pixie project.
- **project_id**, **your five letter project id**, shown next to your dataset name on the Pixie Projects page.
- **project_id**, **your model id which generated the inferences**, visible from the Pixie Models page.

This will import the labels to your storage account under the name **labels_pixie.csv**.

### Other notes

You might get unexpected errors in Pixie if you try to train with less than 20 images. This is being investigated.