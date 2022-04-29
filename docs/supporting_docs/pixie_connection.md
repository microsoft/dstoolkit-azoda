## Pixie 

Pixie is a tool developed by Microsoft Research to assist with the 

### Export labels to Pixie

Pixie is an interactive labelling tool that has built-in model training to accelerate the labelling process.

Repeat the pipeline setup process with the path **/azure-pipelines/PIPELINE-auto-pixie-upload.yml**. Replace \<"Pixie endpoint"\> and \<"Pixie key"\> with your Pixie connection information. This will also require variables added before running, as above:

- **Name**: **PIXIE_API**, **Value**: \<"Pixie endpoint"\>, select **Keep this value secret**, then **OK**
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
