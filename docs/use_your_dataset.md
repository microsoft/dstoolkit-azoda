### Use your own dataset

To use your own dataset add another folder with the same structure as **synthetic_dataset** in the datastore. You can download **Microsoft Azure Storage Explorer** to interact with the datastore. You will find the project files under **<Subscription Name> -> Storage Accounts -> azoda\* -> Blob Containers -> azureml-blobstore-\***.

In Azure DevOps, go to **Pipelines -> Library -> vars**. Then change the **dataset** variable to the new dataset folder name. Then **Save** at the top of the window. All the pipelines will now work with the new specified dataset.

If you don't have annotations, you can just place the images folder in the datastore and use the Custom Vision pipelines:
- azure-pipelines/cv-export-dataset.yml to move the images from the datastore to Custom Vision
- azure-pipelines/cv-import-labels.yml to move the labels from Custom Vision to the data store.

The free custom vision account only permits 2 projects at a time. If you need more you will either have to delete a different project on the account or upgrade to a paid version.