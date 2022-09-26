### Use your own dataset

To use your own dataset add another folder with the same structure as **synthetic_dataset** in the datastore. You can download **Microsoft Azure Storage Explorer** to interact with the datastore. You will find the project files under **<Subscription Name> -> Storage Accounts -> azoda\* -> Blob Containers -> azureml-blobstore-\***. If you don't have annotations, you can just put in the images folder and use the Custom Vision pipelines to get annotations. 

In Azure DevOps, go to **Pipelines -> Library -> vars**. Then change the **dataset** variable to the new dataset folder name. Then **Save** at the top of the window.

The free custom vision account only permits 2 projects at a time. If you need more you will either have to delete a different project on the account or upgrade to a paid version.

Next import and run the **export_dataset.yml** pipeline.
