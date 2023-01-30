### Delete Project Resources

To delete the project resources, use the following script:

```
subscription_id=<enter your subscription id> &&
spn_id=<enter your service principal id> &&
az login &&
keyvault_name="azoda-kv-${subscription_id:0:12}" &&
az keyvault delete --name $keyvault_name --resource-group azoda-rg &&
az ad sp delete --id $spn_id &&
az group delete --name azoda-rg --yes &&
az keyvault purge --name $keyvault_name &&
az cognitiveservices account purge --location westeurope --resource-group azoda-rg --name azodacv
```

The **service principal id** can be found in the variable group in your Azure DevOps project under **Pipelines** -> **Library** -> **vars**.

Alternatively, all the project resources can be deleted from the Azure Portal interface. By default all the resources will be in the resource group ***azoda-rg***. Deleting this will remove all the project files. The keyvault and cognitive services account are often 'soft-deleted' which means that they can be recovered after deletion, these will need to be purged so that the variable names can be reused.

The most expensive resource that you should make sure to delete is the aks resource (the compute behind the deployment). This will a resource group in resource groups section of your chosen subscription with a name of the form **MC_azoda-rg_aks-auto\<12 generated characters\>_\<location\>**. This should be deleted with the commands above, but useful to delete this separately if you want to keep the rest of the project.