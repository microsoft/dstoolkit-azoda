### Delete Project Resources

All the project resources can be deleted from the Azure Portal interface. By default all the resources will be in the resource group ***azoda-rg***. Deleting this will remove all the project files.

To delete with code instead, use the following script:

```
az keyvault delete --name azoda-kv --resource-group azoda-rg
az ad sp delete --id <service principal id>
az group delete --name azoda-rg --yes
az keyvault purge --name azoda-kv
az cognitiveservices account purge --location westeurope --resource-group azoda-rg --name azodacv
```
