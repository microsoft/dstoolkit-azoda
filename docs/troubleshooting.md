# Troubleshooting

## Connect to your Azure Subscription

### Read -p doesn't work

This is a common issue with zsh, this command stores input from the user in a variable. To fix, enter these values manually in the script, e.g.

Change: 
` read -p "Enter organization name: " organization_name ` \
to:
` organization_name="<Your organization name>" `

### Requested to enter SPN key manually

This is due to an outdated version of the Azure CLI. Please [update](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) and restart the terminal. The repo was tested on version 2.41.0.

### Subscription not found

This could be because the subscription is not in the tenant which is loaded by default. In this case, you will need to specify the precise tenant, by replacing the az login command with:

``` az login --tenant <your tenant id>```

You can find your tenant id by searching for Tenant properties in your Azure Portal.

### Insufficient privileges

If you receive an error about privileges, make sure that you have sufficient rights on the chosen subscription to make resources.

### Script doesn't work

If there isn't an easy fix to run the script, you can perform al the steps manually. These steps are described [here](manual_connection_process.md).

### Azure-devops won't update to the right version

First remove the extension: `az extension remove --name azure-devops` then add it again: `az extension add --name azure-devops`.