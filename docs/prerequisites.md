# Prerequisites

## Correct Versions

It is important that have at least the versions azure-cli 2.41.0 and azure-devops 0.25.0. To check this run `az --version`. If your versions are below the required versions also run `sudo apt-get update && sudo apt-get upgrade` before continuing to update your system.

## Azure CLI

Azure CLI is a command-line tool to execute commands related to Azure resources. On Ubuntu/Debian simply use:

``` curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash ```

Otherwise follow the installation instructions [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)

## Azure DevOps CLI

Azure DevOps CLI is an extension to Azure CLI to manage Azure DevOps Services. To install simply use:

``` az extension add --name azure-devops ```

For more information, check [here](https://learn.microsoft.com/en-us/azure/devops/cli/?view=azure-devops)

## jq

jq is a lightweight JSON processor. On Ubuntu/Debian simply use 

``` sudo apt-get install jq ```

Otherwise follow the installation instructions [here](https://stedolan.github.io/jq/download/)