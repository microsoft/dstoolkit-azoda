### Manual Azure subscription

If you do not have a bash terminal or facing other issues, you can try setting up the connection step by step.

We will need to connect it to an Azure Subscription to access our Azure resources, this is called a service connection.

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
