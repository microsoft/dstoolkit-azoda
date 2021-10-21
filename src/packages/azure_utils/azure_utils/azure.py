''' Azure Utils
Azure module to bundle workspace login and function together for reuse
Workspace is set within the config.json file in this directory
'''

import os
import sys
import json

from azureml.core.authentication import AzureCliAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Datastore, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget


# we will hardcode the foler but not the file
__here__ = os.path.dirname(__file__)
env_dir = os.path.join(__here__, '..', '..', '..', '..', 'configuration')


def load_config(config_path):
    """
    Reads a .json config file and returns as a dict
    :return: config (dict)
    """
    with open(config_path) as f:
        config = json.load(f)
    return config


class AML():
    def __init__(self, config_file=None,
                 sub=None, ws=None, rg=None, tenant=None, auth=None,
                 sp_id=None, sp_pw=None,
                 cli_auth=False):
        """
        base AML class containing workspace connection and data/compute
        functions and contains the following attributes:
        config - if config file provided then may avliable as dict
        ws - AML workspace object
        kv - AMl default keyvault object
        data_stores - list of registered AML datastore references
        :param config_file: file namd with ext located in the econfiguration
        folder in the repo
        :param sub: subscription ID as string
        :param ws: workspace name as string
        :param rg: resource group name as string
        :param tennant: tennent id, avliable from portal under AAD
        :param auth: optional auth type such as service principle, if None
        defaults to interactive auth
        :param sp_id: service principle id as string
        :param pw: service principle password as string
        :param cli_auth: bool for if using cli auth
        """

        if all(v is None for v in [sub, ws, rg, tenant]):
            if config_file is not None:
                config_path = os.path.join(env_dir, config_file)
                self.config = load_config(config_path)
                self.ws_tenant = self.config['AML_TENANT_ID']
                self.ws_sub = self.config['AML_SUBSCRIPTION_ID']
                self.ws_rg = self.config['AML_RESOURCE_GROUP']
                self.ws_name = self.config['AML_WORKSPACE_NAME']
            else:
                print('ERROR: No config or workspace arguments given')
                sys.exit(-1)
        else:
            self.ws_tenant = tenant
            self.ws_sub = sub
            self.ws_rg = rg
            self.ws_name = ws

        if cli_auth is True:
            auth = AzureCliAuthentication()

        elif sp_id is not None and sp_pw is not None:
            auth = ServicePrincipalAuthentication(
                tenant_id=tenant,
                service_principal_id=sp_id,
                service_principal_password=sp_pw)

        self.ws = self.load_ws(auth)
        self.kv = self.ws.get_default_keyvault()
        self.data_stores = self.list_datastores()

    def load_ws(self, auth=None):
        """
        Loads the AML workspace based on the config paramters
        :param auth: optional auth type such as service principle, if None
        defaults to interactive auth
        :return: aml workspace object
        """
        if auth is None:
            auth = InteractiveLoginAuthentication(
                tenant_id=self.ws_tenant
                )

        ws = Workspace(subscription_id=self.ws_sub,
                       resource_group=self.ws_rg,
                       workspace_name=self.ws_name,
                       auth=auth)
        return ws

    def list_datastores(self):
        """
        Lists datastores references registered to AML workspace
        :return: list of datasore refs, [ref_1, ref_2, ref_3]
        """
        datastores = self.ws.datastores
        ds_list = [name for name, datastore in datastores.items()]
        return ds_list

    def get_datastore(self, name):
        """
        gets a specific datastore from datastore references
        :param name: name of datastore reference in AML
        :return: AML datastore object
        """
        data_store = Datastore.get(self.ws, datastore_name=name)
        return data_store

    def add_key(self, name, key):
        """
        Creates a new secret in the AML default keyvault
        :param name: name of secret
        :param key: key to store in secret
        """
        self.kv.set_secret(name=name,
                           value=key)

    def get_key(self, name):
        """
        gets a key based on a secret stored in the AML keyvault
        :param name: name of secret to retrieve key from
        :return: key as string
        """
        key = self.kv.get_secret(name)
        return key

    def add_datastore(self, account, container=None, fileshare=None):
        """
        Add a new datastore reference to the AML workspace,
        can be either container or fileshare
        :param account: Azure storage account name
        :param container: the name of the storage account container to register
        :param fileshare: the name of the fileshare to register
        """
        key = self.get_key(account)

        if container is not None:
            store_ref = "{}_{}".format(account, container)
            Datastore.register_azure_blob_container(workspace=self.ws,
                                                    datastore_name=store_ref,
                                                    container_name=container,
                                                    account_name=account,
                                                    account_key=key)

        if fileshare is not None:
            store_ref = "{}_{}".format(account, fileshare)
            Datastore.register_azure_file_share(workspace=self.ws,
                                                datastore_name=store_ref,
                                                file_share_name=fileshare,
                                                account_name=account,
                                                account_key=key,
                                                create_if_not_exists=False)

    def download_datastore(self, name, output_path, prefix=None):
        """
        download a copy locally of the data from a datastore
        :param name: registered datastore name
        :param output_path: local storage path to save the data to
        :param prefix: file/folder prefix to filter what data to download
        """
        datastore = self.get_datastore(name)
        datastore.download(target_path=output_path,
                           prefix=prefix,
                           show_progress=True)

    def create_compute(self,
                       cluster_name,
                       vm_type='STANDARD_NC6',
                       node_count=1):
        """
        create AML training compute cluster
        :param cluster_name: cluster name
        :param vm_type: azure vm type, defaults to 'standard_NC6'
        :param node_count: number of compute nodes
        """
        compute_config = (AmlCompute
                          .provisioning_configuration(vm_size=vm_type,
                                                      max_nodes=node_count))
        compute = ComputeTarget.create(self.ws, cluster_name, compute_config)
        compute.wait_for_completion(show_output=True,
                                    min_node_count=None,
                                    timeout_in_minutes=20)
