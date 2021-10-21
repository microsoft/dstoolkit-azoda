import sys

from azureml.core import Datastore, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.environment import Environment
from azureml.train.hyperdrive import HyperDriveConfig

from azure_utils.azure import AML


class AMLExperiment(AML):
    def __init__(self, experiment_name,
                 config_file=None,
                 ws=None,
                 sub=None,
                 rg=None,
                 tenant=None,
                 sp_id=None,
                 sp_pw=None,
                 cli_auth=False):
        """
        AML experiment class that inherits the base AML class
        contains all the core functions to submit an AML experiment
        :param experiment_name: AML experiment group reference name (str)
        """
        super(AMLExperiment, self).__init__(config_file,
                                            ws, sub, rg, tenant,
                                            sp_id, sp_pw,
                                            cli_auth)

        self.experiment_name = experiment_name

    def set_runconfig(self,
                      src_dir,
                      script,
                      script_params,
                      docker_image=None,
                      username=None,
                      password=None):
        """
        Configures the Azure ML estimator to use on experiment run
        :param src_dir: path to the directory containing scripts
        :param script: training script to use
        :param script_params: dictionary of script parameters
        :param docker_image: optional docker image name to use for training
        defaults to None if no argument given
        """
        env = self.get_env(docker_image, username, password)

        src = ScriptRunConfig(source_directory=src_dir,
                              script=script,
                              arguments=script_params,
                              compute_target=self.compute,
                              environment=env)

        src.run_config.data_references = {self.data_ref.data_reference_name:
                                          self.data_ref.to_config()}

        self.src = src

    def set_compute(self, cluster_name, vm_type='STANDARD_NC6', node_count=1):
        """
        Sets the compute target for the experiment run and creates
        a new one if it doesnt exist in the current AML workspace
        :param cluster_name: cluster name
        :param vm_type: azure vm type, defaults to 'standard_NC6'
        :param node_count: number of compute nodes
        """
        self.compute_name = cluster_name
        try:
            self.compute = ComputeTarget(workspace=self.ws,
                                         name=self.compute_name)
        except ComputeTargetException:
            print('INFO: Compute does not exist, creating new compute')
            self.compute = self.create_compute(cluster_name,
                                               vm_type,
                                               node_count)

        cp_details = self.compute.get_status().serialize()
        self.compute_nodes = (cp_details.get('scaleSettings')
                              .get('maxNodeCount'))

    def set_datastore(self, store_name):
        """
        Sets the datatore mount based on the AML datastore reference
        :param store_name: registered AMl datastore reference name
        """
        self.datastore = Datastore.get(self.ws, datastore_name=store_name)

    def set_data_reference(self, mount_path=None):
        """
        Sets the datareference paths on the AML datareference object.
        :param mount_path: mounting part default set to None
        :returns: None
        """
        self.data_ref = self.datastore.path(mount_path).as_mount()

    def submit_training(self):
        """
        Submits the experiment for training based on the defined
        class attributes
        """
        self.exp = Experiment(self.ws, name=self.experiment_name)
        self.run = self.exp.submit(self.src)

    def submit_hypertune(self, params, policy, metric_name, metric_goal):
        """
        Submits the experiment for hypertune training based on the defined
        class attributes and hypertune policies
        :param params: hyperdrive random parameter sampling object containing
        the desired paramters and there ranges to tune
        :param policy: hyperdrive bandit policy object to enforce early
        stopping of runs
        :param metric_name: name of AMl logged metric to optimise on
        :param metric_goal: hyperdrive metric goal object to set the
        optimisation goal
        """
        if self.compute_nodes < 2:
            print('INFO: Hypertune requires 2 or more nodes!')
            print('INFO: Please resubmit with more nodes')
            sys.exit(0)
        hd = HyperDriveConfig(run_config=self.src,
                              hyperparameter_sampling=params,
                              policy=policy,
                              primary_metric_name=metric_name,
                              primary_metric_goal=metric_goal,
                              max_total_runs=self.compute_nodes,
                              max_concurrent_runs=self.compute_nodes)

        self.exp = Experiment(self.ws, name=self.experiment_name)
        self.run = self.exp.submit(hd)

    def get_env(self, docker_image=None, username=None, password=None):
        """
        creates an Azure ML environment with the docker training image
        :param docker_image: address of docker image to use,
        assumes is in default AML ACR
        :return: AML env object
        """
        env = Environment(name="train_env")
        env.docker.enabled = True
        env.docker.base_image = docker_image

        if username is not None and password is not None:
            acr_address = '{}.azurecr.io'.format(username)
            env.docker.base_image_registry.address = acr_address
            env.docker.base_image_registry.username = username
            env.docker.base_image_registry.password = password

        env.python.user_managed_dependencies = True
        return env
