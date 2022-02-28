import os

from azureml.core import Experiment
from azureml.core.run import get_run
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import Webservice, AksWebservice, AciWebservice
from azureml.core.compute import AksCompute, ComputeTarget

from azure_utils.azure import AML


class AMLDeploy(AML):
    def __init__(self,
                 run_id,
                 experiment,
                 webservice_name,
                 model_name,
                 img_type,
                 config_file=None,
                 ws=None,
                 sub=None,
                 rg=None,
                 tenant=None,
                 sp_id=None,
                 sp_pw=None,
                 cli_auth=False):
        """
        AML deployment class that inherits the base AML class
        contains all the core functions to take a model run and deploy
        to either ACI or AKS
        :param run_id: AML experiment run_id to deploy (str)
        """
        super(AMLDeploy, self).__init__(config_file,
                                        ws, sub, rg, tenant,
                                        sp_id, sp_pw,
                                        cli_auth)
        self.run_id = run_id
        self.experiment = experiment
        self.webservice_name = webservice_name
        self.model_name = model_name
        self.img_type = img_type

    def create_aks(self,
                   compute_name,
                   vm_type='Standard_NC6',
                   nodes=3,
                   app_insights=False,
                   ssl_cert_pem=None,
                   ssl_key_pem=None,
                   ssl_cname=None,
                   exists=False):
        """
        Creates an AKS service based on the provided configuration
        :param compute_name: AKS cluster name
        :param vm_type: Azure VM type to use, defaults to 'Standard_NC6' (str)
        :param nodes: Numebr of VM nodes for service, defaults to 3 (int)
        :param app_insights: Enable app insights, defaults to false (bool)
        :param ssl_cert_pem: PEM-encoded certificate file
        :param ssl_key_pem: PEM-encoded key file
        :param ssl_cname: FQDN of the address that you plan to use for the
        web service. The address that's stamped into the certificate and the
        address that the clients use are compared to verify the identity
        of the web service
        :param exists: Existence of the given compute name, defaults to true (bool)
        """

        if exists:
            target = ComputeTarget(self.ws, compute_name)

        else:

            prov_config = AksCompute.provisioning_configuration(agent_count=nodes,
                                                                vm_size=vm_type)

            if all(v is not None for v in [ssl_cert_pem, ssl_key_pem, ssl_cname]):
                print('ssl params provided')
                print('config will use HTTPS')
                prov_config.enable_ssl(ssl_cert_pem_file=ssl_cert_pem,
                                       ssl_key_pem_file=ssl_key_pem,
                                       ssl_cname=ssl_cname)
            else:
                print('ssl params not provided')
                print('config will use HTTP instead or HTTPS')

            target = (ComputeTarget
                      .create(workspace=self.ws,
                              name=compute_name,
                              provisioning_configuration=prov_config))

            target.wait_for_completion(show_output=True)

        config = (AksWebservice
                  .deploy_configuration(enable_app_insights=app_insights,
                                        cpu_cores=2,
                                        memory_gb=4))

        return target, config

    def create_aci(self,
                   cpu_cores=2,
                   memory_gb=4,
                   app_insights=False,
                   aci_auth=True,
                   ssl_cert_pem=None,
                   ssl_key_pem=None,
                   ssl_cname=None):
        """
        Creates an ACI instances based on the provided configuration
        :param cpu_cores: number of cpu cores, defaults to 2 (int)
        :param memory_gb: memory allocation for instance, defaults to 4 (int)
        :param app_insights: Enable app insights, defaults to false (bool)
        :param aci_auth: Enable key-based auth, defaults to true (bool)
        :param ssl_cert_pem: PEM-encoded certificate file
        :param ssl_key_pem: PEM-encoded key file
        :param ssl_cname: FQDN of the address that you plan to use for the
        web service. The address that's stamped into the certificate and the
        address that the clients use are compared to verify the identity
        of the web service
        """
        target = None

        if all(v is None for v in [ssl_cert_pem, ssl_key_pem, ssl_cname]):
            print('ssl params not provided')
            print('config will use HTTP instead or HTTPS')
            config = (AciWebservice
                      .deploy_configuration(cpu_cores=cpu_cores,
                                            memory_gb=memory_gb,
                                            enable_app_insights=app_insights,
                                            auth_enabled=aci_auth))
        else:
            print('ssl params provided')
            print('config will use HTTPS')
            config = (AciWebservice
                      .deploy_configuration(cpu_cores=cpu_cores,
                                            memory_gb=memory_gb,
                                            enable_app_insights=app_insights,
                                            auth_enabled=aci_auth,
                                            ssl_enabled=True,
                                            ssl_cert_pem_file=ssl_cert_pem,
                                            ssl_key_pem_file=ssl_key_pem,
                                            ssl_cname=ssl_cname))
        return target, config

    def deploy_new_webservice(self,
                              model,
                              infer_config,
                              deploy_config,
                              deploy_target):
        """
        Deploy new webservice using the built docker and service tagret/config
        :param model: AML registered model
        :param infer_config: inference configuration
        :param target: compute target reference
        :param config: compute target config
        """

        service = Model.deploy(workspace=self.ws,
                               name=self.webservice_name,
                               models=[model],
                               inference_config=infer_config,
                               deployment_config=deploy_config,
                               deployment_target=deploy_target)

        service.wait_for_deployment(show_output=True)
        print(service.state)

    def update_existing_webservice(self,
                                   model,
                                   infer_config):
        """
        Update Existing Webservice
        deploy a docker image to the existing webservice
        :param model: AML registered model
        :param infer_config: inference configuration
        """
        service = Webservice(name=self.webservice_name, workspace=self.ws)

        service.update(models=[model], inference_config=infer_config)

        service.wait_for_deployment(show_output=True)

        print(service.state)

    def update_score_script(self, score_file, model_name):
        """
        Update Scoring Script
        Updates the scoring script with the specific model refernce
        :param score_file: base scoring file
        :param model_name: model name
        :return: path to new scoreing file
        """
        file_dir = os.path.dirname(score_file)
        file_name = 'model_score.py'
        new_score_file = os.path.join(file_dir, file_name)

        # Update score.py with params
        with open(score_file, "rt") as score:
            with open(new_score_file, "wt") as new_score:
                for line in score:
                    new_score.write(
                        line
                        .replace('__REPLACE_MODEL_NAME__',
                                 model_name))
        return file_name

    def register_run_model(self):
        """
        Register Model from Experiment Run
        Registers the model object from the AML run
        :return: registered model reference
        """
        exp = Experiment(workspace=self.ws, name=self.experiment)

        run = get_run(exp, self.run_id, rehydrate=True)
        model_path = 'outputs/final_model/model/'

        tags = {'run_id': self.run_id,
                'experiment': self.experiment}

        model = run.register_model(model_name=self.img_type,
                                   model_path=model_path,
                                   tags=tags)
        return model

    def create_inference_config(self,
                                score_file,
                                src_dir,
                                conda_file):
        """
        Create Inference Config
        Defines the configuration for inference including model and scoring
        script
        :param score_file: the path to the scoring file
        :param src_dir: src directory containing entry script and
        all required scripts
        :return: AML inference config object
        """

        new_score_file = self.update_score_script(score_file, self.model_name)

        inference_config = InferenceConfig(entry_script=new_score_file,
                                           runtime="python",
                                           conda_file=conda_file,
                                           description=self.model_name,
                                           source_directory=src_dir)

        return inference_config

    def webservice_exists(self, service_name):
        """
        Webservice Exists
        Checks if a webservice exists and returns a bool confirmation
        with all the dependencies
        :param service_name: service name
        :return: True/False
        """
        service_list = Webservice.list(self.ws)
        service_names = [service.name for service in service_list]

        if service_name in service_names:
            return True
        else:
            return False

    def find_model(self, model_name, tags=None):
        """
        Find Model
        Takes a model name and finds an existing registered model
        :param model_name: model name
        :param tags: list of tags to filter on [key, [key, value]]
        :return: model
        """
        try:
            model = Model(self.ws, model_name, tags=tags)
        except FileNotFoundError:
            model = None
        return model
