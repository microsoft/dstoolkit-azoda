'''
Script to kickoff an AML training

It runs the following steps:
    1. Get AML Workspace
    2. Get/Create Experiment
    3. Get/Create Compute Target
    4. Get Run Configuration
    5. Submit training run

This tool does not output anything but the run can be traced
within the AML workspace in Azure
'''

import os

from azureml.train.hyperdrive import PrimaryMetricGoal, BanditPolicy
from azureml.train.hyperdrive import RandomParameterSampling, choice

from azure_utils.azure import load_config
from azure_utils.experiment import AMLExperiment


def get_configs(__here__):
    exp_config = os.path.join(__here__, 'exp_config.json')
    exp_config = load_config(exp_config)
    return exp_config


def main():
    __here__ = os.path.dirname(__file__)

    exp_config = get_configs(__here__)

    env_config_file = exp_config['ENV_CONFIG_FILE']

    docker_image = exp_config['DOCKER_IMAGE_NAME']
    tf_version = exp_config['TF_VERSION']  # Add tensorflow version (by default tf2)
    desc = exp_config['DESCRIPTION']
    cluster_name = exp_config['COMPUTE_NAME']
    experiment_name = exp_config['EXP_NAME']
    store_name = exp_config['DATA_MOUNT']
    img_type = exp_config['IMAGE_TYPE']
    train_csv = exp_config['TRAIN_CSV']
    test_csv = exp_config['TEST_CSV']
    base_model = (exp_config['MODEL_PARAMS']
                            ['BASE_MODEL'])
    steps = (exp_config['RUN_PARAMS']
                       ['STEPS'])
    eval_conf = exp_config['EVAL_CONF']
    fs_nms_iou = (exp_config['MODEL_PARAMS']
                            ['FIRST_STAGE']
                            ['NMS_IOU_THRESHOLD'])
    fs_nms_score = (exp_config['MODEL_PARAMS']
                              ['FIRST_STAGE']
                              ['NMS_SCORE_THRESHOLD'])
    fs_max_prop = (exp_config['MODEL_PARAMS']
                             ['FIRST_STAGE']
                             ['MAX_PROPOSALS'])
    fs_loc_loss = (exp_config['MODEL_PARAMS']
                             ['FIRST_STAGE']
                             ['LOCALIZATION_LOSS_WEIGHT'])
    fs_obj_loss = (exp_config['MODEL_PARAMS']
                             ['FIRST_STAGE']
                             ['OBJECTNESS_LOSS_WEIGHT'])

    hypertune = exp_config['HYPERTUNE']
    nodes = exp_config['NODES']

    # initialise experiment
    aml_exp = AMLExperiment(experiment_name, config_file=env_config_file)

    # set dataset to be parent dir containing Models and datasets
    aml_exp.set_datastore(store_name)
    aml_exp.set_data_reference()

    print('dataref: ', str(aml_exp.data_ref))

    # set compute target
    aml_exp.set_compute(cluster_name, node_count=nodes)

    if 'inception' in base_model:
        # get run parameters
        script_params = [
            '--desc', desc,
            '--data_dir', str(aml_exp.data_ref),
            '--image_type', img_type,
            '--train_csv', train_csv,
            '--test_csv', test_csv,
            '--base_model', base_model,
            '--steps', steps,
            '--fs_nms_iou', fs_nms_iou,
            '--fs_nms_score', fs_nms_score,
            '--fs_max_prop', fs_max_prop,
            '--fs_loc_loss', fs_loc_loss,
            '--fs_obj_loss', fs_obj_loss]

        training_script = 'train_tf2_hypertune.py' if tf_version == 2 else 'train_hypertune.py'

    else:
        print('INFO: Selected Base Model not yet\
               supported for paramter parsing')

        print('INFO: Using model with default config')

        script_params = [
            '--desc', desc,
            '--data_dir', str(aml_exp.data_ref),
            '--image_type', img_type,
            '--train_csv', train_csv,
            '--test_csv', test_csv,
            '--base_model', base_model,
            '--steps', steps,
            '--eval_conf', eval_conf]

        training_script = 'train_tf2.py' if tf_version == 2 else 'train.py'

        print('INFO: Hypertune not supported with unknown base model')
        print('INFO: Running as single node job')
        hypertune = False

    # create estimator
    scripts = os.path.join(__here__, 'scripts')
    aml_exp.set_runconfig(scripts,
                          training_script,
                          script_params,
                          docker_image=docker_image)

    if hypertune:
        print('INFO: Running as Hypertune Job')
        ps = RandomParameterSampling({
            '--fs_nms_iou': choice(0.5, 0.6, 0.7),
            '--fs_max_prop': choice(100, 200, 300)})
        policy = BanditPolicy(evaluation_interval=100, slack_factor=0.25)
        metric_name = 'Train - Total Training Loss',
        metric_goal = PrimaryMetricGoal.MINIMIZE
        aml_exp.submit_hypertune(ps, policy, metric_name, metric_goal)
    else:
        print('INFO: Running as Single Node Job')
        aml_exp.submit_training()

    print('INFO:', aml_exp.run)


if __name__ == '__main__':
    main()
