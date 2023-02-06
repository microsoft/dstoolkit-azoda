import argparse
import yaml


def generate_aml_config(
    subscription: str, resource_group: str, acr_name: str, workspace: str
) -> dict:
    """Generate a config dictionary for Azure ML.

    Args:
        subscription (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        acr_name (str): Azure container registry name.
        workspace (str): Azure ML workspace name.

    Returns:
        dict: Dictionary containing the config for Azure ML.
    """

    if len(subscription) == 0:
        raise ValueError("subscription cannot be an empty string")

    if len(resource_group) < 1 or len(resource_group) > 90:
        raise ValueError("resource_group must be between 1 and 90 characters long")

    if len(acr_name) < 5 or len(acr_name) > 55:
        raise ValueError("acr_name must be between 5 and 55 characters long")

    if len(workspace) < 3 or len(workspace) > 33:
        raise ValueError("acr_name must be between 3 and 33 characters long")

    # Create the dictionary
    container_reg = (
        f"/subscriptions/{subscription}/resourceGroups/{resource_group}/"
        f"providers/Microsoft.ContainerRegistry/registries/{acr_name}"
    )

    config_dict = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/workspace.schema.json",
        "name": workspace,
        "location": "westeurope",
        "display_name": "Basic workspace",
        "description": "",
        "container_registry": container_reg,
        "hbi_workspace": False,
        "tags": {"purpose": "setup"},
    }

    return config_dict


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # Parse arguments
    parser.add_argument(
        "-a", "--acr_name", help="Your ACR name", required=True, type=str
    )
    parser.add_argument(
        "-s", "--subscription", help="Your Azure subscription", required=True, type=str
    )
    parser.add_argument(
        "-r",
        "--resource_group",
        help="Your Azure resource group",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-w", "--workspace", help="Your AML workspace", required=True, type=str
    )

    return parser.parse_args()


def save_aml_config(config_dict, filename) -> None:
    """Save the config dictionary to a YAML file.

    Args:
        config_dict (dict): Dictionary containing the config for Azure ML.
        filename (str): Filename to save the config to.
    """

    with open(filename, "w") as file:
        yaml.dump(config_dict, file)


if __name__ == "__main__":
    # Get the args
    args = parse_arguments()

    # Generate config
    config_dict = generate_aml_config(
        args.subscription, args.resource_group, args.acr_name, args.workspace
    )

    # Save config
    save_aml_config(config_dict, "aml_config.yaml")
