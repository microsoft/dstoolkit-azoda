import unittest
from unittest import mock
import os
import tempfile
import argparse
import yaml

from write_aml_template_yaml import (
    generate_aml_config,
    parse_arguments,
    save_aml_config,
)


class TestWriteAMLTemplateYAML(unittest.TestCase):
    @mock.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(
            subscription="azoda-sub",
            resource_group="azoda-rg",
            acr_name="azoda-cr",
            workspace="azoda-amlw",
        ),
    )
    def test_parse_arguments(self, _):
        """Test the parse_arguments function."""
        args = parse_arguments()

        self.assertEqual(args.subscription, "azoda-sub")
        self.assertEqual(args.resource_group, "azoda-rg")
        self.assertEqual(args.acr_name, "azoda-cr")
        self.assertEqual(args.workspace, "azoda-amlw")

    def test_parse_arguments_no_args(self):
        """Test the parse_arguments function with no arguments."""
        with self.assertRaises(SystemExit) as cm:
            parse_arguments()

        self.assertEqual(cm.exception.code, 2)

    def test_generate_config(self):
        """Test the generate_aml_config function."""

        subscription = "azoda-sub"
        resource_group = "azoda-rg"
        acr_name = "azoda-cr"
        workspace = "azoda-amlw"

        config_dict = generate_aml_config(
            subscription,
            resource_group,
            acr_name,
            workspace,
        )

        container_reg = (
            f"/subscriptions/{subscription}/resourceGroups/{resource_group}/"
            f"providers/Microsoft.ContainerRegistry/registries/{acr_name}"
        )

        self.assertEqual(config_dict["name"], workspace)
        self.assertEqual(config_dict["container_registry"], container_reg)

    def test_generate_config_invalid_subscription(self):
        """Test the generate_aml_config function with an invalid subscription."""
        with self.assertRaises(ValueError):
            subscription = ""
            resource_group = "azoda-rg"
            acr_name = "azoda-cr"
            workspace = "azoda-amlw"

            generate_aml_config(
                subscription,
                resource_group,
                acr_name,
                workspace,
            )

    def test_generate_config_invalid_resource_group(self):
        """Test the generate_aml_config function with an invalid resource_group."""
        with self.assertRaises(ValueError):
            subscription = "azoda-sub"
            resource_group = "invalid" * 20
            acr_name = "azoda-cr"
            workspace = "azoda-amlw"

            generate_aml_config(
                subscription,
                resource_group,
                acr_name,
                workspace,
            )

    def test_generate_config_invalid_acr_name(self):
        """Test the generate_aml_config function with an invalid resource_group."""
        with self.assertRaises(ValueError):
            subscription = "azoda-sub"
            resource_group = "azoda-rg"
            acr_name = "invalid" * 10
            workspace = "azoda-amlw"

            generate_aml_config(
                subscription,
                resource_group,
                acr_name,
                workspace,
            )

    def test_generate_config_invalid_workspace(self):
        """Test the generate_aml_config function with an invalid workspace."""
        with self.assertRaises(ValueError):
            subscription = "azoda-sub"
            resource_group = "azoda-rg"
            acr_name = "azoda-cr"
            workspace = "invalid" * 10

            generate_aml_config(
                subscription,
                resource_group,
                acr_name,
                workspace,
            )

    def test_save_aml_config(self):
        """Test the save_aml_config function."""
        # Create a temporary directory
        with tempfile.NamedTemporaryFile() as tmpfile:
            # Close the file so it can be used in another function
            tmpfile.close()

            subscription = "azoda-sub"
            resource_group = "azoda-rg"
            acr_name = "azoda-cr"
            workspace = "azoda-amlw"

            # Create config
            config_dict = generate_aml_config(
                subscription,
                resource_group,
                acr_name,
                workspace,
            )

            save_aml_config(
                config_dict,
                tmpfile.name,
            )

            # Check if file exists
            self.assertTrue(os.path.exists(tmpfile.name))

            with open(tmpfile.name, "r") as file:
                # Load and check contents
                config = yaml.safe_load(file)

                container_reg = (
                    f"/subscriptions/{subscription}/resourceGroups/{resource_group}/"
                    f"providers/Microsoft.ContainerRegistry/registries/{acr_name}"
                )

                self.assertEqual(config["name"], workspace)
                self.assertEqual(config["container_registry"], container_reg)


if __name__ == "__main__":
    unittest.main()
