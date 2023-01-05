import unittest
from unittest import mock
import os
import tempfile
import argparse

from write_aml_template_yaml import (
    generate_aml_config,
    parse_arguments,
    save_aml_config,
)


class TestWriteAMLTemplateYAML(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryFile()

    def tearDown(self):
        self.test_dir.close()

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
        with self.assertRaises(SystemExit):
            parse_arguments()

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
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_file_name = os.path.join(
                tmpdirname, "unit_test_save_aml_config", "aml_config.yml"
            )

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

            save_aml_config(
                config_dict,
                config_file_name,
            )

            self.assertTrue(os.path.exists(config_file_name))


if __name__ == "__main__":
    unittest.main()
