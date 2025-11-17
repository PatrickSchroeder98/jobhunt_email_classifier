import unittest
import pandas as pd
from src.data_module.dataloader import DataLoader
from src.exceptions.exceptions import PathError
from unittest.mock import patch, MagicMock


class TestDataLoader(unittest.TestCase):
    """Class with tests associated with DataLoader class."""

    def test_init(self):
        """Method tests the initialization of a model"""
        dl = DataLoader()
        self.assertIsNone(dl.path)

        del dl

    def test_set_path(self):
        """Method tests the setting of a path"""
        dl = DataLoader()
        dl.set_path("./data/example.csv")
        self.assertEqual(dl.path, "./data/example.csv")

        del dl

    def test_get_path(self):
        """Method tests the getting of a path"""
        dl = DataLoader()
        dl.path = "./data/example.csv"
        self.assertEqual(dl.get_path(), "./data/example.csv")

        del dl

    def test_load_data_csv_exception(self):
        """Method tests the loading of a csv file that does not exist"""
        with self.assertRaises(PathError):
            dl = DataLoader()
            dl.path = "./data/example.csv"
            df = dl.load_data_csv()
            del df
        del dl

    @patch("pandas.read_csv")
    def test_load_data_csv_success(self, mock_read_csv):
        """Test that load_data_csv returns DataFrame when CSV is readable."""
        # Prepare mock DataFrame
        mock_df = pd.DataFrame({" email_text ": ["X"], " email_type ": ["Y"]})
        mock_read_csv.return_value = mock_df

        loader = DataLoader()
        loader.set_path("test/path/to/data.csv")

        df = loader.load_data_csv()

        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), ["email_text", "email_type"])
        mock_read_csv.assert_called_once_with(
            "test/path/to/data.csv", on_bad_lines="warn"
        )


if __name__ == "__main__":
    unittest.main()
