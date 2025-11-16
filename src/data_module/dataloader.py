import pandas as pd
from src.exceptions.exceptions import PathError


class DataLoader:
    """Class that handles data loading."""

    def __init__(self):
        """Init method that initializes the path as none object."""
        self.path = None

    def set_path(self, path):
        """Method that can set path."""
        self.path = path

    def get_path(self):
        """Method that returns path."""
        return self.path

    def load_data_csv(self):
        """Method that loads data from csv file and returns dataframe."""
        try:
            df = pd.read_csv(self.get_path(), on_bad_lines="warn")
            df.columns = df.columns.str.strip()
            return df
        except FileNotFoundError:
            raise PathError()
