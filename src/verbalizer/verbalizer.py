import pandas as pd

from typing import Optional

from src.utils.file_utils import load_json

class Verbalizer:
    """
    Turns tabular rows into a simple textual description.
    """

    def __init__(
        self,
        table_data: Optional[pd.DataFrame] = None,
        json_features_names_config: Optional[str] = None
    ):
        self.data: Optional[pd.DataFrame] = table_data
        self.textual_data: Optional[pd.DataFrame] = None

        self.features_names = (
            load_json(json_features_names_config) if json_features_names_config is not None else {}
        )

    def set_table_data(self, table_data: pd.DataFrame) -> None:
        self.data = table_data
        # Invalidate cached textual data (if any)
        self.textual_data = None

    def get_feature_name(self, feature: str) -> str:
        """
        Given a features-name mapping from table keys to human readable names,
        return the mapped name if available; otherwise return the key itself.
        """
        return self.features_names.get(feature, feature)

    def verbalize(self, row: pd.Series):
        simple_text = ""
        for col_index, col_value in enumerate(row):
            if self.data is None:
                raise Exception("You must set the table data before calling verbalize")

            col_name = self.data.columns[col_index]
            if self.without_nulls and pd.isna(col_value):
                continue

            if col_name == "set_type" or col_name == "label":
                continue

            feat_name = self.get_feature_name(col_name)
            simple_text += f"{feat_name}: {self._format_value(col_value)}, "

        # Remove the last comma and space and add a period
        simple_text = simple_text[:-2] + "."
        return simple_text

    def verbalize_table(self) -> pd.DataFrame:
        """
        Iterate over the rows and create a new `text` column using `verbalize`.
        Returns a dataframe with columns: `text`, `set_type`, `label`.
        """
        if self.data is None:
            raise Exception("You must set the table data before getting the textual data")

        df = self.data.copy()
        df["text"] = self.data.apply(lambda row: self.verbalize(row), axis=1)

        textual_columns = ["text", "set_type", "label"]
        self.textual_data = df[textual_columns].copy()
        return self.textual_data.copy()
