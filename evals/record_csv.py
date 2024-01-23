import json
import logging

import pandas as pd

import evals
from evals.record import LocalRecorder


class CSVRecorder(LocalRecorder):
    """Flatten and convert the saved jsonl file to csv and save it to csv_path."""

    def __init__(self, record_path: str, run_spec: evals.base.RunSpec):
        super().__init__(record_path, run_spec)
        self.jsonl_path = record_path
        # self.jsonl_path = "/mnt/c/Users/charl/OneDrive/sommelia/prompt_engineering/experiments/prout.jsonl"
        # self.csv_path = "/mnt/c/Users/charl/OneDrive/sommelia/prompt_engineering/experiments/240121144055DET2ZGAI_gpt-3.5-turbo-1106_sommelia-prompt-04.csv"
        self.csv_path = record_path.replace(".jsonl", ".csv")

    def _flatten_jsonl(self, df: pd.DataFrame) -> pd.DataFrame:
        """json_normalize doesn't work with nested lists, so we need to
        flatten the jsonl file further.

        Each sample is recorder twice, once with 'record_sampling' and once with
        'record_event'. Rows must be merged and deduplicated

        Also 'spec_columns' and 'final_report' are added at the end of the
        logs. We duplicate their values on each row.
        """
        logging.info(f"CSVRecorder: {self.jsonl_path}")
        # Columns to fill backward
        spec_columns = [col for col in df.columns if col.startswith("spec")]
        final_report = [col for col in df.columns if col.startswith("final_report")]

        return (
            df.sort_values(by=["event_id"])
            .reset_index(drop=True)
            .bfill(axis=0, limit=1)
            .assign(**{col: df[col].ffill() for col in spec_columns})
            .assign(**{col: df[col].ffill() for col in final_report})
            .drop_duplicates(subset=["sample_id"], ignore_index=True)
            .dropna(subset=["data_prompt"], axis=0)
            .assign(
                system_prompt=lambda x: x.data_prompt.map(lambda x: x[0]["content"]),
                user_prompt=lambda x: x.data_prompt.map(lambda x: x[1]["content"]),
            )  # Expand data_prompt list into two columns
            .assign(
                wines=lambda x: x.data_sampled.str[0].map(json.loads)
            )  # Convert the list of wines from list of str to dict
            .assign(
                wine1=lambda x: x.wines.map(lambda x: x["wines"][0]),
                wine2=lambda x: x.wines.map(lambda x: x["wines"][1]),
            )  # Expand wines list into new columns
            .assign(
                **{
                    f"wine1_{k}": (lambda k=k: lambda x: x.wine1.map(lambda y: y[k]))()
                    for k in ["name", "color", "appellation", "country", "explanation"]
                }
            )  # Expand wine1 into new columns
            .assign(
                **{
                    f"wine2_{k}": (lambda k=k: lambda x: x.wine2.map(lambda y: y[k]))()
                    for k in ["name", "color", "appellation", "country", "explanation"]
                }
            )  # Expand wine2 into new columns
            .drop(columns=["data_prompt", "data_sampled", "wines", "wine1", "wine2"])
        )

    def save_as_csv(self):
        """Convert the jsonl file to csv and save it to csv_path."""
        with open(self.jsonl_path, "r", newline="\n", encoding="utf-8") as f:
            data = f.readlines()

        # Each jsonl line is converted to a flat dataframe and concatenated to a df
        df = pd.DataFrame()
        for line in data:
            new_df = pd.json_normalize(
                json.loads(line), sep="_", errors="ignore"
            ).reset_index(drop=True)
            df = pd.concat([df, new_df], axis=0, ignore_index=True)

        processed_df = self._flatten_jsonl(df)

        # Save the dataframe to csv
        processed_df.to_csv(self.csv_path, index=False)
