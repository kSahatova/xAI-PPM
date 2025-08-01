from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd


@dataclass
class DatasetColumnSchema:
    """
    Dataclass for managing dataset column schema with automatic derivation of
    static/dynamic and categorical/numerical column groupings.
    """

    # Core identification columns
    case_id_col: str = "Case ID"
    activity_col: str = "Activity"
    resource_col: str = "org:resource"
    timestamp_col: str = "time:timestamp"
    label_col: str = "label"

    # Label values
    pos_label_col: str = "deviant"
    neg_label_col: str = "regular"

    # Dynamic categorical columns (event attributes)
    dynamic_cat_cols: List[str] = field(default_factory=list)

    # Static categorical columns (case attributes known from start)
    static_cat_cols: List[str] = field(default_factory=list)

    # Dynamic numerical columns
    dynamic_num_cols: List[str] = field(default_factory=list)

    # Static numerical columns
    static_num_cols: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization to ensure activity_col is included in dynamic_cat_cols"""
        if self.activity_col not in self.dynamic_cat_cols:
            self.dynamic_cat_cols.append(self.activity_col)

        if self.resource_col not in self.dynamic_cat_cols:
            self.dynamic_cat_cols.append(self.resource_col)

    @property
    def static_cols(self) -> List[str]:
        """All static columns including case_id and label"""
        return (
            self.static_cat_cols
            + self.static_num_cols
            + [self.case_id_col, self.label_col]
        )

    @property
    def dynamic_cols(self) -> List[str]:
        """All dynamic columns including timestamp"""
        return self.dynamic_cat_cols + self.dynamic_num_cols + [self.timestamp_col]

    @property
    def cat_cols(self) -> List[str]:
        """All categorical columns (static + dynamic)"""
        return self.dynamic_cat_cols + self.static_cat_cols

    @property
    def num_cols(self) -> List[str]:
        """All numerical columns (static + dynamic)"""
        return self.dynamic_num_cols + self.static_num_cols

    @property
    def all_cols(self) -> List[str]:
        """All columns in the dataset"""
        return list(set(self.static_cols + self.dynamic_cols))

    @property
    def core_cols(self) -> List[str]:
        """Core system columns (case_id, activity, resource, timestamp, label)"""
        return [
            self.case_id_col,
            self.activity_col,
            self.resource_col,
            self.timestamp_col,
            self.label_col,
        ]

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that a dataframe contains the expected columns
        Returns a dictionary with missing and extra columns
        """
        df_columns = set(df.columns)
        expected_columns = set(self.all_cols)

        missing_cols = list(expected_columns - df_columns)
        extra_cols = list(df_columns - expected_columns)

        return {
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "is_valid": len(missing_cols) == 0,
        }

    def get_encoder_args(self, fillna: bool = True) -> Dict[str, Any]:
        """Get arguments for encoder initialization"""
        return {
            "case_id_col": self.case_id_col,
            "static_cat_cols": self.static_cat_cols,
            "static_num_cols": self.static_num_cols,
            "dynamic_cat_cols": self.dynamic_cat_cols,
            "dynamic_num_cols": self.dynamic_num_cols,
            "fillna": fillna,
        }

    def add_dynamic_categorical_column(self, column_name: str):
        """Add a new dynamic categorical column"""
        if column_name not in self.dynamic_cat_cols:
            self.dynamic_cat_cols.append(column_name)

    def add_dynamic_numerical_column(self, column_name: str):
        """Add a new dynamic numerical column"""
        if column_name not in self.dynamic_num_cols:
            self.dynamic_num_cols.append(column_name)

    def add_static_categorical_column(self, column_name: str):
        """Add a new static categorical column"""
        if column_name not in self.static_cat_cols:
            self.static_cat_cols.append(column_name)

    def add_static_numerical_column(self, column_name: str):
        """Add a new static numerical column"""
        if column_name not in self.static_num_cols:
            self.static_num_cols.append(column_name)

    def remove_column(self, column_name: str):
        """Remove a column from all lists"""
        for col_list in [
            self.dynamic_cat_cols,
            self.dynamic_num_cols,
            self.static_cat_cols,
            self.static_num_cols,
        ]:
            if column_name in col_list:
                col_list.remove(column_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization"""
        return {
            "case_id_col": self.case_id_col,
            "activity_col": self.activity_col,
            "resource_col": self.resource_col,
            "timestamp_col": self.timestamp_col,
            "label_col": self.label_col,
            "pos_label": self.pos_label_col,
            "neg_label": self.neg_label_col,
            "dynamic_cat_cols": self.dynamic_cat_cols.copy(),
            "static_cat_cols": self.static_cat_cols.copy(),
            "dynamic_num_cols": self.dynamic_num_cols.copy(),
            "static_num_cols": self.static_num_cols.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetColumnSchema":
        """Create schema from dictionary"""
        return cls(**data)

    def __str__(self) -> str:
        """String representation of the schema"""
        return f"""DatasetColumnSchema:
                Core Columns:
                    Case ID: {self.case_id_col}
                    Activity: {self.activity_col}
                    Resource: {self.resource_col}
                    Timestamp: {self.timestamp_col}
                    Label: {self.label_col} (pos: {self.pos_label_col}, neg: {self.neg_label_col})
                
                Static Columns:
                    Categorical ({len(self.static_cat_cols)}): {self.static_cat_cols}
                    Numerical ({len(self.static_num_cols)}): {self.static_num_cols}
                
                Dynamic Columns:
                    Categorical ({len(self.dynamic_cat_cols)}): {self.dynamic_cat_cols}
                    Numerical ({len(self.dynamic_num_cols)}): {self.dynamic_num_cols}
                
                Total Columns: {len(self.all_cols)}"""


# Predefined schemas for common datasets
class DatasetSchemas:
    """Collection of predefined schemas for common datasets"""

    @staticmethod
    def bpic2017() -> DatasetColumnSchema:
        """Schema for BPIC 2017 dataset"""
        return DatasetColumnSchema(
            case_id_col="Case ID",
            activity_col="Activity",
            resource_col="org:resource",
            timestamp_col="time:timestamp",
            label_col="label",
            pos_label_col="deviant",
            neg_label_col="regular",
            dynamic_cat_cols=[
                "Activity",
                "org:resource",
                "Action",
                "CreditScore",
                "EventOrigin",
                "lifecycle:transition",
                "Accepted",
                "Selected",
            ],
            static_cat_cols=["ApplicationType", "LoanGoal"],
            dynamic_num_cols=[
                "FirstWithdrawalAmount",
                "MonthlyCost",
                "NumberOfTerms",
                "OfferedAmount",
                "timesincelastevent",
                "timesincecasestart",
                "timesincemidnight",
                "event_nr",
                "month",
                "weekday",
                "hour",
                "open_cases",
            ],
            static_num_cols=["RequestedAmount"],
        )
