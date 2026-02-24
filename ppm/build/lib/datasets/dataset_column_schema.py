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
    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    resource_col: str = "org:resource"
    timestamp_col: str = "time:timestamp"
    label_col: str = "label"

    # Label values
    pos_label_col: str = "deviant"
    neg_label_col: str = "regular"

    # Dynamic categorical columns (event attributes)
    # Note: activity_col and resource_col will be auto-added if not present
    dynamic_cat_cols: List[str] = field(default_factory=list)

    # Static categorical columns (case attributes known from start)
    static_cat_cols: List[str] = field(default_factory=list)

    # Dynamic numerical columns
    dynamic_num_cols: List[str] = field(default_factory=list)

    # Static numerical columns
    static_num_cols: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization to ensure activity_col and resource_col are included"""
        # Ensure no duplicates when adding core columns
        if self.activity_col not in self.dynamic_cat_cols:
            self.dynamic_cat_cols.insert(0, self.activity_col)

        if self.resource_col not in self.dynamic_cat_cols:
            self.dynamic_cat_cols.insert(1, self.resource_col)

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
        """All columns in the dataset (order-preserving, deduplicated)"""
        # Use dict.fromkeys() to preserve order while removing duplicates
        return list(dict.fromkeys(self.static_cols + self.dynamic_cols))

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

    def validate_dataframe(self, df: pd.DataFrame, check_core_only: bool = False) -> Dict[str, Any]:
        """
        Validate that a dataframe contains the expected columns
        
        Args:
            df: DataFrame to validate
            check_core_only: If True, only validate core columns exist
            
        Returns:
            Dictionary with validation results
        """
        df_columns = set(df.columns)
        
        if check_core_only:
            expected_columns = set(self.core_cols)
            missing_cols = list(expected_columns - df_columns)
            return {
                "missing_columns": missing_cols,
                "extra_columns": [],
                "is_valid": len(missing_cols) == 0,
            }
        
        expected_columns = set(self.all_cols)
        missing_cols = list(expected_columns - df_columns)
        extra_cols = list(df_columns - expected_columns)

        # Check if core columns are present
        missing_core_cols = [col for col in self.core_cols if col not in df_columns]

        return {
            "missing_columns": missing_cols,
            "missing_core_columns": missing_core_cols,
            "extra_columns": extra_cols,
            "is_valid": len(missing_cols) == 0,
            "has_core_columns": len(missing_core_cols) == 0,
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

    def add_dynamic_categorical_column(self, column_name: str) -> None:
        """Add a new dynamic categorical column"""
        if column_name not in self.dynamic_cat_cols:
            self.dynamic_cat_cols.append(column_name)

    def add_dynamic_numerical_column(self, column_name: str) -> None:
        """Add a new dynamic numerical column"""
        if column_name not in self.dynamic_num_cols:
            self.dynamic_num_cols.append(column_name)

    def add_static_categorical_column(self, column_name: str) -> None:
        """Add a new static categorical column"""
        if column_name not in self.static_cat_cols:
            self.static_cat_cols.append(column_name)

    def add_static_numerical_column(self, column_name: str) -> None:
        """Add a new static numerical column"""
        if column_name not in self.static_num_cols:
            self.static_num_cols.append(column_name)

    def remove_column(self, column_name: str) -> None:
        """Remove a column from all lists (except core columns)"""
        if column_name in [self.case_id_col, self.activity_col, 
                           self.resource_col, self.timestamp_col, self.label_col]:
            raise ValueError(f"Cannot remove core column: {column_name}")
        
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
            "pos_label_col": self.pos_label_col,
            "neg_label_col": self.neg_label_col,
            "dynamic_cat_cols": [col for col in self.dynamic_cat_cols 
                                if col not in [self.activity_col, self.resource_col]],
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
    def BPI17() -> DatasetColumnSchema:
        """Schema for BPIC 2017 dataset"""
        return DatasetColumnSchema(
            case_id_col="case:concept:name",
            activity_col="concept:name",
            resource_col="org:resource",
            timestamp_col="time:timestamp",
            label_col="outcome",
            dynamic_cat_cols=[
                # activity_col and resource_col will be auto-added
                "Action",
                "CreditScore",
                "EventOrigin",
                "lifecycle:transition",
                "Accepted",
                "Selected",
            ],
            static_cat_cols=["case:ApplicationType", "case:LoanGoal"],
            dynamic_num_cols=[
                "FirstWithdrawalAmount",
                "MonthlyCost",
                "NumberOfTerms",
                "OfferedAmount"
            ],
            static_num_cols=["case:RequestedAmount"],
        )