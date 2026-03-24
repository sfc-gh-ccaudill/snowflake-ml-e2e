"""
Configuration dataclasses for Healthcare ML Pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class SnowflakeConfig:
    connection_name: str
    database: str
    schema_name: str
    warehouse: str


@dataclass
class ComputeConfig:
    compute_pool: str
    instance_family: str
    min_nodes: int
    max_nodes: int


@dataclass
class ModelConfig:
    model_name: str
    target_platforms: List[str]


@dataclass
class TableConfig:
    raw_data: str


@dataclass
class FeatureConfig:
    raw_numeric_features: List[str]
    categorical_features: List[str]
    computed_features: List[str]
    target_column: str
    class_labels: List[str]


@dataclass
class PipelineConfig:
    snowflake: SnowflakeConfig
    compute: ComputeConfig
    model: ModelConfig
    tables: TableConfig
    feature_config: FeatureConfig

    @property
    def full_schema(self) -> str:
        return f"{self.snowflake.database}.{self.snowflake.schema_name}"

    @property
    def full_raw_table(self) -> str:
        return f"{self.full_schema}.{self.tables.raw_data}"


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config(config_path: str = "config.yaml") -> PipelineConfig:
    config_dict = load_config(config_path)

    snowflake_config = SnowflakeConfig(
        connection_name=config_dict.get("snowflake", {}).get(
            "connection_name", os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
        ),
        database=config_dict.get("snowflake", {}).get("database"),
        schema_name=config_dict.get("snowflake", {}).get("schema"),
        warehouse=config_dict.get("snowflake", {}).get("warehouse"),
    )

    compute_config = ComputeConfig(
        compute_pool=config_dict.get("compute", {}).get("compute_pool"),
        instance_family=config_dict.get("compute", {}).get("instance_family"),
        min_nodes=config_dict.get("compute", {}).get("min_nodes"),
        max_nodes=config_dict.get("compute", {}).get("max_nodes"),
    )

    model_cfg = config_dict.get("model", {})
    model_config = ModelConfig(
        model_name=model_cfg.get("model_name"),
        target_platforms=model_cfg.get("target_platforms"),
    )

    tables_cfg = config_dict.get("tables", {})
    table_config = TableConfig(
        raw_data=tables_cfg.get("raw_data"),
    )

    feat_cfg = config_dict.get("feature_config", {})
    feature_config = FeatureConfig(
        raw_numeric_features=feat_cfg.get("raw_numeric_features"),
        categorical_features=feat_cfg.get("categorical_features"),
        computed_features=feat_cfg.get("computed_features"),
        target_column=feat_cfg.get("target_column"),
        class_labels=feat_cfg.get("class_labels"),
    )

    return PipelineConfig(
        snowflake=snowflake_config,
        compute=compute_config,
        model=model_config,
        tables=table_config,
        feature_config=feature_config,
    )
