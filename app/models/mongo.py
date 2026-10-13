import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator


MongoStageOperator = Literal[
    "$addFields",
    "$bucket",
    "$bucketAuto",
    "$count",
    "$group",
    "$limit",
    "$lookup",
    "$match",
    "$project",
    "$replaceRoot",
    "$replaceWith",
    "$set",
    "$skip",
    "$sort",
    "$sortByCount",
    "$unset",
    "$unwind",
]


class GeneratedMongoStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: MongoStageOperator
    specification_json: str

    @field_validator("specification_json")
    @classmethod
    def validate_specification_json(cls, value: str) -> str:
        try:
            json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                "MongoDB stage specification must be valid JSON."
            ) from error
        return value


class GeneratedMongoQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    pipeline: list[GeneratedMongoStage]
    fields_used: list[str]
