"""
Phase 37 - Native Synthetic Data Generator for EVORA.

Generates synthetic data for testing, training, and development purposes.

Supports:
  - Schema-based data generation
  - Random data generators (text, numbers, dates, enums)
  - Batch generation
  - Template-based generation
  - Data anonymization utilities
  - Integration with KnowledgeGraph
  - Integration with MemoryManager
  - Integration with DataPipeline

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

class FieldDataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ENUM = "enum"
    TEXT = "text"


@dataclass
class FieldSchema:
    """Schema definition for a data field."""
    name: str
    data_type: FieldDataType = FieldDataType.STRING
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    length: int = 10
    enum_values: list[str] = field(default_factory=list)
    nullable: bool = False
    template: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "length": self.length,
            "enum_values": self.enum_values,
            "nullable": self.nullable,
            "template": self.template,
        }


@dataclass
class DataSchema:
    """Schema for a data record."""
    schema_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    fields: list[FieldSchema] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "name": self.name,
            "fields": [f.to_dict() for f in self.fields],
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Native Synthetic Data Generator
# ---------------------------------------------------------------------------

class NativeSyntheticDataGenerator:
    """Native synthetic data generator for EVORA.

    Generates synthetic data for testing, training, and development.
    """

    def __init__(self, logger: Optional[Any] = None, seed: Optional[int] = None):
        self.logger = logger
        self._schemas: dict[str, DataSchema] = {}
        self._random = random.Random(seed)

    def create_schema(self, name: str, fields: list[FieldSchema]) -> DataSchema:
        """Create and register a new data schema."""
        schema = DataSchema(name=name, fields=fields)
        self._schemas[schema.schema_id] = schema
        return schema

    def register_schema(self, schema: DataSchema) -> str:
        """Register an existing schema."""
        self._schemas[schema.schema_id] = schema
        return schema.schema_id

    def get_schema(self, schema_id: str) -> Optional[DataSchema]:
        """Retrieve a schema by ID."""
        return self._schemas.get(schema_id)

    def generate_record(self, schema_id: str, null_probability: float = 0.0) -> dict[str, Any]:
        """Generate a single synthetic record based on schema."""
        schema = self._schemas.get(schema_id)
        if schema is None:
            raise ValueError(f"Schema not found: {schema_id}")
        record: dict[str, Any] = {}
        for field in schema.fields:
            if field.nullable and self._random.random() < null_probability:
                record[field.name] = None
            else:
                record[field.name] = self._generate_value(field)
        return record

    def generate_batch(self, schema_id: str, count: int, null_probability: float = 0.0) -> list[dict[str, Any]]:
        """Generate a batch of synthetic records."""
        return [self.generate_record(schema_id, null_probability) for _ in range(count)]

    def _generate_value(self, field: FieldSchema) -> Any:
        """Generate a value for a given field schema."""
        generator_method = {
            FieldDataType.STRING: self._generate_string,
            FieldDataType.INTEGER: self._generate_integer,
            FieldDataType.FLOAT: self._generate_float,
            FieldDataType.BOOLEAN: self._generate_boolean,
            FieldDataType.DATE: self._generate_date,
            FieldDataType.DATETIME: self._generate_datetime,
            FieldDataType.ENUM: self._generate_enum,
            FieldDataType.TEXT: self._generate_text,
        }.get(field.data_type, self._generate_string)
        return generator_method(field)

    def _generate_string(self, field: FieldSchema) -> str:
        length = field.length if field.length > 0 else 10
        return ''.join(self._random.choices(string.ascii_letters, k=length))

    def _generate_integer(self, field: FieldSchema) -> int:
        min_val = int(field.min_value) if field.min_value is not None else 0
        max_val = int(field.max_value) if field.max_value is not None else 100
        return self._random.randint(min_val, max_val)

    def _generate_float(self, field: FieldSchema) -> float:
        min_val = field.min_value if field.min_value is not None else 0.0
        max_val = field.max_value if field.max_value is not None else 100.0
        return round(self._random.uniform(min_val, max_val), 2)

    def _generate_boolean(self, field: FieldSchema) -> bool:
        return self._random.choice([True, False])

    def _generate_date(self, field: FieldSchema) -> str:
        base = datetime.now()
        delta = timedelta(days=self._random.randint(-365, 0))
        return (base + delta).strftime("%Y-%m-%d")

    def _generate_datetime(self, field: FieldSchema) -> str:
        base = datetime.now()
        delta = timedelta(days=self._random.randint(-365, 0), hours=self._random.randint(0, 23))
        return (base + delta).strftime("%Y-%m-%dT%H:%M:%S")

    def _generate_enum(self, field: FieldSchema) -> str:
        if not field.enum_values:
            return self._random.choice(["option_a", "option_b", "option_c"])
        return self._random.choice(field.enum_values)

    def _generate_text(self, field: FieldSchema) -> str:
        sentences = []
        for _ in range(self._random.randint(1, 5)):
            word_count = self._random.randint(3, 15)
            words = ''.join(self._random.choices(string.ascii_lowercase, k=word_count))
            sentences.append(words)
        return ' '.join(sentences)

    def generate_anonymized_record(self, original: dict[str, Any], pii_fields: list[str]) -> dict[str, Any]:
        """Anonymize PII fields in an existing record."""
        anonymized = dict(original)
        for pii_field in pii_fields:
            if pii_field in anonymized:
                anonymized[pii_field] = self._anonymize_value(anonymized[pii_field])
        return anonymized

    def _anonymize_value(self, value: Any) -> Any:
        """Anonymize a single value."""
        if value is None:
            return None
        if isinstance(value, str):
            return ''.join(self._random.choices(string.ascii_letters + string.digits, k=len(value)))
        if isinstance(value, (int, float)):
            return self._random.randint(1000, 9999)
        return value

    def generate_from_template(self, template: str, variables: dict[str, Any]) -> str:
        """Generate text from a template with variables."""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def get_generator_metrics(self) -> dict[str, Any]:
        """Get metrics about the synthetic data generator."""
        return {
            "total_schemas": len(self._schemas),
            "schemas": [s.name for s in self._schemas.values()],
        }
