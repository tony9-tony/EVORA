"""
Phase 37 — Tests for Native Synthetic Data Generator.

Tests synthetic data generation capabilities.
"""

import unittest
from evora.brain.intelligence.synthetic_data import (
    NativeSyntheticDataGenerator,
    DataSchema,
    FieldSchema,
    FieldDataType,
)


class TestNativeSyntheticDataGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = NativeSyntheticDataGenerator(seed=42)

    def test_create_schema(self):
        schema = self.generator.create_schema(
            name="user",
            fields=[FieldSchema(name="username", data_type=FieldDataType.STRING)],
        )
        self.assertEqual(schema.name, "user")
        self.assertEqual(len(schema.fields), 1)

    def test_register_schema(self):
        schema = DataSchema(name="test")
        schema_id = self.generator.register_schema(schema)
        self.assertEqual(schema_id, schema.schema_id)

    def test_get_schema(self):
        schema = self.generator.create_schema(name="test", fields=[])
        retrieved = self.generator.get_schema(schema.schema_id)
        self.assertEqual(retrieved, schema)

    def test_get_schema_invalid(self):
        retrieved = self.generator.get_schema("nonexistent")
        self.assertIsNone(retrieved)

    def test_generate_record_string(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="value", data_type=FieldDataType.STRING, length=5)],
        )
        record = self.generator.generate_record(schema.schema_id)
        self.assertIn("value", record)
        self.assertEqual(len(record["value"]), 5)

    def test_generate_record_integer(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="count", data_type=FieldDataType.INTEGER, min_value=10, max_value=20)],
        )
        record = self.generator.generate_record(schema.schema_id)
        self.assertTrue(10 <= record["count"] <= 20)

    def test_generate_record_float(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="amount", data_type=FieldDataType.FLOAT, min_value=0.0, max_value=1.0)],
        )
        record = self.generator.generate_record(schema.schema_id)
        self.assertTrue(0.0 <= record["amount"] <= 1.0)

    def test_generate_record_boolean(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="active", data_type=FieldDataType.BOOLEAN)],
        )
        record = self.generator.generate_record(schema.schema_id)
        self.assertIsInstance(record["active"], bool)

    def test_generate_record_date(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="created", data_type=FieldDataType.DATE)],
        )
        record = self.generator.generate_record(schema.schema_id)
        self.assertIn("-", record["created"])

    def test_generate_record_enum(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="status", data_type=FieldDataType.ENUM, enum_values=["a", "b", "c"])],
        )
        record = self.generator.generate_record(schema.schema_id)
        self.assertIn(record["status"], ["a", "b", "c"])

    def test_generate_batch(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="id", data_type=FieldDataType.INTEGER, min_value=1, max_value=100)],
        )
        records = self.generator.generate_batch(schema.schema_id, count=10)
        self.assertEqual(len(records), 10)
        for record in records:
            self.assertTrue(1 <= record["id"] <= 100)

    def test_generate_record_invalid_schema(self):
        with self.assertRaises(ValueError):
            self.generator.generate_record("nonexistent")

    def test_nullable_field(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="value", data_type=FieldDataType.STRING, nullable=True)],
        )
        records = self.generator.generate_batch(schema.schema_id, count=100, null_probability=0.5)
        has_none = any(r["value"] is None for r in records)
        has_value = any(r["value"] is not None for r in records)
        self.assertTrue(has_none or has_value)

    def test_generate_anonymized_record(self):
        original = {"name": "John Doe", "email": "john@example.com"}
        anonymized = self.generator.generate_anonymized_record(original, ["name", "email"])
        self.assertNotEqual(anonymized["name"], "John Doe")
        self.assertNotEqual(anonymized["email"], "john@example.com")

    def test_generate_from_template(self):
        template = "Hello, {name}! Your order {order_id} is ready."
        result = self.generator.generate_from_template(template, {"name": "Alice", "order_id": "12345"})
        self.assertEqual(result, "Hello, Alice! Your order 12345 is ready.")

    def test_field_schema_to_dict(self):
        field = FieldSchema(name="test", data_type=FieldDataType.STRING, length=5)
        result = field.to_dict()
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["data_type"], FieldDataType.STRING.value)

    def test_data_schema_to_dict(self):
        schema = DataSchema(name="test")
        result = schema.to_dict()
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["fields"], [])

    def test_get_generator_metrics(self):
        self.generator.create_schema(name="schema1", fields=[])
        self.generator.create_schema(name="schema2", fields=[])
        metrics = self.generator.get_generator_metrics()
        self.assertEqual(metrics["total_schemas"], 2)
        self.assertIn("schema1", metrics["schemas"])

    def test_generate_text(self):
        schema = self.generator.create_schema(
            name="test",
            fields=[FieldSchema(name="description", data_type=FieldDataType.TEXT)],
        )
        record = self.generator.generate_record(schema.schema_id)
        self.assertIsInstance(record["description"], str)


if __name__ == "__main__":
    unittest.main()
