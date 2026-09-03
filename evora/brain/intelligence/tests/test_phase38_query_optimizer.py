"""
Phase 38 — Tests for Native Query Optimizer.

Tests query plan optimization capabilities.
"""

import unittest
from evora.brain.intelligence.query_optimizer import (
    NativeQueryOptimizer,
    QueryPlan,
    QueryPlanNode,
)


class TestNativeQueryOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = NativeQueryOptimizer()

    def test_create_plan(self):
        plan = self.optimizer.create_plan("SELECT * FROM users")
        self.assertEqual(plan.query, "SELECT * FROM users")
        self.assertIsNotNone(plan.root)
        self.assertFalse(plan.cached)

    def test_create_plan_with_cache(self):
        query = "SELECT * FROM users"
        plan1 = self.optimizer.create_plan(query)
        plan2 = self.optimizer.create_plan(query)
        self.assertTrue(plan2.cached)
        self.assertEqual(plan1.plan_id, plan2.plan_id)

    def test_create_plan_invalid(self):
        plan = self.optimizer.create_plan("invalid query")
        self.assertIsNotNone(plan.root)
        self.assertIsNotNone(plan.plan_id)

    def test_optimize_plan(self):
        plan = self.optimizer.create_plan("SELECT * FROM users WHERE age > 18")
        original_cost = plan.estimated_cost
        optimized = self.optimizer.optimize_plan(plan)
        self.assertLessEqual(optimized.optimized_cost, original_cost)

    def test_optimize_plan_no_root(self):
        plan = QueryPlan(plan_id="test", query="")
        optimized = self.optimizer.optimize_plan(plan)
        self.assertIsNone(optimized.root)

    def test_execute_plan(self):
        plan = self.optimizer.create_plan("SELECT * FROM users")
        result = self.optimizer.execute_plan(plan)
        self.assertIn("result", result)
        self.assertIn("plan_id", result)
        self.assertFalse(result["cached"])

    def test_execute_plan_cached(self):
        plan = self.optimizer.create_plan("SELECT * FROM users")
        self.optimizer.execute_plan(plan)
        plan2 = self.optimizer.create_plan("SELECT * FROM users")
        self.assertTrue(plan2.cached)

    def test_predicate_pushdown(self):
        plan = self.optimizer.create_plan("SELECT * FROM users WHERE age > 18")
        optimized = self.optimizer.optimize_plan(plan)
        self.assertEqual(optimized.root.operation, "filtered_scan")

    def test_get_stats(self):
        plan = self.optimizer.create_plan("SELECT * FROM users")
        self.optimizer.execute_plan(plan)
        stats = self.optimizer.get_stats()
        self.assertIn(plan.plan_id, stats)
        self.assertEqual(stats[plan.plan_id]["executions"], 1)

    def test_get_cache_info(self):
        self.optimizer.create_plan("SELECT * FROM users")
        info = self.optimizer.get_cache_info()
        self.assertEqual(info["total_cached"], 1)

    def test_query_plan_node_to_dict(self):
        node = QueryPlanNode(node_id="test", operation="scan", table="users")
        result = node.to_dict()
        self.assertEqual(result["node_id"], "test")
        self.assertEqual(result["operation"], "scan")
        self.assertEqual(result["table"], "users")

    def test_query_plan_to_dict(self):
        plan = QueryPlan(plan_id="test", query="SELECT 1")
        result = plan.to_dict()
        self.assertEqual(result["plan_id"], "test")
        self.assertEqual(result["query"], "SELECT 1")

    def test_execute_plan_no_root(self):
        plan = QueryPlan(plan_id="test", query="")
        result = self.optimizer.execute_plan(plan_c:=plan)
        self.assertEqual(result["result"], {})

    def test_estimate_cost(self):
        plan = self.optimizer.create_plan("SELECT * FROM users")
        self.assertGreater(plan.estimated_cost, 0)


if __name__ == "__main__":
    unittest.main()
