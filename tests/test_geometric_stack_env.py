import itertools
import json
import os
import unittest

from geometric_stack_bench.geometric_stack_env import GeometricStackEnvironment, ShapeType, export_dataset, validate_dataset_file


class TestGeometricStackEnv(unittest.TestCase):
    @staticmethod
    def _is_half_grid(value: float) -> bool:
        return abs(value * 2 - round(value * 2)) < 1e-9 and abs(value - round(value)) > 1e-9

    def test_generate_unique_task(self):
        env = GeometricStackEnvironment(seed=11)
        task = env.generate_task(difficulty="easy", require_unique_order=True)
        self.assertEqual(task.valid_order_count, 1)

        # Validate uniqueness by brute-force image matching.
        ids = [p.piece_id for p in task.pieces]
        pieces_by_id = {p.piece_id: p for p in task.pieces}
        matches = 0
        for perm in itertools.permutations(ids):
            img = env.render_order(pieces_by_id, perm, task.width, task.height)
            if img == task.final_image:
                matches += 1
        self.assertEqual(matches, 1)

    def test_evaluate_order_exact(self):
        env = GeometricStackEnvironment(seed=13)
        task = env.generate_task(difficulty="medium", require_unique_order=True)
        result = env.evaluate_order(task, task.true_order)
        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["valid_order_match"], 1.0)
        self.assertAlmostEqual(result["reward"], 1.0)
        self.assertEqual(result["correct_actions"], float(len(task.true_order)))

    def test_evaluate_order_partial(self):
        env = GeometricStackEnvironment(seed=17)
        task = env.generate_task(difficulty="easy", require_unique_order=True)
        pred = task.true_order[::-1]
        result = env.evaluate_order(task, pred)
        self.assertEqual(result["success"], 0.0)
        self.assertGreaterEqual(result["reward"], 0.0)
        self.assertLess(result["reward"], 1.0)
        self.assertGreaterEqual(result["correct_actions"], 0.0)
        self.assertLessEqual(result["correct_actions"], float(len(task.true_order)))

    def test_invalid_order_penalty(self):
        env = GeometricStackEnvironment(seed=19)
        task = env.generate_task(difficulty="easy", require_unique_order=True)
        bad = task.true_order[:-1]  # missing piece
        result = env.evaluate_order(task, bad)
        self.assertEqual(result["reward"], -1.0)

    def test_evaluate_layers_exact(self):
        env = GeometricStackEnvironment(seed=21)
        task = env.generate_task(difficulty="medium", require_unique_order=True)
        layers = env.order_to_layers(task, task.true_order)
        result = env.evaluate_layers(task, layers)
        self.assertEqual(result["strict_success"], 1.0)
        self.assertAlmostEqual(result["reward"], 1.0)

    def test_evaluate_layers_requires_metadata(self):
        env = GeometricStackEnvironment(seed=22)
        task = env.generate_task(difficulty="easy", require_unique_order=True)
        layers = env.order_to_layers(task, task.true_order)
        # Corrupt shape while preserving z order.
        layers[0]["shape"] = "triangle" if layers[0]["shape"] != "triangle" else "square"
        result = env.evaluate_layers(task, layers)
        self.assertEqual(result["strict_success"], 0.0)
        self.assertLess(result["metadata_accuracy"], 1.0)
        self.assertLess(result["reward"], 1.0)

    def test_standard_canvas_and_half_precision_anchors(self):
        env = GeometricStackEnvironment(seed=29, canvas_size=16)
        for difficulty in ["easy", "medium", "hard", "expert"]:
            task = env.generate_task(difficulty=difficulty, require_unique_order=True, max_generation_tries=800)
            self.assertEqual(task.width, 16)
            self.assertEqual(task.height, 16)
            for p in task.pieces:
                self.assertTrue(self._is_half_grid(p.anchor_x), msg=f"{p.piece_id} anchor_x={p.anchor_x}")
                self.assertTrue(self._is_half_grid(p.anchor_y), msg=f"{p.piece_id} anchor_y={p.anchor_y}")

    def test_triangle_signature_visibility_constraint(self):
        env = GeometricStackEnvironment(seed=33)
        found_triangle_task = False
        for _ in range(20):
            task = env.generate_task(difficulty="hard", require_unique_order=True, max_generation_tries=1000)
            if any(p.shape == ShapeType.TRIANGLE for p in task.pieces):
                found_triangle_task = True
                pieces_by_id = {p.piece_id: p for p in task.pieces}
                _img, owner = env._render_order_with_owner(pieces_by_id, task.true_order, task.width, task.height)
                for p in task.pieces:
                    if p.shape == ShapeType.TRIANGLE:
                        self.assertTrue(env._triangle_has_signature_visibility(p, owner))
                break
        self.assertTrue(found_triangle_task)

    def test_prompt_contains_xy_axis_labels(self):
        env = GeometricStackEnvironment(seed=31)
        task = env.generate_task(difficulty="easy", require_unique_order=True)
        prompt = env.format_task_for_llm(task)
        self.assertIn("Top-left cell is (0,0)", prompt)
        self.assertIn("anchor=(", prompt)
        self.assertIn("axis=(", prompt)
        self.assertIn("00 ", prompt)  # y-label prefix in rendered image block

    def test_ambiguous_orders_reward_if_reconstructs(self):
        env = GeometricStackEnvironment(seed=23)
        task = env.generate_task(difficulty="easy", require_unique_order=False)
        # Ensure we can find an ambiguous sample.
        tries = 0
        while task.valid_order_count <= 1 and tries < 30:
            task = env.generate_task(difficulty="easy", require_unique_order=False)
            tries += 1
        self.assertGreater(task.valid_order_count, 1)
        self.assertIsNotNone(task.valid_orders)

        alt = None
        for order in task.valid_orders:
            if order != task.true_order:
                alt = order
                break
        self.assertIsNotNone(alt)
        result = env.evaluate_order(task, alt)
        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["valid_order_match"], 1.0)
        self.assertAlmostEqual(result["reward"], 1.0)

    def test_export_dataset_integrity(self):
        out = "geometric_stack_bench/tmp_dataset_test.json"
        export_dataset(out, num_tasks=12, seed=42, require_unique_order=False)
        with open(out, "r", encoding="utf-8") as f:
            rows = json.load(f)
        self.assertEqual(len(rows), 12)
        for r in rows:
            self.assertIn("pieces", r)
            self.assertIn("final_image", r)
            self.assertIn("valid_order_count", r)
            self.assertIn("valid_orders", r)
            self.assertIn("answer_layers", r)
            self.assertEqual(r["width"], 16)
            self.assertEqual(r["height"], 16)
            self.assertGreaterEqual(r["valid_order_count"], 1)
            self.assertGreaterEqual(len(r["valid_orders"]), 1)
            for p in r["pieces"]:
                self.assertIn("anchor_x", p)
                self.assertIn("anchor_y", p)
                self.assertIn("axis_x", p)
                self.assertIn("axis_y", p)
                self.assertTrue(self._is_half_grid(float(p["anchor_x"])))
                self.assertTrue(self._is_half_grid(float(p["anchor_y"])))

        report = validate_dataset_file(out)
        self.assertEqual(report["issue_count"], 0, msg="\n".join(report["issues"]))
        os.remove(out)


if __name__ == "__main__":
    unittest.main()
