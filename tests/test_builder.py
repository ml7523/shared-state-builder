import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from state_builder.backend import Generation
from state_builder.cuts import build_records, select_cut_positions
from state_builder.dataset import DATA_PATH, download_math500, load_math500
from state_builder.run import main


class FakeSingleBackend:
    method = "single"
    enable_thinking = True
    think_start_id = 1
    think_end_id = 7
    eos_ids = {0}

    def __init__(self, model_name, revision=None):
        self.model_name = model_name
        self.revision = revision
        self.calls = []

    def model_revision(self):
        return "fake-commit"

    def tokenizer_fingerprint(self):
        return "fake-tokenizer"

    def generation_signature(self):
        return "fake-signature"

    def decode(self, ids):
        return " ".join(str(token) for token in ids)

    def generate_trajectory(self, question, prompt_template, *, budget, seed,
                            temperature, top_p, top_k):
        self.calls.append((question, prompt_template, budget, seed,
                           temperature, top_p, top_k))
        return Generation([101, 102], [1, 11, 12, 13, 14, 15, 7, 20, 0],
                          prompt_template.format(question=question),
                          "mock rendered chat", "eos", 0.25)


class CutTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeSingleBackend("test-model")
        self.row = {"unique_id": "test/algebra/1.json", "problem": "Solve 6x=42",
                    "answer": "7"}

    def test_uniform_cuts_are_inside_unfinished_thinking(self):
        generation = self.backend.generate_trajectory(
            self.row["problem"], "Question: {question}", budget=20, seed=42,
            temperature=0.6, top_p=0.95, top_k=20,
        )
        trajectory, states, labels = build_records(
            backend=self.backend, generation=generation, dataset_index=10,
            row=self.row, total_budget=20, seed=42, num_cut_points=3,
        )
        self.assertEqual(trajectory["output_ids"], [1, 11, 12, 13, 14, 15, 7, 20, 0])
        self.assertEqual(trajectory["reasoning_ids"], [1, 11, 12, 13, 14, 15])
        self.assertEqual(trajectory["answer_ids"], [20])
        self.assertEqual(trajectory["selected_cut_tokens"], [2, 4, 5])
        self.assertEqual([state["spent_tokens"] for state in states], [2, 4, 5])
        self.assertEqual([state["remaining_tokens"] for state in states], [18, 16, 15])
        self.assertEqual(states[1]["prefix_ids"], [1, 11, 12, 13])
        self.assertTrue(all(7 not in state["prefix_ids"] and 0 not in state["prefix_ids"]
                            for state in states))
        self.assertTrue(all("gold_answer" not in state for state in states))
        self.assertTrue(all(state["quantization"] == "none" for state in states))
        self.assertEqual([label["state_id"] for label in labels],
                         [state["state_id"] for state in states])
        self.assertTrue(all(label["gold_answer"] == "7" for label in labels))

    def test_explicit_positions_report_out_of_range_and_too_short_uniform(self):
        positions, rejected, reason = select_cut_positions(
            2, 6, num_cut_points=None, cut_tokens=[2, 6, 8]
        )
        self.assertEqual((positions, rejected, reason), ([2, 6], [8], None))
        self.assertEqual(select_cut_positions(2, 3, num_cut_points=3,
                                              cut_tokens=None),
                         ([], [], "too_short_for_requested_cuts"))

    def test_missing_think_start_is_saved_but_never_exposed_as_state(self):
        generation = Generation([101], [11, 12, 0], "u", "c", "eos", 0.1)
        trajectory, states, labels = build_records(
            backend=self.backend, generation=generation, dataset_index=0,
            row=self.row, total_budget=10, seed=42, num_cut_points=1,
        )
        self.assertEqual(trajectory["state_skip_reason"], "missing_think_start")
        self.assertEqual((states, labels), ([], []))

    def test_pinned_dataset_is_independent_and_valid(self):
        self.assertEqual(len(load_math500()), 500)

    def test_download_verifies_checksum_before_publishing(self):
        class Response:
            def __init__(self, data):
                self.data = data
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def read(self):
                return self.data

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "math500_test.jsonl"
            with patch("state_builder.dataset.urlopen", return_value=Response(b"wrong")):
                with self.assertRaisesRegex(ValueError, "pinned test split"):
                    download_math500(target)
            self.assertFalse(target.exists())
            with patch("state_builder.dataset.urlopen",
                       return_value=Response(DATA_PATH.read_bytes())):
                self.assertEqual(download_math500(target), target)
            self.assertEqual(target.read_bytes(), DATA_PATH.read_bytes())


class RunnerTests(unittest.TestCase):
    def test_shared_run_defaults_to_qwen3_14b(self):
        with TemporaryDirectory() as tmp:
            requested_models = []

            def factory(model_name, revision=None):
                requested_models.append(model_name)
                return FakeSingleBackend(model_name, revision)

            self.assertEqual(main(["--output", str(Path(tmp) / "shared"),
                                   "--num-problems", "1", "--total-budget", "20"],
                                  backend_factory=factory), 0)
            self.assertEqual(requested_models, ["Qwen/Qwen3-14B"])

    def test_one_generation_per_problem_resume_and_separate_labels(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "shared"
            created = []

            def factory(model_name, revision=None):
                backend = FakeSingleBackend(model_name, revision)
                created.append(backend)
                return backend

            args = ["--output", str(output), "--model-name", "test-model",
                    "--start-index", "10", "--num-problems", "2",
                    "--total-budget", "20", "--num-cut-points", "2"]
            self.assertEqual(main(args, backend_factory=factory), 0)
            self.assertEqual(len(created[0].calls), 2)
            trajectories = [json.loads(line) for line in
                            (output / "trajectories.jsonl").read_text().splitlines()]
            states = [json.loads(line) for line in
                      (output / "states.jsonl").read_text().splitlines()]
            labels = [json.loads(line) for line in
                      (output / "labels.jsonl").read_text().splitlines()]
            self.assertEqual((len(trajectories), len(states), len(labels)), (2, 4, 4))
            self.assertEqual({state["problem_id"] for state in states},
                             {trajectory["problem_id"] for trajectory in trajectories})
            self.assertTrue(all("gold_answer" not in row for row in trajectories + states))
            saved = (output / "states.jsonl").read_bytes()
            self.assertEqual(main(args + ["--resume"], backend_factory=factory), 0)
            self.assertEqual(len(created[1].calls), 0)
            self.assertEqual((output / "states.jsonl").read_bytes(), saved)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["physical_collection_generated_tokens"], 18)
            self.assertEqual(summary["shared_states"], 4)


if __name__ == "__main__":
    unittest.main()
