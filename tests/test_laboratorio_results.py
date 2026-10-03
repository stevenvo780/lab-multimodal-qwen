"""Offline lifecycle checks for the four actual laboratory renderers.

AST isolation avoids importing the Streamlit application. NumPy, LabItem,
fingerprinting, validation, embedding normalization and renderer control flow
run normally. UI, GPU embeddings, PCA/UMAP and chart output are synthetic; these
tests do not validate a browser, the GPU, or real dimensionality reduction.
"""
from __future__ import annotations

import __future__
import ast
import copy
import hashlib
import json
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from gpu_client import GPUClient, GPUClientError, encode_image


SOURCE_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://gpu.invalid"
TABS = ("matrix", "search", "arithmetic", "projection")


class StreamlitDouble:
    def __init__(self):
        self.session_state = {}
        self.values = {}
        self.clicked = False
        self.spinner_failure = None
        self.reset_output()

    def reset_output(self):
        self.charts = []
        self.rankings = []
        self.metrics = []
        self.tables = []
        self.errors = []
        self.infos = []
        self.warnings = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def subheader(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def caption(self, *args, **kwargs): pass
    def markdown(self, *args, **kwargs): pass
    def code(self, *args, **kwargs): pass

    def text_input(self, label, value="", *, key, **kwargs):
        return self.values.get(key, value)

    text_area = text_input

    def selectbox(self, label, options, *, key, **kwargs):
        return self.values.get(key, options[0])

    def checkbox(self, label, value=False, *, key, **kwargs):
        return self.values.get(key, value)

    def slider(self, label, low, high, value, *args, key, **kwargs):
        return self.values.get(key, value)

    def button(self, *args, **kwargs):
        return self.clicked

    def columns(self, count, **kwargs):
        return [self] * (count if isinstance(count, int) else len(count))

    def container(self, **kwargs):
        return nullcontext()

    def expander(self, *args, **kwargs):
        return nullcontext()

    @contextmanager
    def spinner(self, *args, **kwargs):
        if self.spinner_failure and self.spinner_failure[0] == "enter":
            raise self.spinner_failure[1]
        yield
        if self.spinner_failure and self.spinner_failure[0] == "exit":
            raise self.spinner_failure[1]

    def plotly_chart(self, figure, **kwargs): self.charts.append(figure)
    def dataframe(self, value, **kwargs): self.tables.append(value)
    def metric(self, *args): self.metrics.append(args)
    def error(self, message): self.errors.append(message)
    def info(self, message): self.infos.append(message)
    def warning(self, message): self.warnings.append(message)


class ProjectionDouble:
    def __init__(self, **kwargs):
        self.explained_variance_ratio_ = np.array([0.6, 0.3])

    def fit_transform(self, vectors):
        return vectors[:, :2].copy()


def load_lab_functions(ui):
    path = SOURCE_ROOT / "laboratorio.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {
        "LabItem", "fingerprint", "lines", "text_items", "normalized_matrix",
        "embed_items", "ranked_results", "show_stale_result", "count_slider",
        "matrix_tab", "search_tab", "arithmetic_input", "arithmetic_tab", "projection_tab",
    }
    constants = {"DEFAULT_QUERIES", "ALTERNATIVE_INSTRUCTION"}
    selected = [node for node in module.body
                if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
                or (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    if len(selected) != len(names) + len(constants):
        raise AssertionError("Expected actual renderer functions and their input helpers")
    namespace = {
        "__name__": __name__, "hashlib": hashlib, "json": json, "Path": Path,
        "dataclass": dataclass, "np": np, "st": ui,
        "GPUClientError": GPUClientError, "encode_image": encode_image,
        "PCA": ProjectionDouble,
        "heatmap": lambda values, *args, **kwargs: ("heatmap", values.copy()),
        "scatter_figure": lambda values, items, title: (title, values.copy()),
        "three_vector_figure": lambda values, show: ("triad", values.copy()),
        "three_vector_geometry": lambda values: (values[:, :3], []),
        "show_ranked": lambda results: ui.rankings.append(results),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class LaboratoryResultTests(unittest.TestCase):
    def setUp(self):
        for target in (
            "socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
            "socket.getaddrinfo", "requests.sessions.Session.request", "gpu_client.requests.request",
        ):
            self.enterContext(patch(target, side_effect=AssertionError("Network is forbidden in unit tests")))
        umap = types.ModuleType("umap")
        umap.UMAP = ProjectionDouble
        self.enterContext(patch.dict(sys.modules, {"umap": umap}))

    def prepare(self, tab):
        self.tab = tab
        self.state_key = tab + "_result"
        self.ui = StreamlitDouble()
        self.lab = load_lab_functions(self.ui)
        self.Item = self.lab["LabItem"]
        self.images = [self.Item("perro.jpg", "imagen", image=b"first-image"),
                       self.Item("gato.jpg", "imagen", image=b"second-image")]
        self.documents = [self.Item("Texto 1", "texto", text="un perro"),
                          self.Item("Texto 2", "texto", text="un gato")]
        self.instruction = "Retrieve related items."
        self.client = GPUClient(API_URL)
        self.calls = []
        self.fail_at = None
        self.failure = None
        self.marker = "first"
        self.client.embed = Mock(side_effect=self.embed)

    def embed(self, payload, *, instruction):
        # Every embedding substep must start with the previous result retired.
        self.assertNotIn(self.state_key, self.ui.session_state)
        self.calls.append((copy.deepcopy(payload), instruction))
        if len(self.calls) == self.fail_at:
            raise self.failure
        vectors = []
        for item in payload:
            raw = json.dumps([item, instruction, self.marker], sort_keys=True).encode()
            digest = hashlib.sha256(raw).digest()
            vector = np.zeros(2048)
            vector[:8] = [value / 255 + 0.01 for value in digest[:8]]
            vectors.append(vector.tolist())
        return vectors

    def run_tab(self, *, clicked=False):
        self.ui.reset_output()
        self.calls.clear()
        self.client.embed.reset_mock()
        self.ui.clicked = clicked
        if self.tab == "matrix":
            self.lab["matrix_tab"](self.client, self.images, self.instruction)
        elif self.tab == "search":
            self.lab["search_tab"](self.client, self.images, self.documents, self.instruction)
        else:
            self.lab[self.tab + "_tab"](self.client, self.images, self.documents)

    def succeed(self):
        self.run_tab(clicked=True)
        self.assertEqual(self.ui.errors, [])
        self.assertTrue(self.ui.charts or self.ui.rankings)
        return self.ui.session_state[self.state_key]

    def assert_no_output(self):
        self.assertEqual(self.ui.charts, [])
        self.assertEqual(self.ui.rankings, [])
        self.assertEqual(self.ui.metrics, [])
        self.assertEqual(self.ui.tables, [])

    def assert_retired(self):
        self.assertNotIn(self.state_key, self.ui.session_state)
        self.assert_no_output()

    def assert_stale(self):
        self.assert_no_output()
        self.assertTrue(any("entradas cambiaron" in message for message in self.ui.infos))
        self.client.embed.assert_not_called()

    def test_success_publishes_complete_result_for_each_renderer(self):
        expected_fields = {
            "matrix": {"key", "scores", "comparison"},
            "search": {"key", "ranked"},
            "arithmetic": {"key", "ranked", "magnitude", "ab", "ac", "triad"},
            "projection": {"key", "pca_points", "pca_variance", "umap_points", "umap_error", "nearest"},
        }
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                result = self.succeed()
                self.assertEqual(set(result), expected_fields[tab])
                self.assertEqual(len(result["key"]), 64)
                self.assertEqual(len(self.calls), 2 if tab in ("matrix", "search") else 1)
                instructions = [instruction for _, instruction in self.calls]
                self.assertEqual(instructions, [None, self.instruction] if tab == "matrix" else
                                 [self.instruction, None] if tab == "search" else [None])

    def test_unchanged_rerun_reuses_success_without_embeddings(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                saved = self.succeed()
                self.run_tab()
                self.assertIs(self.ui.session_state[self.state_key], saved)
                self.assertTrue(self.ui.charts or self.ui.rankings)
                self.client.embed.assert_not_called()

    def test_endpoint_change_hides_previous_output_and_does_not_run(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                saved = self.succeed()
                self.client.base_url = "https://second-gpu.invalid"
                self.run_tab()
                self.assert_stale()
                replacement = self.succeed()
                self.assertNotEqual(replacement["key"], saved["key"])

    def test_equivalent_normalized_endpoint_keeps_current_success(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                saved = self.succeed()
                self.client.base_url = GPUClient(API_URL + "///").base_url
                self.run_tab()
                self.assertIs(self.ui.session_state[self.state_key], saved)
                self.assertTrue(self.ui.charts or self.ui.rankings)
                self.client.embed.assert_not_called()

    def test_same_filename_changed_image_bytes_hides_output(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()
                self.images[0] = self.Item("perro.jpg", "imagen", image=b"changed-image")
                self.run_tab()
                self.assert_stale()

    def test_changed_text_hides_output(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()
                if tab == "matrix":
                    self.ui.values["matrix_queries"] = "una consulta distinta"
                elif tab == "search":
                    self.ui.values["search_query"] = "una consulta distinta"
                elif tab == "arithmetic":
                    self.ui.values["arithmetic_B_text"] = "una entrada distinta"
                else:
                    self.documents[0] = self.Item("Texto 1", "texto", text="texto distinto")
                self.run_tab()
                self.assert_stale()

    def test_changed_candidate_order_hides_output(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()
                self.images.reverse()
                self.run_tab()
                self.assert_stale()

    def test_changed_search_or_matrix_instruction_hides_output(self):
        for tab in ("matrix", "search"):
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()
                self.instruction = "A different retrieval instruction."
                self.run_tab()
                self.assert_stale()

    def test_changed_experiment_settings_hide_output(self):
        settings = (("matrix", "matrix_compare", True),
                    ("search", "search_top_k", 1),
                    ("search", "search_filename", True),
                    ("search", "search_image", "perro.jpg"),
                    ("arithmetic", "arithmetic_top_k", 1),
                    ("arithmetic", "arithmetic_C_image", "gato.jpg"),
                    ("projection", "projection_neighbors", 2),
                    ("projection", "projection_min_dist", 0.4))
        for tab, key, value in settings:
            with self.subTest(tab=tab, setting=key):
                self.prepare(tab)
                self.succeed()
                self.ui.values[key] = value
                self.run_tab()
                self.assert_stale()

    def test_matrix_alternative_instruction_participates_in_identity(self):
        self.prepare("matrix")
        self.ui.values["matrix_compare"] = True
        self.succeed()
        self.ui.values["matrix_alternative"] = "New alternative instruction."
        self.run_tab()
        self.assert_stale()

    def test_failed_same_input_retry_retires_result_and_cannot_resurrect(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()
                self.fail_at = 1
                self.failure = GPUClientError("Synthetic embedding failure")
                self.run_tab(clicked=True)
                self.assert_retired()
                self.assertEqual(self.ui.errors, [str(self.failure)])
                self.run_tab()
                self.assert_retired()
                self.client.embed.assert_not_called()

    def test_unexpected_embedding_failure_or_interruption_retires_result(self):
        for tab in TABS:
            for failure_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(tab=tab, failure=failure_type.__name__):
                    self.prepare(tab)
                    self.succeed()
                    self.fail_at = 1
                    self.failure = failure_type("Synthetic interrupted embedding")
                    with self.assertRaises(failure_type) as raised:
                        self.run_tab(clicked=True)
                    self.assertIs(raised.exception, self.failure)
                    self.assert_retired()
                    self.run_tab()
                    self.assert_retired()
                    self.client.embed.assert_not_called()

    def test_late_embedding_failure_does_not_publish_partial_matrix_or_search(self):
        for tab, compare, failed_call in (("matrix", False, 2), ("matrix", True, 3), ("search", False, 2)):
            with self.subTest(tab=tab, failed_call=failed_call):
                self.prepare(tab)
                self.ui.values["matrix_compare"] = compare
                self.succeed()
                self.fail_at = failed_call
                self.failure = GPUClientError("Synthetic late embedding failure")
                self.run_tab(clicked=True)
                self.assert_retired()
                self.assertEqual(len(self.calls), failed_call)
                self.assertEqual(self.ui.errors, [str(self.failure)])

    def test_invalid_retry_clears_result_before_validation(self):
        cases = (("matrix", "empty_query"), ("matrix", "empty_collection"),
                 ("search", "empty_query"), ("search", "empty_collection"),
                 ("arithmetic", "empty_query"), ("arithmetic", "empty_collection"),
                 ("projection", "empty_collection"))
        for tab, case in cases:
            with self.subTest(tab=tab, invalid=case):
                self.prepare(tab)
                self.succeed()
                original_images, original_documents = self.images, self.documents
                if case == "empty_collection":
                    self.images, self.documents = [], []
                else:
                    key = {"matrix": "matrix_queries", "search": "search_query",
                           "arithmetic": "arithmetic_A_text"}[tab]
                    self.ui.values[key] = " \n "
                self.run_tab(clicked=True)
                self.assert_retired()
                self.assertEqual(len(self.ui.errors), 1)
                self.client.embed.assert_not_called()
                self.images, self.documents = original_images, original_documents
                self.ui.values.clear()
                self.run_tab()
                self.assert_retired()

    def test_api_item_validation_interruption_cannot_restore_old_result(self):
        for tab in ("search", "arithmetic"):
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()

                def interrupt_validation(item):
                    self.assertNotIn(self.state_key, self.ui.session_state)
                    raise KeyboardInterrupt("Synthetic validation interruption")

                with patch.object(self.Item, "api_item", interrupt_validation):
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_tab(clicked=True)
                self.assert_retired()
                self.client.embed.assert_not_called()

    def test_spinner_interruption_never_publishes_result(self):
        for tab in TABS:
            for phase in ("enter", "exit"):
                with self.subTest(tab=tab, phase=phase):
                    self.prepare(tab)
                    self.succeed()
                    self.ui.spinner_failure = (phase, KeyboardInterrupt("Synthetic spinner interruption"))
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_tab(clicked=True)
                    self.assert_retired()
                    self.ui.spinner_failure = None
                    self.run_tab()
                    self.assert_retired()
                    self.client.embed.assert_not_called()

    def test_local_computation_failure_retires_output_after_embeddings(self):
        for tab, helper in (("search", "ranked_results"), ("arithmetic", "ranked_results"),
                            ("projection", "PCA")):
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()
                failure = RuntimeError("Synthetic local computation failure")
                with patch.dict(self.lab, {helper: Mock(side_effect=failure)}):
                    with self.assertRaises(RuntimeError) as raised:
                        self.run_tab(clicked=True)
                self.assertIs(raised.exception, failure)
                self.assert_retired()
                self.assertTrue(self.calls)

    def test_successful_recovery_publishes_fresh_complete_result(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                old = self.succeed()
                self.fail_at = 1
                self.failure = GPUClientError("Synthetic failure before recovery")
                self.run_tab(clicked=True)
                self.assert_retired()
                self.fail_at = None
                self.marker = "recovered"
                recovered = self.succeed()
                self.assertIsNot(recovered, old)
                self.assertEqual(recovered["key"], old["key"])
                field = {"matrix": "scores", "search": "ranked", "arithmetic": "triad",
                         "projection": "pca_points"}[tab]
                self.assertNotEqual(repr(recovered[field]), repr(old[field]))
                self.run_tab()
                self.assertIs(self.ui.session_state[self.state_key], recovered)
                self.client.embed.assert_not_called()

    def test_retry_only_retires_its_own_experiment_result(self):
        for tab in TABS:
            with self.subTest(tab=tab):
                self.prepare(tab)
                self.succeed()
                unrelated = {other + "_result": {"sentinel": other} for other in TABS if other != tab}
                self.ui.session_state.update(unrelated)
                self.fail_at = 1
                self.failure = GPUClientError("Synthetic failure")
                self.run_tab(clicked=True)
                self.assert_retired()
                for key, value in unrelated.items():
                    self.assertIs(self.ui.session_state[key], value)

    def test_matrix_blank_alternative_retires_previous_complete_comparison(self):
        self.prepare("matrix")
        self.ui.values["matrix_compare"] = True
        saved = self.succeed()
        self.assertIsNotNone(saved["comparison"])
        self.ui.values["matrix_alternative"] = " \n "
        self.run_tab(clicked=True)
        self.assert_retired()
        self.assertEqual(len(self.ui.errors), 1)
        self.assertEqual(len(self.calls), 2)

    def test_arithmetic_near_zero_composite_retires_previous_result(self):
        self.prepare("arithmetic")
        self.succeed()
        vectors = np.zeros((7, 2048))
        vectors[0, 0] = 1
        vectors[1, 0] = 1
        with patch.dict(self.lab, {"embed_items": Mock(return_value=vectors)}):
            self.run_tab(clicked=True)
        self.assert_retired()
        self.assertIn("casi nulo", self.ui.errors[0])

    def test_projection_retains_fresh_pca_fallback_when_umap_fails(self):
        for failure_type in (ImportError, ValueError, RuntimeError):
            with self.subTest(failure=failure_type.__name__):
                self.prepare("projection")
                old = self.succeed()
                self.marker = "fresh-pca"
                failure = failure_type("Synthetic unavailable UMAP")
                with patch.object(sys.modules["umap"], "UMAP", side_effect=failure):
                    result = self.succeed()
                self.assertIsNot(result, old)
                self.assertIsNone(result["umap_points"])
                self.assertEqual(result["umap_error"], str(failure))
                self.assertNotEqual(repr(result["pca_points"]), repr(old["pca_points"]))
                self.assertEqual(len(self.ui.charts), 1)
                self.assertEqual(len(self.ui.warnings), 1)

    def test_projection_umap_interruption_does_not_publish_partial_pca(self):
        self.prepare("projection")
        self.succeed()
        with patch.object(sys.modules["umap"], "UMAP", side_effect=KeyboardInterrupt("Synthetic UMAP stop")):
            with self.assertRaises(KeyboardInterrupt):
                self.run_tab(clicked=True)
        self.assert_retired()


if __name__ == "__main__":
    unittest.main()
