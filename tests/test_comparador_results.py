"""Offline lifecycle checks of the actual comparator renderer and cache.

AST isolation substitutes Streamlit, charts and PCA. Input validation, Pillow,
signatures, normalization and embedding-cache control flow run unchanged.
These fixtures do not validate a browser, real PCA, GPU inference or HTTP.
"""
from __future__ import annotations

import __future__
import ast
import copy
import hashlib
import io
import json
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from gpu_client import GPUClient, GPUClientError, encode_image
from tests.test_laboratorio_results import StreamlitDouble


SOURCE_ROOT = Path(__file__).resolve().parents[1]
RESULT = "comparador_result"
QUERY = "comparador_query_result"
CACHE = "comparador_embedding_cache"


def picture(color="red"):
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="PNG")
    return output.getvalue()


class Upload(io.BytesIO):
    def __init__(self, data, name="same-name.png"):
        super().__init__(data)
        self.name = name


class ComparatorUI(StreamlitDouble):
    def __init__(self):
        super().__init__()
        self.buttons = set()
        self.spinner_phase = None
        self.spinner_error = None

    def title(self, *args): pass
    def header(self, *args): pass

    def file_uploader(self, *args, key, **kwargs):
        return self.values.get(key)

    def radio(self, *args, **kwargs):
        return self.selectbox(*args, **kwargs)

    def button(self, *args, key, **kwargs):
        return key in self.buttons

    @contextmanager
    def spinner(self, *args, **kwargs):
        if self.spinner_phase == "enter":
            raise self.spinner_error
        yield
        if self.spinner_phase == "exit":
            raise self.spinner_error


class Projection:
    def __init__(self, **kwargs):
        self.explained_variance_ratio_ = np.array([0.6, 0.3, 0.1])

    def fit_transform(self, matrix):
        return matrix[:, :3].copy()


def load_comparator(ui):
    path = SOURCE_ROOT / "comparador_lab.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"CompareItem", "_validate_image", "_from_upload", "_normalized_matrix",
             "_embed_cached", "_collection_signature", "_pca_3d", "_inputs", "render_comparator"}
    constants = {"MAX_IMAGE_BYTES", "MAX_IMAGE_PIXELS", "MAX_EXTRA_IMAGES", "MAX_EXTRA_TEXTS", "MAX_TEXT_CHARS"}
    selected = [node for node in parsed.body
                if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
                or (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in constants for target in node.targets))]
    if len(selected) != len(names) + len(constants):
        raise AssertionError("Expected actual comparator renderer, validation and cache helpers")
    namespace = {
        "__name__": __name__, "hashlib": hashlib, "io": io, "json": json,
        "dataclass": dataclass, "np": np, "Image": Image, "st": ui,
        "GPUClientError": GPUClientError, "encode_image": encode_image,
        "EXPECTED_DIMENSION": 2048, "PCA": Projection,
        "_pca_figure": lambda points, items: ("pca", points.copy()),
        "_triangle_figure": lambda matrix: ("triangle", matrix.copy()),
        "_pair_metrics": lambda matrix: [], "_interior_angles": lambda matrix: [],
        "_render_neighbors": lambda items, matrix, query, excluded, top_k:
            ui.rankings.append((matrix.copy(), query.copy(), excluded, top_k)),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class ComparatorResultTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
                       "socket.getaddrinfo", "requests.sessions.Session.request", "gpu_client.requests.request"):
            self.enterContext(patch(target, side_effect=AssertionError("Network is forbidden in unit tests")))
        self.prepare()

    def prepare(self):
        self.ui = ComparatorUI()
        self.lab = load_comparator(self.ui)
        self.client = GPUClient("https://gpu.invalid")
        self.calls = []
        self.failure = None
        self.fail_at = None
        self.client.embed = Mock(side_effect=self.embed)

    def embed(self, payload, *, instruction):
        if "comparador_calculate" in self.ui.buttons:
            self.assertNotIn(RESULT, self.ui.session_state)
            self.assertNotIn(QUERY, self.ui.session_state)
        elif "comparador_query_calculate" in self.ui.buttons:
            self.assertNotIn(QUERY, self.ui.session_state)
        self.calls.append(copy.deepcopy(payload))
        if len(self.calls) == self.fail_at:
            raise self.failure
        vectors = []
        for item in payload:
            digest = hashlib.sha256(json.dumps([item, instruction], sort_keys=True).encode()).digest()
            vector = np.zeros(2048)
            vector[:8] = np.array(list(digest[:8])) / 255 + 0.01
            vectors.append(vector.tolist())
        return vectors

    def render(self, *buttons):
        self.ui.reset_output()
        self.calls.clear()
        self.client.embed.reset_mock()
        self.ui.buttons = set(buttons)
        self.lab["render_comparator"](self.client)

    def collection(self):
        self.render("comparador_calculate")
        self.assertEqual(self.ui.errors, [])
        self.assertEqual(len(self.ui.charts), 2)
        result = self.ui.session_state[RESULT]
        self.assertEqual(set(result), {"signature", "matrix", "points", "explained"})
        return result

    def query(self):
        if RESULT not in self.ui.session_state:
            self.collection()
        self.ui.values["comparador_query_source"] = "Nueva entrada"
        self.ui.values.setdefault("comparador_query_text", "consulta original")
        self.render("comparador_query_calculate")
        self.assertEqual(self.ui.errors, [])
        self.assertEqual(len(self.ui.rankings), 1)
        result = self.ui.session_state[QUERY]
        self.assertEqual(set(result), {"signature", "vector"})
        return result

    def assert_collection_retired(self):
        self.assertNotIn(RESULT, self.ui.session_state)
        self.assertNotIn(QUERY, self.ui.session_state)
        self.assertEqual(self.ui.charts, [])
        self.assertEqual(self.ui.rankings, [])

    def assert_query_retired(self):
        self.assertNotIn(QUERY, self.ui.session_state)
        self.assertIn(RESULT, self.ui.session_state)
        self.assertEqual(self.ui.rankings, [])

    def test_identical_rerun_reuses_complete_collection_and_query(self):
        collection = self.collection()
        query = self.query()
        cache = self.ui.session_state[CACHE]
        self.render()
        self.assertIs(self.ui.session_state[RESULT], collection)
        self.assertIs(self.ui.session_state[QUERY], query)
        self.assertIs(self.ui.session_state[CACHE], cache)
        self.assertEqual(len(self.ui.charts), 2)
        self.assertEqual(len(self.ui.rankings), 1)
        self.client.embed.assert_not_called()

    def test_collection_change_and_return_never_resurrect_results(self):
        changes = {"comparador_A_text": "different", "comparador_instruction": "different",
                   "comparador_extra_texts": "different", "comparador_A_image": Upload(picture()),
                   "comparador_extra_images": [Upload(picture())]}
        for key, value in changes.items():
            with self.subTest(input=key):
                self.prepare()
                self.query()
                self.ui.values[key] = value
                self.render()
                self.assert_collection_retired()
                self.client.embed.assert_not_called()
                self.ui.values.pop(key)
                self.render()
                self.assert_collection_retired()
                self.client.embed.assert_not_called()

    def test_endpoint_change_and_return_retires_both_results(self):
        self.query()
        self.client.base_url = "https://other-gpu.invalid"
        self.render()
        self.assert_collection_retired()
        self.client.base_url = "https://gpu.invalid"
        self.render()
        self.assert_collection_retired()

    def test_equivalent_trimmed_text_and_normalized_endpoint_reuse_success(self):
        saved = self.collection()
        self.ui.values["comparador_A_text"] = " \n un perro en la playa \t"
        self.client.base_url = GPUClient("https://gpu.invalid///").base_url
        self.render()
        self.assertIs(self.ui.session_state[RESULT], saved)
        self.client.embed.assert_not_called()

    def test_same_image_filename_with_changed_content_retires_result(self):
        self.ui.values["comparador_A_image"] = Upload(picture())
        self.collection()
        self.ui.values["comparador_A_image"] = Upload(picture("blue"))
        self.render()
        self.assert_collection_retired()

    def test_collection_validation_failure_and_return_retire_both_results(self):
        cases = {"comparador_A_text": " \n ", "comparador_A_image": Upload(b"invalid"),
                 "comparador_extra_texts": "line\n" * 31,
                 "comparador_extra_images": [Upload(picture())] * 21}
        for clicked in (False, True):
            for key, value in cases.items():
                with self.subTest(input=key, clicked=clicked):
                    self.prepare()
                    self.query()
                    self.ui.values[key] = value
                    self.render(*(["comparador_calculate"] if clicked else []))
                    self.assert_collection_retired()
                    self.assertEqual(len(self.ui.errors), 1)
                    self.client.embed.assert_not_called()
                    self.ui.values.pop(key)
                    self.render()
                    self.assert_collection_retired()

    def test_collection_input_preparation_exception_or_interruption_retires_both(self):
        for kind in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception=kind.__name__):
                self.prepare()
                self.query()
                failure = kind("Synthetic preparation interruption")
                upload = Mock(getvalue=Mock(side_effect=failure))
                self.ui.values["comparador_A_image"] = upload
                with self.assertRaises(kind) as raised:
                    self.render()
                self.assertIs(raised.exception, failure)
                self.assert_collection_retired()
                self.ui.values.pop("comparador_A_image")
                self.render()
                self.assert_collection_retired()

    def test_failed_collection_retry_retires_both_results_before_embeddings(self):
        for kind in (GPUClientError, OSError, ValueError, RuntimeError, KeyboardInterrupt):
            with self.subTest(exception=kind.__name__):
                self.prepare()
                self.query()
                self.ui.session_state.pop(CACHE)
                self.fail_at = 1
                self.failure = kind("Synthetic collection retry failure")
                if kind in (RuntimeError, KeyboardInterrupt):
                    with self.assertRaises(kind):
                        self.render("comparador_calculate")
                else:
                    self.render("comparador_calculate")
                    self.assertEqual(self.ui.errors, [str(self.failure)])
                self.assert_collection_retired()
                self.render()
                self.assert_collection_retired()
                self.client.embed.assert_not_called()

    def test_late_embedding_failure_never_publishes_partial_collection(self):
        self.query()
        self.ui.session_state.pop(CACHE)
        self.ui.values["comparador_A_image"] = Upload(picture())
        self.fail_at = 2
        self.failure = GPUClientError("Synthetic image batch failure")
        self.render("comparador_calculate")
        self.assert_collection_retired()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.ui.session_state[CACHE]), 2)

    def test_cached_collection_pca_failure_retires_results_but_preserves_cache(self):
        self.query()
        cache = self.ui.session_state[CACHE]
        failure = RuntimeError("Synthetic PCA failure")
        with patch.dict(self.lab, {"_pca_3d": Mock(side_effect=failure)}):
            with self.assertRaises(RuntimeError) as raised:
                self.render("comparador_calculate")
        self.assertIs(raised.exception, failure)
        self.assert_collection_retired()
        self.assertIs(self.ui.session_state[CACHE], cache)
        self.client.embed.assert_not_called()

    def test_collection_spinner_failure_never_publishes_partial_result(self):
        for phase in ("enter", "exit"):
            for kind in (GPUClientError, KeyboardInterrupt):
                with self.subTest(phase=phase, exception=kind.__name__):
                    self.prepare()
                    self.query()
                    self.ui.spinner_phase = phase
                    self.ui.spinner_error = kind("Synthetic spinner failure")
                    if kind is KeyboardInterrupt:
                        with self.assertRaises(kind):
                            self.render("comparador_calculate")
                    else:
                        self.render("comparador_calculate")
                    self.assert_collection_retired()
                    self.ui.spinner_phase = None
                    self.render()
                    self.assert_collection_retired()

    def test_query_change_and_return_never_resurrects_neighbors(self):
        for key, value in (("comparador_query_text", "changed query"),
                           ("comparador_query_image", Upload(picture()))):
            with self.subTest(input=key):
                self.prepare()
                self.query()
                original = self.ui.values.get(key)
                self.ui.values[key] = value
                self.render()
                self.assert_query_retired()
                self.client.embed.assert_not_called()
                self.ui.values[key] = original
                self.render()
                self.assert_query_retired()

    def test_query_source_switch_and_return_retires_saved_query(self):
        self.query()
        self.ui.values["comparador_query_source"] = "Elemento de la colección"
        self.render()
        self.assertNotIn(QUERY, self.ui.session_state)
        self.ui.values["comparador_query_source"] = "Nueva entrada"
        self.render()
        self.assert_query_retired()

    def test_invalid_query_image_or_empty_retry_retires_only_query(self):
        cases = (("comparador_query_image", Upload(b"invalid"), False),
                 ("comparador_query_image", Upload(b"invalid"), True),
                 ("comparador_query_text", " \n ", True))
        for key, value, clicked in cases:
            with self.subTest(input=key, clicked=clicked):
                self.prepare()
                self.query()
                collection = self.ui.session_state[RESULT]
                original = self.ui.values.get(key)
                self.ui.values[key] = value
                self.render(*(["comparador_query_calculate"] if clicked else []))
                self.assert_query_retired()
                self.assertIs(self.ui.session_state[RESULT], collection)
                self.assertEqual(len(self.ui.errors), 1)
                self.client.embed.assert_not_called()
                self.ui.values[key] = original
                self.render()
                self.assert_query_retired()

    def test_query_preparation_exception_or_interruption_retires_query(self):
        for kind in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception=kind.__name__):
                self.prepare()
                self.query()
                failure = kind("Synthetic query preparation interruption")
                self.ui.values["comparador_query_image"] = Mock(getvalue=Mock(side_effect=failure))
                with self.assertRaises(kind):
                    self.render()
                self.assert_query_retired()
                self.ui.values.pop("comparador_query_image")
                self.render()
                self.assert_query_retired()

    def test_failed_query_retry_retires_neighbors_and_preserves_collection(self):
        for kind in (GPUClientError, OSError, ValueError, RuntimeError, KeyboardInterrupt):
            with self.subTest(exception=kind.__name__):
                self.prepare()
                self.query()
                self.ui.session_state.pop(CACHE)
                self.fail_at = 1
                self.failure = kind("Synthetic query retry failure")
                if kind in (RuntimeError, KeyboardInterrupt):
                    with self.assertRaises(kind):
                        self.render("comparador_query_calculate")
                else:
                    self.render("comparador_query_calculate")
                    self.assertEqual(self.ui.errors, [str(self.failure)])
                self.assert_query_retired()
                self.render()
                self.assert_query_retired()
                self.client.embed.assert_not_called()

    def test_query_payload_interruption_retires_before_validation(self):
        self.query()
        failure = KeyboardInterrupt("Synthetic payload interruption")
        with patch.object(self.lab["CompareItem"], "payload", side_effect=failure):
            with self.assertRaises(KeyboardInterrupt):
                self.render("comparador_query_calculate")
        self.assert_query_retired()
        self.client.embed.assert_not_called()

    def test_query_spinner_failure_does_not_publish_or_render_partial_vector(self):
        for phase in ("enter", "exit"):
            for kind in (GPUClientError, KeyboardInterrupt):
                with self.subTest(phase=phase, exception=kind.__name__):
                    self.prepare()
                    self.query()
                    self.ui.spinner_phase = phase
                    self.ui.spinner_error = kind("Synthetic query spinner failure")
                    if kind is KeyboardInterrupt:
                        with self.assertRaises(kind):
                            self.render("comparador_query_calculate")
                    else:
                        self.render("comparador_query_calculate")
                    self.assert_query_retired()
                    self.ui.spinner_phase = None
                    self.render()
                    self.assert_query_retired()

    def test_successful_collection_and_query_recovery_reuses_cached_embeddings(self):
        old_collection = self.collection()
        old_query = self.query()
        cache = self.ui.session_state[CACHE]
        with patch.dict(self.lab, {"_pca_3d": Mock(side_effect=ValueError("Synthetic failed PCA"))}):
            self.render("comparador_calculate")
        self.assert_collection_retired()
        collection = self.collection()
        self.assertIsNot(collection, old_collection)
        self.assertEqual(collection["signature"], old_collection["signature"])
        self.client.embed.assert_not_called()
        self.render()
        self.assert_query_retired()
        query = self.query()
        self.assertIsNot(query, old_query)
        self.assertEqual(query["signature"], old_query["signature"])
        self.assertIs(self.ui.session_state[CACHE], cache)
        self.client.embed.assert_not_called()
        self.ui.spinner_phase = "exit"
        self.ui.spinner_error = GPUClientError("Synthetic failed cached query retry")
        self.render("comparador_query_calculate")
        self.assert_query_retired()
        self.ui.spinner_phase = None
        recovered_query = self.query()
        self.assertIsNot(recovered_query, query)
        self.assertEqual(recovered_query["signature"], query["signature"])
        self.assertIs(self.ui.session_state[CACHE], cache)
        self.client.embed.assert_not_called()

    def test_explicit_same_input_retry_preserves_cache_and_replaces_results(self):
        collection = self.collection()
        query = self.query()
        cache = self.ui.session_state[CACHE]
        replacement = self.collection()
        self.assertIsNot(replacement, collection)
        self.assertNotIn(QUERY, self.ui.session_state)
        self.client.embed.assert_not_called()
        replacement_query = self.query()
        self.assertIsNot(replacement_query, query)
        self.assertIs(self.ui.session_state[CACHE], cache)
        self.client.embed.assert_not_called()

    def test_clear_cache_discards_results_without_restoring_them(self):
        self.query()
        self.render("comparador_clear_cache")
        self.assert_collection_retired()
        self.assertNotIn(CACHE, self.ui.session_state)
        self.render()
        self.assert_collection_retired()

    def test_comparator_invalidation_preserves_other_experiment_results(self):
        self.query()
        unrelated = {name: {"sentinel": name} for name in
                     ("matrix_result", "search_result", "enterprise_prediction", "tomografia_scans")}
        self.ui.session_state.update(unrelated)
        self.ui.values["comparador_A_text"] = ""
        self.render()
        self.assert_collection_retired()
        for name, saved in unrelated.items():
            self.assertIs(self.ui.session_state[name], saved)
