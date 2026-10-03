"""Offline rerun tests of the actual layer-scan renderer, without Streamlit.

The renderer and image preparation are AST-isolated from the app's optional UI
and training imports. Pillow and GPUClient run normally; only Streamlit/chart
output and the HTTP transport are doubled. This is not a browser or GPU test.
"""

import __future__
import ast
import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import call, patch

import requests
from PIL import Image, PngImagePlugin

from gpu_client import GPUClient, GPUClientError, encode_image


API_URL = "https://gpu.invalid"
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def png(color="red", *, comment=None):
    output = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    if comment is not None:
        metadata.add_text("Comment", comment)
    Image.new("RGB", (24, 16), color).save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


def response(*, marker=1.0, model="test-model"):
    payload = {
        "dimension": 2048,
        "input_token_count": 7,
        "layers": [
            {"layer": index, "rms": marker, "mean": -0.25, "std": 0.5,
             "delta_cos": None if index == 1 else 0.9}
            for index in range(1, 29)
        ],
    }
    if model is not None:
        payload["model"] = model
    result = requests.Response()
    result.status_code = 200
    result.url = API_URL + "/inspect"
    result.encoding = "utf-8"
    result._content = json.dumps(payload).encode("utf-8")
    return result


class UploadedImage(io.BytesIO):
    def __init__(self, data, name="same-name.png"):
        super().__init__(data)
        self.name = name


class StreamlitDouble:
    def __init__(self):
        self.session_state = {}
        self.phrase = "un perro"
        self.selected = "perro.jpg"
        self.upload = UploadedImage(png())
        self.clicked = False
        self.charts = []
        self.metrics = []
        self.errors = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def reset_output(self):
        self.charts.clear()
        self.metrics.clear()
        self.errors.clear()

    def divider(self):
        pass

    def header(self, *args):
        pass

    def write(self, *args):
        pass

    def caption(self, *args):
        pass

    def columns(self, count):
        return [self] * count

    def text_input(self, *args, **kwargs):
        return self.phrase

    def selectbox(self, *args, **kwargs):
        return self.selected

    def file_uploader(self, *args, **kwargs):
        return self.upload

    def button(self, *args, **kwargs):
        return self.clicked

    def spinner(self, *args):
        return nullcontext()

    def metric(self, *args):
        self.metrics.append(args)

    def plotly_chart(self, figure, **kwargs):
        self.charts.append(figure)

    def error(self, message):
        self.errors.append(message)


def load_scan_functions(ui, root):
    path = SOURCE_ROOT / "tomografia_lab.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name in {"prepare_scan_image", "render_layer_scan"}]
    if len(selected) != 2:
        raise AssertionError("Expected the real image preparation and scan renderer")
    namespace = {
        "hashlib": hashlib, "io": io, "Image": Image, "GPUClientError": GPUClientError,
        "encode_image": encode_image, "st": ui, "ROOT": root,
        "scan_chart": lambda scans, field, *_: (copy.deepcopy(scans), field),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace["render_layer_scan"], namespace["prepare_scan_image"]


class LayerScanRerunTests(unittest.TestCase):
    def setUp(self):
        for target in (
            "socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection",
            "socket.getaddrinfo", "requests.sessions.Session.request",
        ):
            self.enterContext(patch(target, side_effect=AssertionError("Network is forbidden in unit tests")))
        self.request = self.enterContext(patch(
            "gpu_client.requests.request", autospec=True,
            side_effect=AssertionError("Unexpected HTTP request: configure a response"),
        ))
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(
            prefix=".scan-test-", dir=SOURCE_ROOT)))
        (self.root / "samples").mkdir()
        self.ui = StreamlitDouble()
        self.client = GPUClient(API_URL, embed_timeout=9.5)
        self.render, self.prepare = load_scan_functions(self.ui, self.root)

    def run_scan(self, *, clicked=False, replies=None):
        self.ui.reset_output()
        self.ui.clicked = clicked
        self.request.reset_mock()
        self.request.side_effect = replies if replies is not None else AssertionError("Unexpected HTTP request")
        self.render(self.client)

    def succeed(self, *, marker=1.0):
        self.run_scan(clicked=True, replies=[response(marker=marker), response(marker=marker)])
        self.assertEqual(self.ui.errors, [])
        self.assertEqual(len(self.ui.charts), 2)
        return copy.deepcopy(self.ui.session_state["tomografia_scans"])

    def assert_empty(self):
        self.assertNotIn("tomografia_scans", self.ui.session_state)
        self.assertEqual(self.ui.charts, [])
        self.assertEqual(self.ui.metrics, [])

    def test_success_records_normalized_request_effective_bytes_and_actual_model(self):
        self.ui.phrase = " \t un perro \n"
        saved = self.succeed()
        compact = self.prepare(self.ui.upload.getvalue())
        self.assertEqual(saved["request"], ("un perro", hashlib.sha256(compact).hexdigest(), API_URL))
        self.assertEqual(saved["models"], {"Texto": "test-model", "Imagen": "test-model"})
        self.assertEqual(set(saved["scans"]), {"Texto", "Imagen"})
        self.assertEqual(self.request.call_args_list, [
            call("POST", API_URL + "/inspect", json={"item": {"text": "un perro"}}, timeout=9.5),
            call("POST", API_URL + "/inspect", json={"item": {"image_base64": encode_image(compact)}}, timeout=9.5),
        ])

    def test_same_request_rerenders_without_network(self):
        saved = self.succeed()
        self.run_scan()
        self.assertEqual(self.ui.session_state["tomografia_scans"], saved)
        self.assertEqual(len(self.ui.charts), 2)
        self.request.assert_not_called()

    def test_changed_prompt_hides_old_curves_without_click(self):
        self.succeed()
        self.ui.phrase = "un gato"
        self.run_scan()
        self.assert_empty()
        self.request.assert_not_called()
        self.ui.phrase = "un perro"
        self.run_scan()
        self.assert_empty()  # Returning to old inputs must not resurrect a discarded result.

    def test_whitespace_only_prompt_change_keeps_same_effective_request(self):
        saved = self.succeed()
        self.ui.phrase = "\n un perro \t"
        self.run_scan()
        self.assertEqual(self.ui.session_state["tomografia_scans"], saved)
        self.assertEqual(len(self.ui.charts), 2)
        self.request.assert_not_called()

    def test_same_filename_changed_bytes_hides_old_curves(self):
        self.succeed()
        old_name = self.ui.upload.name
        self.ui.upload = UploadedImage(png("blue"), name=old_name)
        self.run_scan()
        self.assert_empty()
        self.request.assert_not_called()

    def test_filename_and_metadata_do_not_replace_effective_image_identity(self):
        self.ui.upload = UploadedImage(png(comment="before"), name="first.png")
        before = self.ui.upload.getvalue()
        saved = self.succeed()
        self.ui.upload = UploadedImage(png(comment="after"), name="second.png")
        self.assertNotEqual(before, self.ui.upload.getvalue())
        self.assertEqual(self.prepare(before), self.prepare(self.ui.upload.getvalue()))
        self.run_scan()
        self.assertEqual(self.ui.session_state["tomografia_scans"], saved)
        self.assertEqual(len(self.ui.charts), 2)
        self.request.assert_not_called()

    def test_sample_file_changed_in_place_invalidates_result(self):
        sample = self.root / "samples" / self.ui.selected
        sample.write_bytes(png("red"))
        self.ui.upload = None
        self.succeed()
        sample.write_bytes(png("blue"))
        self.run_scan()
        self.assert_empty()
        self.request.assert_not_called()

    def test_endpoint_change_hides_old_curves(self):
        self.succeed()
        self.client = GPUClient("https://other-gpu.invalid")
        self.run_scan()
        self.assert_empty()
        self.request.assert_not_called()

    def test_timeout_retry_clears_previous_result_before_first_request(self):
        self.succeed()

        def timeout(*args, **kwargs):
            self.assertNotIn("tomografia_scans", self.ui.session_state)
            raise requests.Timeout("offline timeout")

        self.run_scan(clicked=True, replies=timeout)
        self.assert_empty()
        self.assertEqual(self.request.call_count, 1)
        self.assertIn("tardó demasiado", self.ui.errors[0])
        self.run_scan()
        self.assert_empty()
        self.request.assert_not_called()

    def test_second_modality_failure_never_publishes_partial_results(self):
        self.succeed()
        calls = 0

        def partial_failure(*args, **kwargs):
            nonlocal calls
            calls += 1
            self.assertNotIn("tomografia_scans", self.ui.session_state)
            if calls == 2:
                raise requests.Timeout("offline image timeout")
            return response(marker=2.0)

        self.run_scan(clicked=True, replies=partial_failure)
        self.assert_empty()
        self.assertEqual(calls, 2)
        self.assertIn("tardó demasiado", self.ui.errors[0])

    def test_successful_retry_replaces_failure_with_complete_fresh_pair(self):
        self.succeed()
        self.run_scan(clicked=True, replies=[response(marker=2.0), requests.Timeout("offline")])
        self.assert_empty()
        saved = self.succeed(marker=3.0)
        self.assertEqual(set(saved["scans"]), {"Texto", "Imagen"})
        self.assertTrue(all(result["layers"][0]["rms"] == 3.0 for result in saved["scans"].values()))

    def test_empty_text_preserves_image_only_support(self):
        self.succeed()
        self.ui.phrase = " \n\t"
        self.run_scan(clicked=True, replies=[response()])
        self.assertEqual(self.ui.errors, [])
        self.assertEqual(self.request.call_count, 1)
        self.assertEqual(set(self.ui.session_state["tomografia_scans"]["scans"]), {"Imagen"})
        self.assertEqual(len(self.ui.charts), 2)
        self.assertEqual(len(self.ui.metrics), 1)
        self.assertEqual(set(self.request.call_args.kwargs["json"]["item"]), {"image_base64"})

    def test_invalid_image_hides_saved_result_with_or_without_retry(self):
        for clicked in (False, True):
            with self.subTest(clicked=clicked):
                self.ui.upload = UploadedImage(png())
                self.succeed()
                self.ui.upload = UploadedImage(b"not an image")
                self.run_scan(clicked=clicked)
                self.assert_empty()
                self.assertIn("no se pudo abrir", self.ui.errors[0])
                self.request.assert_not_called()

    def test_pillow_safety_rejection_clears_state_before_preparation(self):
        for clicked in (False, True):
            with self.subTest(clicked=clicked):
                self.succeed()
                original_upload = self.ui.upload
                self.ui.upload = UploadedImage(png("blue"))
                active_during_preparation = []

                def reject_image(*args, **kwargs):
                    active_during_preparation.append("tomografia_scans" in self.ui.session_state)
                    raise Image.DecompressionBombError("synthetic Pillow safety rejection")

                with patch.object(Image, "open", side_effect=reject_image):
                    self.run_scan(clicked=clicked)
                self.assert_empty()
                self.assertEqual(active_during_preparation, [False])
                self.assertEqual(self.ui.errors, ["synthetic Pillow safety rejection"])
                self.request.assert_not_called()
                self.ui.upload = original_upload
                self.run_scan()
                self.assert_empty()  # Valid bytes returning cannot resurrect an invalidated scan.
                self.request.assert_not_called()
                self.succeed(marker=2.0)

    def test_unexpected_preparation_errors_and_interrupts_propagate_with_state_cleared(self):
        for failure_type in (RuntimeError, KeyboardInterrupt):
            for clicked in (False, True):
                with self.subTest(failure_type=failure_type.__name__, clicked=clicked):
                    self.succeed()
                    original_upload = self.ui.upload
                    self.ui.upload = UploadedImage(png("blue"))
                    active_during_preparation = []
                    failure = failure_type("synthetic interrupted image preparation")

                    def interrupt_image(*args, **kwargs):
                        active_during_preparation.append("tomografia_scans" in self.ui.session_state)
                        raise failure

                    with patch.object(Image, "open", side_effect=interrupt_image):
                        with self.assertRaises(failure_type) as raised:
                            self.run_scan(clicked=clicked)
                    self.assertIs(raised.exception, failure)
                    self.assert_empty()
                    self.assertEqual(active_during_preparation, [False])
                    self.assertEqual(self.ui.errors, [])
                    self.request.assert_not_called()
                    self.ui.upload = original_upload
                    self.run_scan()
                    self.assert_empty()
                    self.request.assert_not_called()
                    self.succeed(marker=2.0)

    def test_unreadable_upload_clears_state_before_reading_bytes(self):
        for clicked in (False, True):
            with self.subTest(clicked=clicked):
                self.succeed()
                active_during_read = []

                def reject_read():
                    active_during_read.append("tomografia_scans" in self.ui.session_state)
                    raise ValueError("synthetic closed upload")

                with patch.object(self.ui.upload, "getvalue", side_effect=reject_read):
                    self.run_scan(clicked=clicked)
                self.assert_empty()
                self.assertEqual(active_during_read, [False])
                self.assertEqual(self.ui.errors, ["synthetic closed upload"])
                self.request.assert_not_called()
                self.run_scan()
                self.assert_empty()
                self.request.assert_not_called()

    def test_missing_sample_hides_saved_result(self):
        self.succeed()
        self.ui.upload = None
        self.run_scan()
        self.assert_empty()
        self.assertEqual(len(self.ui.errors), 1)
        self.request.assert_not_called()

    def test_legacy_unidentified_result_is_not_rendered(self):
        self.ui.session_state["tomografia_scans"] = {"Texto": response().json()}
        self.run_scan()
        self.assert_empty()
        self.request.assert_not_called()

    def test_model_identity_is_not_fabricated_when_api_omits_it(self):
        self.run_scan(clicked=True, replies=[response(model=None), response(model=None)])
        self.assertEqual(self.ui.errors, [])
        self.assertEqual(self.ui.session_state["tomografia_scans"]["models"], {"Texto": None, "Imagen": None})
        self.assertEqual(len(self.ui.charts), 2)

    def test_model_name_from_one_response_is_not_assumed_for_the_other(self):
        self.run_scan(clicked=True, replies=[response(model="known"), response(model=None)])
        self.assertEqual(self.ui.errors, [])
        self.assertEqual(self.ui.session_state["tomografia_scans"]["models"], {"Texto": "known", "Imagen": None})
        self.assertEqual(len(self.ui.charts), 2)

    def test_model_change_between_modalities_rejects_entire_pair(self):
        self.succeed()
        self.run_scan(clicked=True, replies=[response(model="first"), response(model="second")])
        self.assert_empty()
        self.assertIn("modelo cambió", self.ui.errors[0])


if __name__ == "__main__":
    unittest.main()
