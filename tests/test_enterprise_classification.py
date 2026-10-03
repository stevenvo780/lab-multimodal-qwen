"""Offline regression tests for the uploaded-document prediction lifecycle.

Streamlit and unrelated chart dependencies are replaced before the application
is imported. NumPy/Pillow exercise the actual image preparation and classifier;
the GPU client is always a fake and socket/DNS access is forbidden.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image, PngImagePlugin


def load_case_module():
    streamlit = types.ModuleType("streamlit")
    streamlit.cache_data = lambda **kwargs: lambda function: function
    graph_objects = types.ModuleType("plotly.graph_objects")
    plotly = types.ModuleType("plotly")
    plotly.graph_objects = graph_objects
    decomposition = types.ModuleType("sklearn.decomposition")
    decomposition.PCA = mock.Mock()
    sklearn = types.ModuleType("sklearn")
    sklearn.decomposition = decomposition
    source = importlib.util.find_spec("caso_empresa").origin
    spec = importlib.util.spec_from_file_location("_enterprise_classification_test_case", source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        "streamlit": streamlit, "plotly": plotly,
        "plotly.graph_objects": graph_objects, "sklearn": sklearn,
        "sklearn.decomposition": decomposition,
    }):
        spec.loader.exec_module(module)
    return module


class Upload(io.BytesIO):
    def __init__(self, content: bytes, name: str = "document.png"):
        super().__init__(content)
        self.name = name


def image_data(color="red", image_format="PNG", *, note=None):
    output = io.BytesIO()
    options = {}
    if note is not None:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Comment", note)
        options["pnginfo"] = metadata
    Image.new("RGB", (32, 24), color=color).save(output, image_format, **options)
    return output.getvalue()


class EnterpriseClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_case_module()

    def setUp(self):
        for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo"):
            guard = mock.patch(target, side_effect=AssertionError("Network forbidden in classification tests"))
            guard.start()
            self.addCleanup(guard.stop)
        self.state = {}
        self.ui = mock.MagicMock()
        self.ui.session_state = self.state
        self.ui.selectbox.return_value = 0
        self.ui.checkbox.return_value = True
        self.ui.text_input.return_value = "a receipt"
        self.ui.columns.side_effect = lambda columns, **kwargs: [mock.MagicMock() for _ in columns]
        self.ui.tabs.return_value = [mock.MagicMock() for _ in range(4)]
        self.clicked = False
        self.ui.button.side_effect = lambda *args, **kwargs: (
            kwargs.get("key") == "enterprise_predict" and self.clicked
        )
        self.documents = [
            {"id": str(index), "label": label, "image": "fixture.jpg", "text": "Reference",
             "source": "Fixture", "source_url": "https://example.invalid", "license": "Fixture"}
            for index, label in enumerate(("receipt", "receipt", "chart", "chart"))
        ]
        self.images = np.zeros((4, 2048), dtype=np.float32)
        self.images[:2, 0] = 1
        self.images[2:, 1] = 1
        self.client = mock.Mock(base_url="http://127.0.0.1:18765", embed_timeout=180, batch_size=4)
        self.client.embed.return_value = [[1.0] + [0.0] * 2047]
        self.red = image_data("red")
        self.blue = image_data("blue")
        for target, value in {
            "st": self.ui,
            "MANIFEST": mock.Mock(),
            "EMBEDDINGS": mock.Mock(),
            "read_case": mock.Mock(side_effect=lambda *args: (self.documents, self.images, self.images.copy())),
            "image_bytes": mock.Mock(return_value=b"sample-preview"),
            "neighbor_tree": mock.Mock(return_value=[]),
            "tree_figure": mock.Mock(return_value=object()),
            "map_figure": mock.Mock(return_value=(object(), 0.5)),
        }.items():
            patcher = mock.patch.object(self.case, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def render(self, content, *, name="document.png", click=False):
        self.ui.reset_mock()
        self.clicked = click
        self.ui.file_uploader.return_value = None if content is None else Upload(content, name)
        self.case.render_enterprise_case(self.client)

    def assert_no_prediction(self):
        self.assertNotIn("enterprise_prediction", self.state)
        self.ui.success.assert_not_called()
        self.assertFalse(any(call.kwargs.get("width") == 280 for call in self.ui.image.call_args_list))

    def test_success_sends_and_shows_the_prepared_image(self):
        self.render(self.red, click=True)
        picture = self.case.web_image(self.red)
        self.client.embed.assert_called_once_with([{"image_base64": self.case.encode_image(picture)}])
        self.assertEqual(self.state["enterprise_prediction"][1], picture)
        self.ui.image.assert_any_call(picture, width=280)
        self.ui.success.assert_called_once()

    def test_same_filename_with_changed_content_invalidates_without_click(self):
        self.render(self.red, click=True)
        self.render(self.blue)
        self.assert_no_prediction()
        self.assertEqual(self.client.embed.call_count, 1)

    def test_same_content_survives_an_unrelated_rerun(self):
        self.render(self.red, click=True)
        original = self.state["enterprise_prediction"]
        self.render(self.red)
        self.assertIs(self.state["enterprise_prediction"], original)
        self.ui.success.assert_called_once()
        self.assertEqual(self.client.embed.call_count, 1)

    def test_identical_content_with_a_new_filename_keeps_prediction(self):
        self.render(self.red, click=True)
        original = self.state["enterprise_prediction"]
        self.render(self.red, name="renamed.png")
        self.assertIs(self.state["enterprise_prediction"], original)
        self.ui.success.assert_called_once()

    def test_equivalent_prepared_image_ignores_discarded_metadata(self):
        first = image_data(note="first")
        second = image_data(note="second")
        self.assertNotEqual(first, second)
        self.assertEqual(self.case.web_image(first), self.case.web_image(second))
        self.render(first, click=True)
        original = self.state["enterprise_prediction"]
        self.render(second)
        self.assertIs(self.state["enterprise_prediction"], original)
        self.ui.success.assert_called_once()

    def test_removing_upload_clears_prediction_and_cannot_resurrect_it(self):
        self.render(self.red, click=True)
        self.render(None)
        self.assert_no_prediction()
        self.render(self.red)
        self.assert_no_prediction()
        self.assertEqual(self.client.embed.call_count, 1)

    def test_changed_content_cannot_resurrect_result_when_previous_image_returns(self):
        self.render(self.red, click=True)
        self.render(self.blue)
        self.render(self.red)
        self.assert_no_prediction()

    def test_invalid_image_clears_previous_prediction(self):
        self.render(self.red, click=True)
        self.render(b"invalid image")
        self.assert_no_prediction()
        self.ui.error.assert_called_once()
        self.assertEqual(self.client.embed.call_count, 1)

    def test_pillow_safety_error_clears_prediction_without_resurrection(self):
        self.render(self.red, click=True)
        error = Image.DecompressionBombError("Image exceeds Pillow pixel safety limit")
        with mock.patch.object(self.case.Image, "open", side_effect=error):
            self.render(self.blue)
        self.assert_no_prediction()
        self.ui.error.assert_called_once_with(str(error))
        self.render(self.red)
        self.assert_no_prediction()
        self.assertEqual(self.client.embed.call_count, 1)

    def test_interrupted_preparation_propagates_and_clears_prediction(self):
        self.render(self.red, click=True)
        with mock.patch.object(self.case.Image, "open", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.render(self.blue)
        self.assert_no_prediction()
        self.ui.error.assert_not_called()
        self.render(self.red)
        self.assert_no_prediction()
        self.assertEqual(self.client.embed.call_count, 1)

    def test_unexpected_preparation_error_propagates_and_clears_prediction(self):
        self.render(self.red, click=True)
        error = RuntimeError("Unexpected preparation failure")
        with mock.patch.object(self.case.Image, "open", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "Unexpected preparation failure"):
                self.render(self.blue)
        self.assert_no_prediction()
        self.ui.error.assert_not_called()
        self.render(self.red)
        self.assert_no_prediction()
        self.assertEqual(self.client.embed.call_count, 1)

    def test_failed_retry_clears_before_request_and_stays_empty(self):
        self.render(self.red, click=True)

        def fail(*args, **kwargs):
            self.assertNotIn("enterprise_prediction", self.state)
            raise self.case.GPUClientError("GPU unavailable")

        self.client.embed.side_effect = fail
        self.render(self.red, click=True)
        self.assert_no_prediction()
        self.ui.error.assert_called_once_with("GPU unavailable")
        self.render(self.red)
        self.assert_no_prediction()
        self.assertEqual(self.client.embed.call_count, 2)

    def test_new_content_success_replaces_the_image_and_result_together(self):
        self.render(self.red, click=True)
        first = self.state["enterprise_prediction"]
        self.client.embed.return_value = [[0.0, 1.0] + [0.0] * 2046]
        self.render(self.blue, click=True)
        second = self.state["enterprise_prediction"]
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(second[1], self.case.web_image(self.blue))
        self.assertEqual(first[2][0][0], "receipt")
        self.assertEqual(second[2][0][0], "chart")
        self.ui.success.assert_called_once()
        self.assertEqual(self.client.embed.call_count, 2)

    def test_failed_classification_does_not_publish_embedding_or_old_prediction(self):
        self.render(self.red, click=True)
        with mock.patch.object(self.case, "classify", side_effect=ValueError("Classification failed")):
            self.render(self.red, click=True)
        self.assert_no_prediction()
        self.ui.error.assert_called_once_with("Classification failed")

    def test_invalid_embedding_does_not_publish_a_prediction(self):
        self.render(self.red, click=True)
        self.client.embed.return_value = [[float("nan")] + [0.0] * 2047]
        self.render(self.red, click=True)
        self.assert_no_prediction()
        self.ui.error.assert_called_once()

    def test_interrupted_retry_leaves_no_active_prediction(self):
        self.render(self.red, click=True)
        self.client.embed.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.render(self.red, click=True)
        self.assert_no_prediction()

    def test_success_is_published_only_after_classification_finishes(self):
        self.render(self.red, click=True)
        classify = self.case.classify

        def classify_without_published_state(*args):
            self.assertNotIn("enterprise_prediction", self.state)
            return classify(*args)

        with mock.patch.object(self.case, "classify", side_effect=classify_without_published_state):
            self.render(self.red, click=True)
        self.assertIn("enterprise_prediction", self.state)
        self.ui.success.assert_called_once()

    def test_endpoint_change_invalidates_prediction(self):
        self.render(self.red, click=True)
        self.client.base_url = "http://127.0.0.1:28765"
        self.render(self.red)
        self.assert_no_prediction()

    def test_reference_vector_change_invalidates_prediction(self):
        self.render(self.red, click=True)
        self.images[0, 0] = 0.8
        self.images[0, 2] = 0.6
        self.render(self.red)
        self.assert_no_prediction()

    def test_reference_label_change_invalidates_prediction(self):
        self.render(self.red, click=True)
        for doc in self.documents[:2]:
            doc["label"] = "bank_statement"
        self.render(self.red)
        self.assert_no_prediction()

    def test_timeout_and_batch_size_do_not_change_single_image_semantics(self):
        self.render(self.red, click=True)
        original = self.state["enterprise_prediction"]
        self.client.embed_timeout = 300
        self.client.batch_size = 2
        self.render(self.red)
        self.assertIs(self.state["enterprise_prediction"], original)
        self.ui.success.assert_called_once()

    def test_legacy_filename_key_is_discarded(self):
        self.state["enterprise_prediction"] = ("document.png", b"stale", [("receipt", 1.0)])
        self.render(self.red)
        self.assert_no_prediction()

    def test_supported_upload_formats_still_classify(self):
        for extension, image_format in (("jpg", "JPEG"), ("jpeg", "JPEG"), ("png", "PNG"), ("webp", "WEBP")):
            with self.subTest(extension=extension):
                content = image_data(image_format=image_format)
                self.render(content, name=f"document.{extension}", click=True)
                self.assertEqual(self.state["enterprise_prediction"][1], self.case.web_image(content))
                self.ui.success.assert_called_once()
                self.ui.error.assert_not_called()
        self.assertEqual(self.client.embed.call_count, 4)

    def test_exif_orientation_is_applied_to_identity_and_display(self):
        image = Image.new("RGB", (12, 24), "red")
        exif = image.getexif()
        exif[274] = 6
        output = io.BytesIO()
        image.save(output, "JPEG", exif=exif)
        self.render(output.getvalue(), click=True)
        prepared = self.state["enterprise_prediction"][1]
        with Image.open(io.BytesIO(prepared)) as result:
            self.assertEqual(result.size, (24, 12))
        self.render(output.getvalue())
        self.ui.success.assert_called_once()

    def test_existing_image_size_limit_clears_previous_prediction(self):
        self.render(self.red, click=True)
        self.render(b"x" * (16 * 1024 * 1024 + 1))
        self.assert_no_prediction()
        self.ui.error.assert_called_once_with("La imagen supera 16 MiB.")
        self.assertEqual(self.client.embed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
