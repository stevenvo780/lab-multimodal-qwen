"""Exercise the real client and requests responses with mocked HTTP transport."""

import __future__
import ast
import base64
import copy
import json
import socket
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import requests

from gpu_client import DEFAULT_API_URL, EXPECTED_DIMENSION, GPUClient, GPUClientError, encode_image


API_URL = "https://gpu.invalid"


def response(payload=None, *, status=200, body=None):
    """Use requests' real JSON decoding and HTTP error behavior, without I/O."""
    result = requests.Response()
    result.status_code = status
    result.url = API_URL
    result.encoding = "utf-8"
    result._content = json.dumps(payload).encode("utf-8") if body is None else body
    return result


def health_payload(**changes):
    return {"status": "ok", "model_loaded": True, "model": "Qwen3-VL-Embedding-2B",
            "dimension": 2048, **changes}


def embed_payload(*values):
    return {"dimension": 2048, "vectors": [[value] * 2048 for value in values]}


def inspection_payload():
    return {
        "dimension": 2048,
        "layers": [
            {"layer": index, "rms": 1.0, "mean": -0.25, "std": 0.5,
             "delta_cos": None if index == 1 else 0.9}
            for index in range(1, 29)
        ],
    }


def load_scan_chart():
    """Execute the actual chart function body with only Plotly calls doubled.

    This checks the caller's data access without importing Streamlit or starting
    an app. It is not native Plotly rendering or an end-to-end UI test.
    """
    path = Path(__file__).resolve().parents[1] / "tomografia_lab.py"
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "scan_chart")
    isolated = ast.Module(body=[function], type_ignores=[])
    plotly = Mock(spec=["Figure", "Scatter"])
    namespace = {"go": plotly}
    exec(compile(isolated, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["scan_chart"], plotly


class OfflineClientTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {"QWEN_GPU_API_URL": API_URL}))
        # These guards fail even if production code stops using requests.request.
        for target in (
            "socket.socket.connect", "socket.socket.connect_ex",
            "socket.create_connection", "socket.getaddrinfo",
            "requests.sessions.Session.request",
        ):
            self.enterContext(patch(target, side_effect=AssertionError("Network is forbidden in unit tests")))
        self.request = self.enterContext(patch(
            "gpu_client.requests.request", autospec=True,
            side_effect=AssertionError("Unexpected HTTP request: configure a response"),
        ))
        self.client = GPUClient(API_URL, health_timeout=1.25, embed_timeout=9.5, batch_size=2)

    def reply(self, payload=None, **kwargs):
        self.request.side_effect = None
        self.request.return_value = response(payload, **kwargs)

    def operations(self):
        return (
            ("health", self.client.health),
            ("embed", lambda: self.client.embed([{"text": "test"}])),
            ("inspect", lambda: self.client.inspect({"text": "test"})),
        )

    def test_network_guard_blocks_unmocked_transport_and_dns(self):
        for operation in (
            lambda: requests.Session().request("GET", API_URL),
            lambda: socket.create_connection(("gpu.invalid", 443)),
            lambda: socket.getaddrinfo("gpu.invalid", 443),
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(AssertionError, "Network is forbidden"):
                operation()

    def test_dimension_contract_is_2048(self):
        self.assertEqual(EXPECTED_DIMENSION, 2048)

    def test_base_url_precedence_and_trailing_slashes(self):
        with patch.dict("os.environ", {"QWEN_GPU_API_URL": "https://env.invalid///"}, clear=True):
            self.assertEqual(GPUClient().base_url, "https://env.invalid")
            self.assertEqual(GPUClient(API_URL + "///").base_url, API_URL)
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(GPUClient().base_url, DEFAULT_API_URL)
        self.request.assert_not_called()

    def test_invalid_url_and_nonpositive_batch_size_fail_before_http(self):
        for kwargs in ({"base_url": "ftp://gpu.invalid"}, {"batch_size": 0}, {"batch_size": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GPUClient(**kwargs)
        self.request.assert_not_called()

    def test_encode_image_round_trip_and_empty_rejection(self):
        data = b"\x00\xffoffline image bytes"
        self.assertEqual(base64.b64decode(encode_image(data)), data)
        with self.assertRaisesRegex(GPUClientError, "vacía"):
            encode_image(b"")
        self.request.assert_not_called()

    def test_health_accepts_loaded_and_lazy_model_and_preserves_metadata(self):
        for loaded in (True, False):
            with self.subTest(loaded=loaded):
                payload = health_payload(model_loaded=loaded)
                self.reply(payload)
                self.assertEqual(self.client.health(), payload)
                self.request.assert_called_with("GET", API_URL + "/health", timeout=1.25)

    def test_health_rejects_missing_or_unhealthy_status(self):
        for status in (None, "loading", "error", True):
            with self.subTest(status=status):
                self.reply(health_payload(status=status))
                with self.assertRaisesRegex(GPUClientError, "estado 'ok'"):
                    self.client.health()
        payload = health_payload()
        del payload["status"]
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "estado 'ok'"):
            self.client.health()

    def test_health_requires_boolean_model_loaded(self):
        for loaded in (None, 0, 1, "true", [], {}):
            with self.subTest(loaded=loaded):
                self.reply(health_payload(model_loaded=loaded))
                with self.assertRaisesRegex(GPUClientError, "model_loaded"):
                    self.client.health()
        payload = health_payload()
        del payload["model_loaded"]
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "model_loaded"):
            self.client.health()

    def test_health_requires_expected_dimension(self):
        for dimension in (None, 0, 2047, 2049, "2048"):
            with self.subTest(dimension=dimension):
                self.reply(health_payload(dimension=dimension))
                with self.assertRaisesRegex(GPUClientError, "Dimensión inesperada"):
                    self.client.health()
        payload = health_payload()
        del payload["dimension"]
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "Dimensión inesperada"):
            self.client.health()

    def test_embed_batches_preserve_order_trim_payload_and_repeat_instruction(self):
        items = [
            {"text": "  uno \n", "ignored": "drop"},
            {"image_base64": "aW1hZ2U="},
            {"text": " tres ", "image_base64": "bWl4ZWQ=", "ignored": "drop"},
            {"text": "\t", "image_base64": "b3Rybw=="},
            {"text": " cinco ", "image_base64": ""},
        ]
        original = copy.deepcopy(items)
        self.request.side_effect = [response(embed_payload(1, 2)), response(embed_payload(3, 4)),
                                    response(embed_payload(5))]
        actual = self.client.embed(items, instruction=" \n Buscar documentos \t")
        self.assertEqual(actual, [[float(index)] * 2048 for index in range(1, 6)])
        self.assertTrue(all(type(value) is float for vector in actual for value in vector))
        self.assertEqual(items, original)
        expected_batches = [
            [{"text": "uno"}, {"image_base64": "aW1hZ2U="}],
            [{"text": "tres", "image_base64": "bWl4ZWQ="}, {"image_base64": "b3Rybw=="}],
            [{"text": "cinco"}],
        ]
        self.assertEqual(self.request.call_args_list, [
            call("POST", API_URL + "/embed", json={"items": batch, "instruction": "Buscar documentos"},
                 timeout=9.5) for batch in expected_batches
        ])

    def test_blank_or_absent_instruction_is_omitted(self):
        for instruction in (None, "", " \n\t"):
            with self.subTest(instruction=instruction):
                self.reply(embed_payload(0))
                self.client.embed([{"text": "test"}], instruction=instruction)
                self.request.assert_called_with("POST", API_URL + "/embed",
                                                json={"items": [{"text": "test"}]}, timeout=9.5)

    def test_invalid_instruction_fails_before_http(self):
        for instruction in (7, False, [], {}):
            with self.subTest(instruction=instruction), self.assertRaisesRegex(GPUClientError, "instrucción"):
                self.client.embed([{"text": "test"}], instruction=instruction)
        self.request.assert_not_called()

    def test_invalid_items_are_all_validated_before_any_batch(self):
        for item in (None, "text", [], {}, {"text": " \n"}, {"image_base64": ""},
                     {"text": 12}, {"image_base64": 12}, {"text": "ok", "image_base64": False}):
            with self.subTest(item=item), self.assertRaises(GPUClientError):
                self.client.embed([{"text": "first"}, {"text": "second"}, item])
        with self.assertRaisesRegex(GPUClientError, "al menos una"):
            self.client.embed([])
        self.request.assert_not_called()

    def test_embed_rejects_wrong_or_missing_dimension(self):
        for dimension in (None, 2047, 2049, "2048"):
            with self.subTest(dimension=dimension):
                payload = embed_payload(0)
                payload["dimension"] = dimension
                self.reply(payload)
                with self.assertRaisesRegex(GPUClientError, "dimensión"):
                    self.client.embed([{"text": "test"}])
        payload = embed_payload(0)
        del payload["dimension"]
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "dimensión"):
            self.client.embed([{"text": "test"}])

    def test_embed_rejects_missing_wrong_type_and_wrong_vector_count(self):
        for vectors in (None, {}, "bad", [], [[0] * 2048, [0] * 2048]):
            with self.subTest(vectors_type=type(vectors).__name__, count=len(vectors) if vectors is not None else None):
                self.reply({"dimension": 2048, "vectors": vectors})
                with self.assertRaisesRegex(GPUClientError, "número inesperado"):
                    self.client.embed([{"text": "test"}])
        self.reply({"dimension": 2048})
        with self.assertRaisesRegex(GPUClientError, "número inesperado"):
            self.client.embed([{"text": "test"}])

    def test_embed_rejects_invalid_vector_shape(self):
        for vector in (None, {}, "bad", [], [0] * 2047, [0] * 2049):
            with self.subTest(kind=type(vector).__name__, size=len(vector) if vector is not None else None):
                self.reply({"dimension": 2048, "vectors": [vector]})
                with self.assertRaisesRegex(GPUClientError, "vector 1.*dimensión inválida"):
                    self.client.embed([{"text": "test"}])

    def test_embed_converts_numeric_strings_and_integers(self):
        self.reply(embed_payload("0.25", 2))
        self.assertEqual(self.client.embed([{"text": "first"}, {"text": "second"}]),
                         [[0.25] * 2048, [2.0] * 2048])

    def test_embed_preserves_boolean_coercion_and_zero_vectors(self):
        self.reply(embed_payload(True, 0))
        self.assertEqual(self.client.embed([{"text": "first"}, {"text": "second"}]),
                         [[1.0] * 2048, [0.0] * 2048])

    def test_embed_rejects_nonnumeric_values_with_original_cause(self):
        for value in (None, {}, [], "bad"):
            with self.subTest(value=value):
                payload = embed_payload(0)
                payload["vectors"][0][-1] = value
                self.reply(payload)
                with self.assertRaisesRegex(GPUClientError, "vector 1.*no numéricos") as caught:
                    self.client.embed([{"text": "test"}])
                self.assertIsInstance(caught.exception.__cause__, (TypeError, ValueError))

    def test_embed_wraps_out_of_range_integer_conversion(self):
        payload = embed_payload(0)
        payload["vectors"][0][-1] = 10 ** 400
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "vector 1.*no numéricos") as caught:
            self.client.embed([{"text": "test"}])
        self.assertIsInstance(caught.exception.__cause__, OverflowError)

    def test_embed_rejects_nonfinite_numbers_and_numeric_strings(self):
        for value in (float("nan"), float("inf"), -float("inf"), "nan", "inf", "-inf", "1e400"):
            with self.subTest(value=value):
                payload = embed_payload(0)
                payload["vectors"][0][-1] = value
                self.reply(payload)
                with self.assertRaisesRegex(GPUClientError, "vector 1.*no finitos"):
                    self.client.embed([{"text": "test"}])

    def test_later_batch_failure_uses_global_position_and_stops(self):
        invalid = embed_payload(3, 4)
        invalid["vectors"][1] = []
        self.request.side_effect = [response(embed_payload(1, 2)), response(invalid)]
        with self.assertRaisesRegex(GPUClientError, "vector 4.*dimensión inválida"):
            self.client.embed([{"text": str(index)} for index in range(5)])
        self.assertEqual(self.request.call_count, 2)

    def test_later_batch_count_must_match_that_batch(self):
        self.request.side_effect = [response(embed_payload(1, 2)), response(embed_payload(3, 4))]
        with self.assertRaisesRegex(GPUClientError, "número inesperado"):
            self.client.embed([{"text": str(index)} for index in range(3)])
        self.assertEqual(self.request.call_count, 2)

    def test_transport_errors_are_wrapped_for_every_endpoint(self):
        for error_type, message in ((requests.Timeout, "tardó demasiado"),
                                    (requests.ConnectionError, "No hay conexión"),
                                    (requests.RequestException, "No se pudo leer")):
            for name, operation in self.operations():
                with self.subTest(error=error_type.__name__, endpoint=name):
                    error = error_type("offline transport failure")
                    self.request.side_effect = error
                    with self.assertRaisesRegex(GPUClientError, message) as caught:
                        operation()
                    self.assertIs(caught.exception.__cause__, error)

    def test_http_status_and_json_detail_are_reported_for_every_endpoint(self):
        for name, operation in self.operations():
            with self.subTest(endpoint=name):
                self.reply({"detail": "invalid input"}, status=422)
                with self.assertRaisesRegex(GPUClientError, "HTTP 422.*invalid input") as caught:
                    operation()
                self.assertIsInstance(caught.exception.__cause__, requests.HTTPError)

    def test_http_non_json_error_body_is_truncated(self):
        self.reply(status=503, body=b"x" * 300)
        with self.assertRaises(GPUClientError) as caught:
            self.client.health()
        self.assertEqual(str(caught.exception), "La API GPU devolvió HTTP 503. " + "x" * 250)

    def test_http_nonobject_or_missing_detail_still_reports_status(self):
        for payload in ([], "failure", None, {}):
            with self.subTest(payload=payload):
                self.reply(payload, status=500)
                with self.assertRaises(GPUClientError) as caught:
                    self.client.health()
                self.assertEqual(str(caught.exception), "La API GPU devolvió HTTP 500.")

    def test_malformed_json_is_wrapped_for_every_endpoint(self):
        for name, operation in self.operations():
            with self.subTest(endpoint=name):
                self.reply(body=b'{"unfinished":')
                with self.assertRaisesRegex(GPUClientError, "No se pudo leer") as caught:
                    operation()
                self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_nonobject_json_is_rejected_for_every_endpoint(self):
        for payload in ([], None, "text", 7, True):
            for name, operation in self.operations():
                with self.subTest(payload=payload, endpoint=name):
                    self.reply(payload)
                    with self.assertRaisesRegex(GPUClientError, "JSON inesperado"):
                        operation()

    def test_inspect_text_and_image_payloads_preserve_valid_layer_order(self):
        for item in ({"text": "test", "ignored": "drop"}, {"image_base64": "aW1hZ2U="}):
            with self.subTest(item=item):
                original = dict(item)
                payload = inspection_payload()
                self.reply(payload)
                self.assertEqual(self.client.inspect(item), payload)
                self.assertEqual(item, original)
                clean = {key: value for key, value in item.items() if key != "ignored"}
                self.request.assert_called_with("POST", API_URL + "/inspect",
                                                json={"item": clean}, timeout=9.5)

    def test_inspect_rejects_empty_mixed_and_nonmapping_input_before_http(self):
        for item in (None, [], "text", {}, {"text": ""}, {"text": "test", "image_base64": "aW1hZ2U="}):
            with self.subTest(item=item), self.assertRaises(GPUClientError):
                self.client.inspect(item)
        self.request.assert_not_called()

    def test_inspect_requires_expected_dimension(self):
        for dimension in (None, 2047, 2049, "2048"):
            with self.subTest(dimension=dimension):
                payload = inspection_payload()
                payload["dimension"] = dimension
                self.reply(payload)
                with self.assertRaisesRegex(GPUClientError, "dimensión inesperada"):
                    self.client.inspect({"text": "test"})
        payload = inspection_payload()
        del payload["dimension"]
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "dimensión inesperada"):
            self.client.inspect({"text": "test"})

    def test_inspect_requires_exactly_28_layers(self):
        valid = inspection_payload()["layers"]
        for layers in (None, {}, [], valid[:-1], valid + [valid[-1]]):
            with self.subTest(size=len(layers) if layers is not None else None):
                self.reply({"dimension": 2048, "layers": layers})
                with self.assertRaisesRegex(GPUClientError, "28 capas"):
                    self.client.inspect({"text": "test"})
        self.reply({"dimension": 2048})
        with self.assertRaisesRegex(GPUClientError, "28 capas"):
            self.client.inspect({"text": "test"})

    def test_inspect_rejects_reordered_duplicate_missing_and_malformed_layers(self):
        valid = inspection_payload()["layers"]
        swapped = copy.deepcopy(valid)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        for layers in (list(reversed(valid)), swapped, [valid[0]] * 28,
                       valid[:-1] + [{}], valid[:-1] + [None], valid[:-1] + [{"layer": "28"}]):
            with self.subTest(last=layers[-1]):
                self.reply({"dimension": 2048, "layers": layers})
                with self.assertRaisesRegex(GPUClientError, "capas desordenadas"):
                    self.client.inspect({"text": "test"})

    def test_inspect_requires_finite_numeric_layer_statistics(self):
        for field in ("rms", "mean", "std"):
            for value in (None, "1", [], float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    payload = inspection_payload()
                    payload["layers"][-1][field] = value
                    self.reply(payload)
                    with self.assertRaisesRegex(GPUClientError, f"capa 28.*{field}"):
                        self.client.inspect({"text": "test"})
            payload = inspection_payload()
            del payload["layers"][-1][field]
            self.reply(payload)
            with self.assertRaisesRegex(GPUClientError, f"capa 28.*{field}"):
                self.client.inspect({"text": "test"})

    def test_inspect_requires_consecutive_cosine_after_first_layer(self):
        for value in (None, "0.9", [], float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                payload = inspection_payload()
                payload["layers"][1]["delta_cos"] = value
                self.reply(payload)
                with self.assertRaisesRegex(GPUClientError, "capa 2.*coseno consecutivo"):
                    self.client.inspect({"text": "test"})
        payload = inspection_payload()
        del payload["layers"][1]["delta_cos"]
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "capa 2.*coseno consecutivo"):
            self.client.inspect({"text": "test"})

    def test_inspect_rejects_out_of_range_integer_statistics(self):
        for field in ("rms", "mean", "std", "delta_cos"):
            with self.subTest(field=field):
                payload = inspection_payload()
                payload["layers"][1][field] = 10 ** 400
                self.reply(payload)
                message = "coseno" if field == "delta_cos" else field
                with self.assertRaisesRegex(GPUClientError, f"capa 2.*{message}"):
                    self.client.inspect({"text": "test"})

    def test_inspect_rejects_missing_first_cosine_before_chart(self):
        payload = inspection_payload()
        del payload["layers"][0]["delta_cos"]
        scan_chart, _ = load_scan_chart()
        # The real caller body indexes this key even for the first layer.
        with self.assertRaisesRegex(KeyError, "delta_cos"):
            scan_chart({"Texto": payload}, "delta_cos", "Title", "Cosine")
        self.reply(payload)
        with self.assertRaisesRegex(GPUClientError, "capa 1.*coseno"):
            self.client.inspect({"text": "test"})

    def test_inspect_requires_null_first_cosine(self):
        # gpu_remote/embed.py emits None: layer one has no previous layer.
        for value in (0, 0.9, False, "0.9", [], float("nan"), float("inf")):
            with self.subTest(value=value):
                payload = inspection_payload()
                payload["layers"][0]["delta_cos"] = value
                self.reply(payload)
                with self.assertRaisesRegex(GPUClientError, "capa 1.*coseno"):
                    self.client.inspect({"text": "test"})

    def test_inspect_preserves_null_first_cosine_for_actual_chart_body(self):
        self.reply(inspection_payload())
        result = self.client.inspect({"text": "test"})
        scan_chart, plotly = load_scan_chart()
        scan_chart({"Texto": result}, "delta_cos", "Title", "Cosine")
        chart_data = plotly.Scatter.call_args.kwargs
        self.assertEqual(chart_data["x"], list(range(1, 29)))
        self.assertEqual(chart_data["y"], [None] + [0.9] * 27)
        self.assertIsNone(result["layers"][0]["delta_cos"])


if __name__ == "__main__":
    unittest.main()
