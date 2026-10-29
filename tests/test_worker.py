# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Worker contracts using tiny actual rasters and an injected model, never ML."""

import base64
import copy
import hashlib
import http.client
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from pvextent import worker

TOKEN = "worker-test-" + "a" * 32
ROOF = [[[[.125, .125], [.875, .125], [.875, .875], [.125, .875], [.125, .125]]]]


def image_request() -> dict:
    image = Image.new("RGB", (32, 32), (255, 128, 0))
    stream = BytesIO()
    image.save(stream, format="PNG")
    payload = stream.getvalue()
    image.close()
    return {"schemaVersion": 1, "image": {"mimeType": "image/png", "bytesBase64": base64.b64encode(payload).decode("ascii"),
            "sha256": hashlib.sha256(payload).hexdigest(), "width": 32, "height": 32}, "roofPixels": copy.deepcopy(ROOF)}


class TestWorkerPayload(unittest.TestCase):
    def test_import_does_not_load_torch_or_start_worker(self):
        result = subprocess.run([sys.executable, "-c", "import pvextent.worker,sys; assert 'torch' not in sys.modules; print('offline-import')"], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "offline-import")

    def test_decodes_exact_original_pixels_and_snapshot_identity(self):
        value = image_request()
        decoded = worker.decode_request(value)
        self.assertEqual(decoded.sha256, value["image"]["sha256"])
        self.assertEqual(decoded.rgb.size, (32, 32))
        self.assertEqual(decoded.rgb.getpixel((0, 0)), (255, 128, 0))
        self.assertEqual(decoded.roof, ROOF)
        decoded.rgb.close()

    def test_rejects_wrong_hash_dimensions_mime_base64_or_extra_fields(self):
        for key, value in [("sha256", "0" * 64), ("width", 33), ("width", 1025), ("height", True),
                           ("mimeType", "image/jpeg"), ("mimeType", "image/svg+xml"), ("bytesBase64", "not_base64")]:
            with self.subTest(key=key, value=value):
                request = image_request(); request["image"][key] = value
                with self.assertRaises(worker.WorkerError):
                    worker.decode_request(request)
        request = image_request(); request["sourceUrl"] = "https://example.invalid/never-fetch"
        with self.assertRaises(worker.WorkerError):
            worker.decode_request(request)

    def test_rejects_truncated_raster_even_with_matching_digest(self):
        value = image_request()
        raw = base64.b64decode(value["image"]["bytesBase64"])[:40]
        value["image"].update(bytesBase64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaises(worker.WorkerError):
            worker.decode_request(value)

    def test_roof_bounds_closure_area_and_vertex_budget(self):
        bad_roofs = [[], [[[[0, 0], [1, 0], [1, 1], [0, .1]]]],
                     [[[[0, 0], [1, 0], [0, 0], [0, 0]]]],
                     [[[[0, 0], [1.01, 0], [1, 1], [0, 0]]]],
                     [[[[0, 0], [float("nan"), 0], [1, 1], [0, 0]]]],
                     [[[[0, 0], [True, 0], [1, 1], [0, 0]]]],
                     [[[([0, 0] if i % 2 == 0 else [1, 1]) for i in range(20_001)]]]]
        for value in bad_roofs:
            with self.subTest(value=str(value)[:80]):
                with self.assertRaises(worker.WorkerError):
                    worker.validate_roof(value)

    def test_roof_mask_preserves_holes_and_disjoint_polygons(self):
        polygons = [[[[0, 0], [.6, 0], [.6, .6], [0, .6], [0, 0]],
                     [[.2, .2], [.4, .2], [.4, .4], [.2, .4], [.2, .2]]],
                    [[[.8, .8], [1, .8], [1, 1], [.8, 1], [.8, .8]]]]
        mask = worker.roof_mask((100, 100), worker.validate_roof(polygons))
        self.assertEqual(mask.getpixel((10, 10)), 255)
        self.assertEqual(mask.getpixel((30, 30)), 0)
        self.assertEqual(mask.getpixel((70, 70)), 0)
        self.assertEqual(mask.getpixel((90, 90)), 255)

    def test_components_keep_only_roof_pixels_and_filter_tiny_fragments(self):
        values = np.zeros((32, 32), dtype=np.float32)
        values[0:3, 0:3] = 1  # outside target roof
        values[8:12, 10:16] = 1
        values[20, 20] = 1  # below four-pixel threshold
        regions = worker.component_regions(values, worker.roof_mask((32, 32), ROOF))
        self.assertEqual(regions, [{"x": 10 / 32, "y": 8 / 32, "width": 6 / 32, "height": 4 / 32}])
        for values in (np.zeros((31, 32)), np.full((32, 32), float("nan")), np.full((32, 32), 1.1)):
            with self.assertRaises(worker.WorkerError):
                worker.component_regions(values, worker.roof_mask((32, 32), ROOF))

    def test_existing_vidda_pipeline_receives_native_rgb_and_fixed_profile(self):
        probabilities = np.zeros((32, 32), dtype=np.float32)
        probabilities[8:12, 8:12] = .6
        predict = Mock(return_value={"probabilities": probabilities})
        model, normalization = object(), object()
        segmenter = worker.ViddaSegmenter(model, "mps", predict, normalization)
        request = worker.decode_request(image_request())
        result = segmenter(request)
        self.assertEqual(result, [{"x": .25, "y": .25, "width": .125, "height": .125}])
        call = predict.call_args.kwargs
        self.assertIs(call["model"], model); self.assertIs(call["normalize_fn"], normalization)
        self.assertEqual({key: call[key] for key in ("tile_size", "overlap", "tta_mode", "temperature", "scales", "device", "batch_size")},
                         {"tile_size": 400, "overlap": .25, "tta_mode": "none", "temperature": 1., "scales": (1.,), "device": "mps", "batch_size": 1})
        np.testing.assert_allclose(call["image"][0, 0], [1., 128 / 255., 0.])
        self.assertEqual(call["image"].shape, (32, 32, 3))
        request.rgb.close()

    def test_model_load_is_weights_only_strict_offline_and_explicit(self):
        model, builder, predict, normalize = Mock(), Mock(), Mock(), Mock()
        builder.return_value = model
        torch = SimpleNamespace(load=Mock(return_value={"baseline_weights_version": "1.0", "model_state_dict": {"layer": "tensor"}}), set_num_threads=Mock(),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        modules = {"torch": torch,
                   "pvextent.models.deeplabv3plus": SimpleNamespace(build_deeplabv3plus_resnet50=builder),
                   "pvextent.models.resnet_unet": SimpleNamespace(normalize_rgb=normalize),
                   "pvextent.inference.tiler": SimpleNamespace(predict_tiled=predict)}
        config = worker.WorkerConfig(Path("/not-read-by-test.pt"), worker.VIDDA_V5_SHA256, TOKEN)
        with patch.object(worker, "checkpoint_bytes", return_value=b"verified artifact") as checked, patch.dict(sys.modules, modules):
            loaded = worker.ViddaSegmenter.load(config)
        checked.assert_called_once_with(config.checkpoint, config.checkpoint_sha256)
        builder.assert_called_once_with(None)
        self.assertEqual(torch.load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        model.load_state_dict.assert_called_once_with({"layer": "tensor"}, strict=True)
        model.to.assert_called_once_with("mps"); model.eval.assert_called_once()
        torch.set_num_threads.assert_called_once_with(2)
        self.assertIs(loaded.predictor, predict)

    def test_explicit_accelerator_does_not_fall_back_to_cpu(self):
        torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
                                cuda=SimpleNamespace(is_available=lambda: False))
        self.assertEqual(worker.select_device(torch, "cpu"), "cpu")
        self.assertEqual(worker.select_device(torch, "auto"), "cpu")
        for requested in ("mps", "cuda", "unknown"):
            with self.assertRaises(worker.WorkerError):
                worker.select_device(torch, requested)

    def test_checkpoint_and_private_token_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            companion = Path(str(path) + ".sha256")
            path.write_bytes(b"bounded checkpoint fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            companion.write_text(digest)
            with self.assertRaises(worker.WorkerError):
                worker.checkpoint_bytes(path, worker.VIDDA_V5_SHA256)
            with patch.object(worker, "VIDDA_V5_SHA256", digest):
                self.assertEqual(worker.checkpoint_bytes(path, digest), b"bounded checkpoint fixture")
                path.write_bytes(b"changed")
                with self.assertRaises(worker.WorkerError):
                    worker.checkpoint_bytes(path, digest)
            token = Path(directory) / "token"
            token.write_text(TOKEN + "\n"); token.chmod(0o600)
            self.assertEqual(worker.read_token({"PVEXTENT_WORKER_TOKEN_FILE": str(token)}), TOKEN)
            token.chmod(0o644)
            with self.assertRaises(worker.WorkerError):
                worker.read_token({"PVEXTENT_WORKER_TOKEN_FILE": str(token)})
            with self.assertRaises(worker.WorkerError):
                worker.read_token({"PVEXTENT_WORKER_TOKEN": TOKEN, "PVEXTENT_WORKER_TOKEN_FILE": str(token)})
            for kwargs in ({"host": "0.0.0.0"}, {"token": "short"}, {"device": "unavailable"}, {"checkpoint_sha256": "0" * 64}, {"threads": 0}, {"threads": 5}):
                values = {"checkpoint": path, "checkpoint_sha256": worker.VIDDA_V5_SHA256, "token": TOKEN, **kwargs}
                with self.assertRaises(worker.WorkerError):
                    worker.WorkerConfig(**values).validate()


class TestWorkerHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predict = Mock(return_value=[{"x": .25, "y": .25, "width": .1, "height": .1}])
        cls.application = worker.WorkerApplication(TOKEN, cls.predict)
        cls.server = worker.WorkerServer(("127.0.0.1", 0), worker.handler_for(cls.application))
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(timeout=2)

    def setUp(self):
        self.predict.reset_mock()
        self.predict.return_value = [{"x": .25, "y": .25, "width": .1, "height": .1}]
        self.predict.side_effect = None

    def request(self, method="GET", path="/health", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=3)
        raw = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        connection.request(method, path, raw, {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        status, actual_headers = response.status, dict(response.getheaders())
        data = json.loads(response.read())
        connection.close()
        return status, actual_headers, data

    def test_health_is_authenticated_ready_and_versioned_without_inference(self):
        status, headers, value = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(value["protocol"], worker.PROTOCOL)
        self.assertEqual(value["modelVersion"], worker.MODEL_VERSION)
        self.assertEqual(value["modelSha256"], worker.VIDDA_V5_SHA256)
        self.assertEqual(value["qualityStatus"], "experimental")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.predict.assert_not_called()
        for authorization in ("Bearer wrong", "Bearer ä", ""):
            self.assertEqual(self.request(headers={"Authorization": authorization})[0], 401)

    def test_segment_echoes_exact_hash_and_keeps_proposals_experimental(self):
        request = image_request()
        status, headers, value = self.request("POST", "/v1/segment", request)
        self.assertEqual(status, 200)
        self.assertEqual(value["imageSha256"], request["image"]["sha256"])
        self.assertEqual(value["regions"], self.predict.return_value)
        self.assertEqual(value["outcome"], "candidates")
        self.assertEqual(value["qualityStatus"], "experimental")
        self.assertLessEqual(value["durationMs"], 30_000)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.predict.return_value = []
        value = self.request("POST", "/v1/segment", request)[2]
        self.assertEqual(value["outcome"], "no_candidates")
        self.assertTrue(any("do not establish PV absence" in limitation for limitation in value["limitations"]))

    def test_concurrent_image_is_rejected_but_health_remains_available(self):
        self.application.slot.acquire()
        try:
            self.assertEqual(self.request("POST", "/v1/segment", image_request())[0], 429)
            self.assertTrue(self.request()[2]["busy"])
            self.predict.assert_not_called()
        finally:
            self.application.slot.release()

    def test_bad_payloads_content_types_and_body_limits_never_infer(self):
        self.assertEqual(self.request("POST", "/v1/segment", b"not json")[0], 400)
        self.assertEqual(self.request("POST", "/v1/segment", b'{"value":NaN}')[0], 400)
        self.assertEqual(self.request("POST", "/v1/segment", image_request(), {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/v1/segment", b"", {"Content-Length": str(worker.MAX_BODY_BYTES + 1)})[0], 413)
        self.assertEqual(self.request("POST", "/v1/segment", image_request(), {"Transfer-Encoding": "chunked"})[0], 400)
        self.assertEqual(self.request("GET", "/health?token=not-accepted")[0], 404)
        self.assertEqual(self.request("POST", "/v1/segment", {"url": "https://example.invalid"})[0], 400)
        self.predict.assert_not_called()

    def test_inference_failures_release_slot_without_leaking_payload_or_traceback(self):
        self.predict.side_effect = RuntimeError("internal private detail")
        status, _, value = self.request("POST", "/v1/segment", image_request())
        self.assertEqual(status, 503)
        self.assertNotIn("private detail", json.dumps(value))
        self.assertFalse(self.application.slot.locked())
        self.predict.side_effect = None
        self.assertEqual(self.request("POST", "/v1/segment", image_request())[0], 200)


if __name__ == "__main__":
    unittest.main()
