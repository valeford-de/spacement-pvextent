# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Bounded, authenticated HTTP worker for experimental PVExtent proposals.

Importing this module never imports Torch, loads a checkpoint or starts a server.
Only ``main`` explicitly loads the model and starts the loopback listener.
Images and masks remain in memory; this service does not fetch source URLs.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import deque
from dataclasses import dataclass
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
import warnings

from PIL import Image, ImageDraw, UnidentifiedImageError

PROTOCOL = "spacement-pvextent-worker/v1"
MODEL_VERSION = "pvextent-vidda-v1.0.0-v5"
VIDDA_V5_SHA256 = "9380f5b7ebf5af23aeb445832402714865e387db5ce496d070575aada6e351a6"
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_IMAGE_SIDE = 1024
MAX_VERTICES = 20_000
MAX_REGIONS = 200
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_INFERENCE_MS = 30_000
MIN_COMPONENT_PIXELS = 4
HEX_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class WorkerError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


@dataclass(frozen=True)
class WorkerConfig:
    checkpoint: Path
    checkpoint_sha256: str
    token: str
    device: str = "mps"
    host: str = "127.0.0.1"
    port: int = 8791
    threads: int = 2

    def validate(self) -> None:
        if self.host != "127.0.0.1" or type(self.port) is not int or not 1 <= self.port <= 65535:
            raise WorkerError("invalid_configuration", "The worker must bind to 127.0.0.1 and a valid port.")
        if self.checkpoint_sha256 != VIDDA_V5_SHA256:
            raise WorkerError("invalid_configuration", "This worker release is pinned to the documented Vidda V5 checkpoint.")
        if self.device not in ("mps", "cpu", "cuda", "auto"):
            raise WorkerError("invalid_configuration", "Device must be mps, cpu, cuda or auto.")
        if type(self.threads) is not int or not 1 <= self.threads <= 4:
            raise WorkerError("invalid_configuration", "CPU thread count must be between one and four.")
        if (not 32 <= len(self.token) <= 4096 or not self.token.isascii()
                or any(ord(character) <= 32 or ord(character) == 127 for character in self.token)):
            raise WorkerError("invalid_configuration", "A private, whitespace-free bearer token of at least 32 ASCII characters is required.")


def read_token(environment: dict[str, str]) -> str:
    value, filename = environment.get("PVEXTENT_WORKER_TOKEN", ""), environment.get("PVEXTENT_WORKER_TOKEN_FILE", "")
    if value and filename:
        raise WorkerError("invalid_configuration", "Set one token source, not both token and token file.")
    if filename:
        path = Path(filename)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4097 or path.stat().st_mode & 0o077:
            raise WorkerError("invalid_configuration", "The token file must be a private regular file (mode 600).")
        value = path.read_text(encoding="ascii").strip()
    return value


def checkpoint_bytes(path: Path, expected: str) -> bytes:
    """Verify the fixed release hash and its companion before deserialization."""
    if expected != VIDDA_V5_SHA256:
        raise WorkerError("invalid_checkpoint", "Unsupported checkpoint identity.")
    companion = path.with_suffix(path.suffix + ".sha256")
    if (path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_CHECKPOINT_BYTES
            or companion.is_symlink() or not companion.is_file() or companion.stat().st_size > 128):
        raise WorkerError("invalid_checkpoint", "Expected a bounded local checkpoint and .sha256 companion.")
    if companion.read_text(encoding="ascii").strip() != expected:
        raise WorkerError("invalid_checkpoint", "The checkpoint companion differs from the pinned release.")
    with path.open("rb") as stream:
        payload = stream.read(MAX_CHECKPOINT_BYTES + 1)
    if len(payload) > MAX_CHECKPOINT_BYTES or hashlib.sha256(payload).hexdigest() != expected:
        raise WorkerError("invalid_checkpoint", "The checkpoint checksum differs from the pinned release.")
    return payload


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_roof(value: Any) -> list:
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise WorkerError("invalid_roof", "Expected normalized roof polygons, including inner rings.")
    count = 0
    for polygon in value:
        if not isinstance(polygon, list) or not 1 <= len(polygon) <= 100:
            raise WorkerError("invalid_roof", "Invalid polygon rings.")
        for ring in polygon:
            if not isinstance(ring, list) or len(ring) < 4:
                raise WorkerError("invalid_roof", "Roof rings must be closed and contain at least four vertices.")
            for point in ring:
                count += 1
                if (count > MAX_VERTICES or not isinstance(point, list) or len(point) != 2
                        or any(not _finite_number(coordinate) or not 0 <= coordinate <= 1 for coordinate in point)):
                    raise WorkerError("invalid_roof", "Roof vertices must stay within the immutable image.")
            area = abs(sum(point[0] * ring[(i + 1) % len(ring)][1]
                           - ring[(i + 1) % len(ring)][0] * point[1] for i, point in enumerate(ring))) / 2
            if ring[0] != ring[-1] or area <= 1e-10:
                raise WorkerError("invalid_roof", "Roof rings must be closed and nondegenerate.")
    return value


@dataclass
class ImageRequest:
    rgb: Image.Image
    sha256: str
    roof: list


def decode_request(value: Any) -> ImageRequest:
    if (not isinstance(value, dict) or set(value) != {"schemaVersion", "image", "roofPixels"}
            or type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1):
        raise WorkerError("invalid_request", "Expected schemaVersion 1, image and roofPixels only.")
    roof = validate_roof(value["roofPixels"])
    image = value["image"]
    if (not isinstance(image, dict) or set(image) != {"mimeType", "bytesBase64", "sha256", "width", "height"}
            or image["mimeType"] not in ("image/png", "image/jpeg")
            or not isinstance(image["sha256"], str) or not HEX_SHA256.fullmatch(image["sha256"])
            or any(type(image[key]) is not int or not 32 <= image[key] <= MAX_IMAGE_SIDE for key in ("width", "height"))
            or not isinstance(image["bytesBase64"], str) or len(image["bytesBase64"]) > 4 * ((MAX_IMAGE_BYTES + 2) // 3)):
        raise WorkerError("invalid_image", "Expected a bounded PNG/JPEG snapshot, dimensions and SHA-256.")
    try:
        raw = base64.b64decode(image["bytesBase64"], validate=True)
    except (binascii.Error, ValueError) as error:
        raise WorkerError("invalid_image", "Image bytes are not valid base64.") from error
    if not 0 < len(raw) <= MAX_IMAGE_BYTES or hashlib.sha256(raw).hexdigest() != image["sha256"]:
        raise WorkerError("invalid_image", "Image bytes do not match the bounded snapshot checksum.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw), formats=("PNG", "JPEG")) as decoded:
                actual_mime = {"PNG": "image/png", "JPEG": "image/jpeg"}.get(decoded.format)
                if (actual_mime != image["mimeType"] or decoded.size != (image["width"], image["height"])
                        or getattr(decoded, "n_frames", 1) != 1):
                    raise WorkerError("invalid_image", "Decoded image type and dimensions must match the snapshot.")
                decoded.verify()
            with Image.open(BytesIO(raw), formats=("PNG", "JPEG")) as decoded:
                rgb = decoded.convert("RGB")
                rgb.load()
    except WorkerError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise WorkerError("invalid_image", "The snapshot is not a valid bounded raster image.") from error
    return ImageRequest(rgb=rgb, sha256=image["sha256"], roof=roof)


def roof_mask(size: tuple[int, int], polygons: list) -> Image.Image:
    """Union exteriors while preserving holes per polygon, in image coordinates."""
    from PIL import ImageChops
    width, height = size
    combined = Image.new("L", size, 0)
    for polygon in polygons:
        part = Image.new("L", size, 0)
        draw = ImageDraw.Draw(part)
        for i, ring in enumerate(polygon):
            draw.polygon([(min(width - 1, x * width), min(height - 1, y * height)) for x, y in ring], fill=0 if i else 255)
        combined = ImageChops.lighter(combined, part)
    return combined


def component_regions(probabilities: Any, roof: Image.Image) -> list[dict[str, float]]:
    """Four-connected candidate boxes, clipped to the actual roof mask first."""
    import numpy as np
    values = np.asarray(probabilities)
    width, height = roof.size
    if values.shape != (height, width) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise WorkerError("invalid_prediction", "The model did not return a finite native-resolution probability map.", 502)
    pending = np.logical_and(values >= .5, np.asarray(roof) != 0)
    regions: list[dict[str, float]] = []
    for row in range(height):
        for column in np.flatnonzero(pending[row]):
            column = int(column)
            if not pending[row, column]:
                continue
            pending[row, column] = False
            queue = deque([(column, row)])
            count, left, right, top, bottom = 0, column, column, row, row
            while queue:
                x, y = queue.popleft()
                count += 1
                left, right, top, bottom = min(left, x), max(right, x), min(top, y), max(bottom, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < width and 0 <= ny < height and pending[ny, nx]:
                        pending[ny, nx] = False
                        queue.append((nx, ny))
            if count >= MIN_COMPONENT_PIXELS:
                regions.append({"x": left / width, "y": top / height,
                                "width": (right + 1 - left) / width, "height": (bottom + 1 - top) / height})
                if len(regions) > MAX_REGIONS:
                    raise WorkerError("too_many_candidates", "Too many disconnected proposals; manual image review is required.", 422)
    return regions


class ViddaSegmenter:
    """The measured Vidda V5 preprocessing and inference profile, without TTA."""

    def __init__(self, model: Any, device: str, predictor: Callable, normalize: Callable):
        self.model, self.device, self.predictor, self.normalize = model, device, predictor, normalize

    @classmethod
    def load(cls, config: WorkerConfig) -> "ViddaSegmenter":
        config.validate()
        payload = checkpoint_bytes(config.checkpoint, config.checkpoint_sha256)
        # Heavy imports happen only when the operator explicitly starts the worker.
        import torch
        from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
        from pvextent.models.resnet_unet import normalize_rgb
        from pvextent.inference.tiler import predict_tiled
        device = select_device(torch, config.device)
        torch.set_num_threads(config.threads)
        document = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
        del payload
        if not isinstance(document, dict) or document.get("baseline_weights_version") != "1.0" or not isinstance(document.get("model_state_dict"), dict):
            raise WorkerError("invalid_checkpoint", "Unsupported Vidda weights-only checkpoint.")
        model = build_deeplabv3plus_resnet50(None)
        model.load_state_dict(document["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        del document
        return cls(model, device, predict_tiled, normalize_rgb)

    def __call__(self, request: ImageRequest) -> list[dict[str, float]]:
        import numpy as np
        image = np.asarray(request.rgb, dtype=np.float32) / 255.0
        result = self.predictor(model=self.model, image=image, tile_size=400, overlap=.25,
                                tta_mode="none", temperature=1., scales=(1.,), device=self.device,
                                normalize_fn=self.normalize, batch_size=1)
        return component_regions(result["probabilities"], roof_mask(request.rgb.size, request.roof))


def select_device(torch: Any, requested: str) -> str:
    """An explicitly requested accelerator never falls back to CPU."""
    if requested not in ("mps", "cpu", "cuda", "auto"):
        raise WorkerError("invalid_configuration", "Unknown inference device.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise WorkerError("device_unavailable", "MPS was requested but is unavailable; the worker will not fall back to CPU.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise WorkerError("device_unavailable", "CUDA was requested but is unavailable; the worker will not fall back to CPU.")
    return requested


class WorkerApplication:
    def __init__(self, token: str, segmenter: Callable[[ImageRequest], list[dict[str, float]]],
                 model_sha256: str = VIDDA_V5_SHA256):
        self.token, self.segmenter, self.model_sha256 = token, segmenter, model_sha256
        self.slot = threading.Lock()

    def health(self) -> dict:
        return {"protocol": PROTOCOL, "status": "ready", "modelVersion": MODEL_VERSION,
                "modelSha256": self.model_sha256, "qualityStatus": "experimental",
                "outputPolicy": "experimental_candidates_only", "busy": self.slot.locked()}

    def segment(self, value: Any) -> dict:
        if not self.slot.acquire(blocking=False):
            raise WorkerError("busy", "A PVExtent image is already being processed. Please retry later.", 429)
        try:
            start = time.monotonic()
            request = decode_request(value)
            try:
                regions = self.segmenter(request)
            finally:
                request.rgb.close()
            duration = round((time.monotonic() - start) * 1000, 3)
            if duration > MAX_INFERENCE_MS:
                raise WorkerError("inference_timeout", "The bounded inference deadline was exceeded; no proposal was published.", 504)
            if (not isinstance(regions, list) or len(regions) > MAX_REGIONS
                    or any(not isinstance(region, dict) or set(region) != {"x", "y", "width", "height"}
                           or any(not _finite_number(number) or not 0 <= number <= 1 for number in region.values())
                           or region["width"] <= 0 or region["height"] <= 0
                           or region["x"] + region["width"] > 1 or region["y"] + region["height"] > 1 for region in regions)):
                raise WorkerError("invalid_prediction", "The model returned invalid normalized proposal regions.", 502)
            return {"protocol": PROTOCOL, "modelVersion": MODEL_VERSION, "modelSha256": self.model_sha256,
                    "imageSha256": request.sha256, "regions": regions, "durationMs": duration,
                    "qualityStatus": "experimental", "outcome": "candidates" if regions else "no_candidates",
                    "limitations": ["Experimental model proposals require a dated image review.",
                                    "No candidates do not establish PV absence or the current condition.",
                                    "Region boxes can include gaps and must be assigned to the selected roof."]}
        finally:
            self.slot.release()


def handler_for(application: WorkerApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Spacement-PVExtent/1"
        sys_version = ""

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, _format: str, *_args: Any) -> None:
            # Never record image payloads, request headers or bearer credentials.
            pass

        def send_json(self, status: int, value: dict) -> None:
            raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(raw)

        def dispatch(self) -> None:
            try:
                authorization = self.headers.get_all("Authorization", [])
                if len(authorization) != 1 or not hmac.compare_digest(authorization[0].encode("utf-8"), f"Bearer {application.token}".encode("ascii")):
                    raise WorkerError("unauthorized", "A valid worker bearer credential is required.", 401)
                if self.command == "GET" and self.path == "/health":
                    return self.send_json(200, application.health())
                if self.command != "POST" or self.path != "/v1/segment":
                    raise WorkerError("not_found", "Unknown worker operation.", 404)
                if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                    raise WorkerError("invalid_request", "Only an uncompressed, length-delimited JSON request is supported.")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
                    raise WorkerError("length_required", "One valid Content-Length header is required.", 411)
                length = int(lengths[0])
                if not 0 < length <= MAX_BODY_BYTES:
                    raise WorkerError("body_limit", "The worker request exceeds its byte budget.", 413)
                if not re.fullmatch(r"application/json(?:\s*;\s*charset=utf-8)?", self.headers.get("Content-Type", ""), re.I):
                    raise WorkerError("content_type", "JSON content is required.", 415)
                if application.slot.locked():
                    raise WorkerError("busy", "A PVExtent image is already being processed. Please retry later.", 429)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise WorkerError("invalid_request", "The request body is incomplete.")
                try:
                    value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
                    raise WorkerError("invalid_json", "The request must contain finite, valid JSON.") from error
                return self.send_json(200, application.segment(value))
            except WorkerError as error:
                self.send_json(error.status, {"error": {"code": error.code, "message": str(error)}})
            except (BrokenPipeError, ConnectionError, TimeoutError):
                self.close_connection = True
            except Exception:
                self.send_json(503, {"error": {"code": "inference_failed", "message": "PVExtent could not complete this image; manual review remains available."}})

        do_GET = dispatch
        do_POST = dispatch
        do_PUT = dispatch
        do_DELETE = dispatch
        do_OPTIONS = dispatch

    return Handler


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spacement PVExtent experimental loopback worker")
    parser.add_argument("--checkpoint", default=os.environ.get("PVEXTENT_CHECKPOINT"), help="Local Vidda V5 best.pt with its .sha256 companion")
    parser.add_argument("--checkpoint-sha256", default=os.environ.get("PVEXTENT_CHECKPOINT_SHA256", VIDDA_V5_SHA256))
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda", "auto"), default=os.environ.get("PVEXTENT_DEVICE", "mps"))
    parser.add_argument("--threads", type=int, default=os.environ.get("PVEXTENT_THREADS", "2"))
    args = parser.parse_args(argv)
    try:
        if not args.checkpoint:
            raise WorkerError("invalid_configuration", "Set PVEXTENT_CHECKPOINT or --checkpoint to the existing Vidda V5 artifact.")
        config = WorkerConfig(checkpoint=Path(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
                              token=read_token(dict(os.environ)), device=args.device, host=args.host, port=args.port, threads=args.threads)
        config.validate()
        segmenter = ViddaSegmenter.load(config)
        application = WorkerApplication(config.token, segmenter, config.checkpoint_sha256)
        with WorkerServer((config.host, config.port), handler_for(application)) as server:
            print(f"Spacement PVExtent ready at http://{config.host}:{config.port}; {MODEL_VERSION}; device={segmenter.device}; experimental proposals only", flush=True)
            server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"PVExtent worker did not start: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
