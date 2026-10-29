# Experimental HTTP worker

Spacement PVExtent can run as a separate, authenticated loopback service on the
Mac mini. It uses the existing Vidda V5 DeepLabV3+/ResNet50 model and its measured
inference profile: native RGB, ImageNet normalization, 400 px tiles, 0.25 overlap,
threshold 0.5, no TTA, one scale and temperature 1.0. The service loads the
complete weights offline with `encoder_weights=None`; it does not download a
model, fetch an image URL, train, or write received imagery to disk.

This worker release accepts only the existing Vidda V5 checkpoint:

```text
runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt
SHA-256 9380f5b7ebf5af23aeb445832402714865e387db5ce496d070575aada6e351a6
```

The companion `best.pt.sha256` must match too. Keep both artifacts out of Git.
Readiness is reported only after the checksum, strict weights-only state load,
offline architecture construction and requested device initialization succeed.
The architecture's existing constructor initializes the segmentation model
without downloading pretrained encoder weights. There is no random-model or
CPU fallback when the operator explicitly requests MPS.

## Start on the Mac mini

Use an existing Python 3.11+ environment prepared with this repository's locked
`baseline` dependencies. Install dependencies separately if they are missing;
starting this worker never installs them. Configure a private copy of
[`ops/worker.env.example`](../ops/worker.env.example), with absolute paths for
the checkpoint and a private bearer-token file (mode 600, at least 32 random
ASCII characters). Alternatively set `PVEXTENT_WORKER_TOKEN`; set one token
source only. Token contents are never printed by the service.

From the repository root, after exporting that configuration:

```sh
.venv/bin/python -m pvextent.worker --device mps --host 127.0.0.1 --port 8791
```

`PVEXTENT_CHECKPOINT` selects the local artifact; `PVEXTENT_CHECKPOINT_SHA256`
defaults to the fixed release hash above, and `PVEXTENT_DEVICE` defaults to
`mps`. CPU threads default to two (`PVEXTENT_THREADS` / `--threads`, range 1–4).
The `--checkpoint`, `--checkpoint-sha256` and `--device` arguments can
also supply these values. A different checkpoint hash is deliberately rejected
by this fixed worker release. The port can be set with `--port`.

The service binds exclusively to `127.0.0.1`. Use an existing authenticated SSH
route and a loopback-only forwarding port for the local preview, for example:

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:48791:127.0.0.1:8791 mini-remote
```

Configure the preview's server-side `PVEXTENT_WORKER_URL` as
`http://127.0.0.1:48791` and use the same `PVEXTENT_WORKER_TOKEN`. Keep those
settings out of browser-exposed variables. No change to the production OSINT
application or public Cloudflare ingress is required. A separately configured
HTTPS route must retain authentication and the same bounded worker contract.

## Isolated source releases

Keep worker source separate from the existing training/research checkout. The
running preview uses this layout on the Mac mini:

| Path below `/Users/paulroder/` | Purpose |
| --- | --- |
| `spacement-pvextent-worker/releases/worker-v1-bce383d74aa4/src/pvextent/` | Isolated Python source snapshot, 60 source files and approximately 198 kB |
| `spacement-pvextent/.venv/bin/python` | Existing Python 3.11 environment, reused without installing or updating dependencies |
| `spacement-pvextent/runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt` | Existing 107 MB checkpoint, read in place with its SHA-256 companion |
| `spacement-pvextent-worker/worker-token` | Private mode-600 bearer-token file; no token value belongs in Git or the plist |
| `spacement-pvextent-worker/logs/worker.log` | Ready message and normal process output |
| `spacement-pvextent-worker/logs/worker.err.log` | Startup errors and process diagnostics |
| `Library/LaunchAgents/co.spacement.pvextent-preview.plist` | Private mode-600 user LaunchAgent |

The release directory name is a source-snapshot identifier, not a Git commit
claim. The tested `src/pvextent/worker.py` has SHA-256
`44b5a0e0b0351d5d6f114c3e30555308a336074d0c4835c0daf9a6ed624e25a6`.
The checkpoint identity is separate and remains the fixed model hash above.

For an update, create a new, empty release directory and copy only Python
source. Do not pull into, reset, synchronize over, or replace the existing
research checkout. An example source transfer, from this repository, is:

```sh
rsync -a --prune-empty-dirs --include='*/' --include='*.py' --exclude='*' \
  src/ mini-remote:spacement-pvextent-worker/releases/worker-v1-20261025-next/src/
```

Create the destination's new release directory first, and choose a different
name if it already exists. Treat completed release directories as immutable.
This transfer includes the package's Python files and `__init__.py` files;
it excludes datasets, image files, checkpoints, environment files, bytecode,
and virtual environments. Retain and verify the source file checksums with the
release. A Python syntax check does not load the model.

Point the private LaunchAgent's `WorkingDirectory` and `PYTHONPATH` to the new
release. The latter must end in `/src`, so it selects the isolated package
instead of the existing checkout. Keep the existing interpreter, checkpoint
and token paths unchanged. Restart the agent, verify the authenticated health
response and one explicit image request, and retain the preceding release for
rollback. Switching back means restoring those two paths and restarting the
agent; it does not require changing model files or the research checkout.

## User LaunchAgent

[`ops/co.spacement.pvextent-preview.plist.example`](../ops/co.spacement.pvextent-preview.plist.example)
reproduces the current process configuration. It uses MPS explicitly, sets
`PYTORCH_ENABLE_MPS_FALLBACK=0`, limits CPU threads to two, and listens on
`127.0.0.1:8791`. `RunAtLoad` starts it when loaded; `KeepAlive.SuccessfulExit`
is false, so unexpected failures are restarted with a 30-second throttle.
No access-token value is embedded in the template.

Adapt all absolute paths and the source release ID before installing a private
copy at `~/Library/LaunchAgents/co.spacement.pvextent-preview.plist`. Create the
worker/log directories first; launchd does not create missing parent directories
for log files. Use a mode-600 token file and plist. The example's user is
`paulroder`, UID 501; substitute the actual logged-in user's UID on another host.
Run these commands as that user on the Mini:

```sh
plutil -lint "$HOME/Library/LaunchAgents/co.spacement.pvextent-preview.plist"
launchctl bootstrap gui/501 "$HOME/Library/LaunchAgents/co.spacement.pvextent-preview.plist"
launchctl print gui/501/co.spacement.pvextent-preview
```

Restart an already loaded worker:

```sh
launchctl kickstart -k gui/501/co.spacement.pvextent-preview
```

Stop and unload it, including its automatic restarts:

```sh
launchctl bootout gui/501/co.spacement.pvextent-preview
```

Load it again with `bootstrap`. Killing only the Python process is not a
persistent stop because launchd can restart it. `launchctl print` and the ready
line in `logs/worker.log` show the service/device state. Authentication and one
explicit image request still need separate verification.

This is a **user LaunchAgent**, not a system LaunchDaemon. The `gui/501` session
must exist: it starts after that user logs in and is unavailable before login or
after logout. `RunAtLoad` does not make it an unattended boot service. A later
decision about operation before login and durable ML jobs remains separate.
The local preview also needs its SSH forwarding process; stopping that process
disconnects the preview even while the Mini's worker continues to run.

## Contract

Every request needs `Authorization: Bearer <private worker token>`. Credentials
in query strings are not accepted. Responses are JSON with `Cache-Control:
no-store`. The worker exposes no CORS policy or unauthenticated health details.

`GET /health` returns:

```json
{
  "protocol": "spacement-pvextent-worker/v1",
  "status": "ready",
  "modelVersion": "pvextent-vidda-v1.0.0-v5",
  "modelSha256": "9380f5b7ebf5af23aeb445832402714865e387db5ce496d070575aada6e351a6",
  "qualityStatus": "experimental",
  "outputPolicy": "experimental_candidates_only",
  "busy": false
}
```

`POST /v1/segment` accepts only a JSON object containing `schemaVersion: 1`,
`image` and `roofPixels`. `image` contains `mimeType` (`image/png` or
`image/jpeg`), `bytesBase64`, `sha256`, `width` and `height`. Actual decoded
type, dimensions and SHA-256 must match. The input stays in its original pixel
grid; images are not resized to make small modules appear detectable.

`roofPixels` is a normalized multipolygon: polygons contain an exterior ring
followed by holes; rings contain closed `[x,y]` pairs in `[0,1]`, with origin at
the top left. The mask clips model candidates to the selected roof. Connected
components with at least four pixels yield normalized `{x,y,width,height}`
boxes. Boxes may span gaps or holes and still require visual assignment.

The response contains `protocol`, `modelVersion`, `modelSha256`, `imageSha256`,
`regions` and `durationMs`, matching the preview's adapter. Additional
`qualityStatus: "experimental"`, `outcome: "candidates" | "no_candidates"` and
limitations make the proposal status explicit. The worker never emits a
confirmed positive, a confirmed negative, current PV freedom, an installation
date or a demolition verdict. A dated human review remains separate.

## Limits and failure behavior

- One image inference at a time. Other image requests get `429`; health remains
  available. There is no hidden in-memory job queue or automatic retry.
- At most 2 MiB raster bytes, 32–1024 px per side, 4 MiB JSON, 20,000 roof
  vertices and 200 returned regions. Invalid payloads fail before model use.
- The client deadline is 30 seconds. A computation cannot safely be interrupted
  inside an MPS kernel; the slot stays occupied until it finishes, and results
  past 30 seconds are discarded with a timeout response. This synchronous
  worker is not the later durable ML job processor.
- Image bytes exist only for the request lifetime. HTTP access logs and request
  payload logs are disabled. Model exceptions return an opaque failure rather
  than credentials, payloads or tracebacks.
- Model proposals remain experimental. The Vidda NRW transfer results and
  limitations in the main README still apply; this transport does not change
  accuracy or constitute a production quality release.

## Test without model inference

```sh
.venv/bin/python -m unittest tests.test_worker -v
```

The tests decode tiny synthetic PNGs, inject a model/predictor, exercise the real
loopback HTTP contract and verify masking, holes, hash identity, offline loader
arguments, byte limits, auth, single-flight behavior and failures. They do not
load a real checkpoint, download weights, run GPU inference or claim benchmark
results.

## Observed integration check on 25 October 2026

The isolated `worker-v1-bce383d74aa4` source release ran on the Mac mini through
the existing Python environment, fixed Vidda V5 artifact and explicit MPS
device. A real, archived 768 × 768 px NRW DOP image of Maschinenstraße 22,
captured on 2 July 2025, completed in **1.536323 seconds**. The returned image
hash matched the stored image bytes, and the preview persisted the experimental
proposal. The model returned zero candidate regions. The existing manual
“Unklar” finding remained unchanged.

This verifies one transport/inference/persistence path. It does not measure
accuracy, establish PV absence, prove continuous availability after logout or
reboot, or complete the later durable ML job-processing module.
