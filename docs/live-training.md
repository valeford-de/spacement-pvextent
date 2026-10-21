# Live training monitor

Start the local, read-only monitor from the repository's installed environment:

```sh
uv run --locked --no-sync pvextent monitor --repo-dir . --port 59654
```

Open the printed `http://127.0.0.1:59654/` address. The page refreshes every
two seconds and shows the current 1,000-photo mask proposal and independent
check counts. It also reports when `data/pvextent-training-v3` is prepared.
Starting the monitor does not start training.
`--no-sync` also avoids changing optional dependencies while another process
is training or reviewing data. Prepare the project environment first using
the [development setup](../README.md#development-setup) or baseline guide.

The default **Live / nächster Lauf** view selects an active baseline run when
one exists. Otherwise it explicitly reports that no training is active.
Completed runs can be selected separately; their history is labelled as
completed. Existing runs without batch telemetry show their recorded epoch
losses and validation metrics, rather than reconstructed batch values.

New baseline runs append flushed scalar events to `telemetry.jsonl` during
training, including each real optimization batch, completed epoch, run
completion and handled failure. The monitor reads these events in the staging
directory while a run is active, then reads the published directory after
completion. It shows training loss by optimizer step and internal validation
IoU/Dice by epoch. The logger does not change model updates, sampling or RNG
state. Exact CPU continuation remains covered by the baseline regression tests.

The server binds only to loopback. Its fixed UI and `/status.json` endpoint
read bounded local JSON logs and reports; they do not load imagery, masks,
checkpoints or model libraries. The page has no training controls or external
services. The browser must stay on this computer; the localhost link is not
a published dashboard. Stop the monitor process to close the service.

Internal validation scores remain development measurements, with the
limitations recorded in the [training audit](research/training-extension-audit.md).
The monitor does not evaluate the original test set or external pilot.
