# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Generate a tiny structural example, not real imagery or training evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New output directory; existing paths are never overwritten")
    target = parser.parse_args().directory
    if target.exists():
        parser.error("output path already exists; choose a new directory")
    target.mkdir(parents=True)
    (target / "images").mkdir()
    (target / "masks").mkdir()
    (target / "dataset.toml").write_text(
        'schema_version = "1.0"\ndataset_id = "pvextent-synthetic-demo"\n'
        'dataset_version = "1"\n\n[sources.synthetic]\n'
        'name = "Generated structural example; not orthophotography"\n'
        'url = "https://github.com/valeford-de/spacement-pvextent"\n'
        'version = "synthetic-v1"\nimage_license = "Apache-2.0"\n'
        'annotation_license = "Apache-2.0"\nattribution = "Copyright 2026 Valeford"\n',
        encoding="utf-8",
    )
    records = []
    for index, name in enumerate(("positive", "negative"), start=1):
        image = Image.new("RGB", (32, 32), (40, 50, 60))
        mask = Image.new("L", image.size, 0)
        if name == "positive":
            ImageDraw.Draw(image).rectangle((8, 8, 23, 23), fill=(0, 150, 120))
            ImageDraw.Draw(mask).rectangle((8, 8, 23, 23), fill=1)
            ImageDraw.Draw(mask).rectangle((0, 0, 3, 3), fill=255)
        image_path, mask_path = target / "images" / f"{name}.png", target / "masks" / f"{name}.png"
        image.save(image_path)
        mask.save(mask_path)
        records.append({
            "id": f"demo-{name}",
            "image": image_path.relative_to(target).as_posix(),
            "mask": mask_path.relative_to(target).as_posix(),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "width": 32, "height": 32, "gsd_m": 0.1,
            "source_id": "synthetic", "capture_date": None,
            "spatial_group": f"synthetic-area-{index}", "building_ids": [],
            "split": "unassigned", "annotation_status": "unreviewed",
            "georeference": None,
        })
    (target / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )
    print(f"Created {len(records)} synthetic examples in {target}. GSD is illustrative; no real places or acquisition dates are claimed.")


if __name__ == "__main__":
    main()
