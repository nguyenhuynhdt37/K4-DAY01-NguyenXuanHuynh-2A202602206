# Third-party data and software notice

> **Status:** license inventory for the classroom release. This notice is not legal advice.

## Ultralytics

The notebook installs `ultralytics==8.4.145` and downloads three public YOLO11n checkpoints on first use.
Ultralytics publishes its open-source package and models under AGPL-3.0 and also offers an enterprise
license. This classroom repository is public and licensed under `AGPL-3.0-only` to follow the open-source
distribution path; it does not grant permission for a private or proprietary product integration.

- Project: <https://github.com/ultralytics/ultralytics>
- License information: <https://www.ultralytics.com/license>
- Package release: <https://pypi.org/project/ultralytics/8.4.145/>

## COCO sample images

The notebook downloads three COCO 2017 validation images at runtime and records their COCO image ID and
SHA-256 checksum. The image files are not committed to this repository. Their license records were checked
against the official `instances_val2017.json` metadata. All three selected images are CC BY 2.0:

| Sample | COCO image ID | Original work / creator | Source | License |
| --- | ---: | --- | --- | --- |
| `traffic` | 210273 | “Wuhan” / Tauno Tõhk (`toehk`) | <https://www.flickr.com/photo.gne?id=5336041838> | <https://creativecommons.org/licenses/by/2.0/> |
| `kitchen` | 397133 | “Kitchen” / Maggie Stephens (`Pot Noodle`) | <https://www.flickr.com/photo.gne?id=6255196340> | <https://creativecommons.org/licenses/by/2.0/> |
| `dining` | 166918 | “Big Wine Thing” / `WordRidden` | <https://www.flickr.com/photo.gne?id=4745624149> | <https://creativecommons.org/licenses/by/2.0/> |

The generated PNG files add model predictions/charts to the source images and are marked as modified in
the `IMAGE_ATTRIBUTION.md` file generated inside each learner's output archive. CC BY 2.0 still requires
appropriate credit and a license link, and other rights may apply.

- COCO Terms of Use: <https://cocodataset.org/#termsofuse>
- COCO 2017 annotations: <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>
- CC BY 2.0 terms: <https://creativecommons.org/licenses/by/2.0/>

## Repository material

Repository-owned code and documentation are licensed under `AGPL-3.0-only`; see `LICENSE`. Ultralytics,
model checkpoints, COCO images, GitHub/Google interface screenshots, names, logos, and trademarks remain
subject to their respective owners' terms. The notebook downloads model checkpoints and sample images at
runtime rather than redistributing them in this repository. Preserve this notice and the generated
per-image attribution in copies of the coursework.
