import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "day1_understand_labels.ipynb"
PINNED_VERSION = "8.4.145"


class RepositoryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )
        cls.code_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def test_notebook_is_valid_and_has_no_stored_runtime_output(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        cell_ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertTrue(all(cell_ids))
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        code_cells = [
            cell for cell in self.notebook["cells"] if cell.get("cell_type") == "code"
        ]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(cell.get("outputs") == [] for cell in code_cells))

    def test_python_cells_parse_after_removing_colab_magic(self):
        python_source = "\n".join(
            line for line in self.code_source.splitlines() if not line.startswith("%")
        )
        ast.parse(python_source)

    def test_ultralytics_version_is_pinned_in_notebook_and_requirements(self):
        requirement = (ROOT / "requirements.txt").read_text(encoding="utf-8").strip()
        self.assertEqual(requirement, f"ultralytics=={PINNED_VERSION}")
        self.assertIn(f"%pip -q install ultralytics=={PINNED_VERSION}", self.code_source)
        self.assertNotIn("install -U ultralytics", self.code_source)

    def test_classroom_models_and_output_filenames_are_preserved(self):
        expected_models = {
            "yolo11n-cls.pt",
            "yolo11n.pt",
            "yolo11n-seg.pt",
        }
        expected_outputs = {
            "classification_predictions.json",
            "detection_predictions.json",
            "segmentation_predictions.json",
            "classification_top5.png",
            "detection_predictions.png",
            "segmentation_prediction.png",
            "IMAGE_ATTRIBUTION.md",
            "day1_lab_outputs.zip",
        }
        for value in expected_models | expected_outputs:
            with self.subTest(value=value):
                self.assertIn(value, self.all_source)

    def test_asset_manifests_have_fixed_sha256_values(self):
        checksums = set(re.findall(r'"sha256": "([0-9a-f]{64})"', self.code_source))
        self.assertEqual(len(checksums), 7)
        for expected in (
            "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
            "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
            "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
        ):
            self.assertIn(expected, checksums)
        self.assertIn("coco_image_id", self.code_source)
        self.assertIn("Checksum sai", self.code_source)
        self.assertEqual(self.code_source.count('"license_url": "https://creativecommons.org/licenses/by/2.0/"'), 3)

    def test_report_template_is_prepared_without_overwriting_student_work(self):
        self.assertIn("REPORT_TEMPLATE_ASSET", self.code_source)
        report_template_sha256 = hashlib.sha256(
            (ROOT / "reports" / "REPORT_TEMPLATE.md").read_bytes()
        ).hexdigest()
        self.assertIn(
            report_template_sha256,
            self.code_source,
        )
        self.assertRegex(
            self.code_source,
            r'REPORT_TEMPLATE_ASSET = \{\s*"url": "https://raw\.githubusercontent\.com/'
            r'[^\"]+/[0-9a-f]{7,40}/reports/REPORT_TEMPLATE\.md"',
        )
        self.assertIn(
            "https://raw.githubusercontent.com/VinUni-AI20k/"
            "Day1-Data-Overview-AI-ML-DL-Student/"
            "372b90e8e867530e559eed0424d4a594ec74163c/"
            "reports/REPORT_TEMPLATE.md",
            self.code_source,
        )
        self.assertIn("if not REPORT_PATH.exists()", self.code_source)
        self.assertIn("Giữ nguyên REPORT.md hiện có", self.code_source)
        self.assertIn('target.with_name(f".{target.name}.download")', self.code_source)
        self.assertIn("temporary_target.replace(target)", self.code_source)

    def test_models_are_downloaded_and_verified_before_loading(self):
        first_model_load = self.code_source.index("classification_model = YOLO")
        verification = self.code_source.index("MODEL_SHA256[model_file] = actual_sha256")
        self.assertLess(verification, first_model_load)
        self.assertIn("https://github.com/ultralytics/assets/releases/download/v8.4.0/", self.code_source)

    def test_setup_clears_stale_output(self):
        self.assertIn("shutil.rmtree(OUTPUT_DIR)", self.code_source)

    def test_submission_contract_keeps_identity_out_of_artifacts(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        report = (ROOT / "reports" / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
        self.assertIn("<KHOA>-DAY01-HoVaTen-MSSV", readme)
        self.assertIn("Họ tên và MSSV chỉ xuất hiện trong tên repository", readme)
        for document in (readme, report):
            self.assertIn("<KHOA>-DAY01-report.zip", document)
            self.assertNotIn("**Họ và tên:**", document)
            self.assertNotIn("**MSSV:**", document)
        self.assertNotIn("HO_VA_TEN_KHONG_DAU", self.code_source)
        self.assertNotRegex(self.code_source, r'^MSSV\s*=', msg="Notebook must not collect MSSV")

    def test_report_folder_is_the_git_submission_boundary(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        guide = (ROOT / "GUIDE.md").read_text(encoding="utf-8")
        report_template = (ROOT / "reports" / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
        self.assertTrue((ROOT / "report" / ".gitkeep").is_file())
        for document in (readme, guide, report_template):
            self.assertIn("report/", document)
        self.assertIn("git add report", readme)
        self.assertIn("commit", readme)
        self.assertIn("push", readme)

    def test_report_evidence_is_not_ignored_by_git(self):
        runtime_output = subprocess.run(
            ["git", "check-ignore", "--no-index", "day1_lab_outputs/runtime.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        submitted_output = subprocess.run(
            ["git", "check-ignore", "--no-index", "report/day1_lab_outputs/evidence.json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(runtime_output.returncode, 0, runtime_output.stderr)
        self.assertEqual(submitted_output.returncode, 1, submitted_output.stdout)

    def test_colab_packages_one_complete_submission_zip_in_drive(self):
        cell_ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertGreater(
            cell_ids.index("drive-submission-package"),
            cell_ids.index("validate-and-archive"),
        )
        for value in (
            'from google.colab import drive',
            'drive.mount("/content/drive")',
            '/content/drive/MyDrive/AI20K-Day1',
            'submission_name = f"{KHOA}-DAY01-report"',
            'shutil.copy2(REPORT_PATH',
            'shutil.copytree(OUTPUT_DIR',
            'root_dir=submission_dir',
            'zipfile.ZipFile(path)',
            'archive.testzip()',
            'pending_drive_path.replace(submission_archive_path)',
            'SKIP: ô lưu Google Drive chỉ chạy trong Colab',
        ):
            with self.subTest(value=value):
                self.assertIn(value, self.code_source)
        package_source = "".join(
            next(
                cell["source"]
                for cell in self.notebook["cells"]
                if cell.get("id") == "drive-submission-package"
            )
        )
        self.assertIn('"REPORT.md"', package_source)
        self.assertNotIn('f"{submission_name}/REPORT.md"', package_source)

    def test_submission_zip_has_repo_ready_root_layout(self):
        package_source = "".join(
            next(
                cell["source"]
                for cell in self.notebook["cells"]
                if cell.get("id") == "drive-submission-package"
            )
        )
        package_source = package_source.replace('KHOA = "KX"', 'KHOA = "K4"')

        with tempfile.TemporaryDirectory(prefix="day1-submission-test-") as temporary_directory:
            test_root = Path(temporary_directory)
            output_dir = test_root / "day1_lab_outputs"
            visual_dir = output_dir / "visuals"
            visual_dir.mkdir(parents=True)
            json_names = {
                "classification_predictions.json",
                "detection_predictions.json",
                "segmentation_predictions.json",
            }
            png_names = {
                "visuals/classification_top5.png",
                "visuals/detection_predictions.png",
                "visuals/segmentation_prediction.png",
            }
            for filename in {"IMAGE_ATTRIBUTION.md", *json_names, *png_names}:
                target = output_dir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"test evidence")

            report_path = test_root / "REPORT.md"
            report_path.write_text(
                "# Báo cáo đã hoàn thành\n\nDẫn chứng: sample-001.\n",
                encoding="utf-8",
            )
            drive_output_dir = test_root / "drive" / "MyDrive" / "AI20K-Day1"
            package_source = package_source.replace(
                "/content/drive/MyDrive/AI20K-Day1",
                str(drive_output_dir),
            )

            class FakeDrive:
                def __init__(self):
                    self.mount_calls = []

                def mount(self, path):
                    self.mount_calls.append(path)

            fake_drive = FakeDrive()
            google_module = types.ModuleType("google")
            colab_module = types.ModuleType("google.colab")
            colab_module.drive = fake_drive
            google_module.colab = colab_module
            namespace = {
                "Path": Path,
                "OUTPUT_DIR": output_dir,
                "REPORT_PATH": report_path,
                "REPORT_TEMPLATE_ASSET": {"sha256": "0" * 64},
                "sha256_file": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
                "re": re,
                "required_json": {name: {} for name in json_names},
                "required_png": list(png_names),
                "shutil": shutil,
                "zipfile": zipfile,
            }
            previous_directory = Path.cwd()
            try:
                os.chdir(test_root)
                with patch.dict(
                    sys.modules,
                    {"google": google_module, "google.colab": colab_module},
                ):
                    exec(
                        compile(package_source, "drive-submission-package", "exec"),
                        namespace,
                    )
            finally:
                os.chdir(previous_directory)

            archive_path = Path(namespace["submission_archive_path"])
            with zipfile.ZipFile(archive_path) as archive:
                actual_files = {
                    item.filename for item in archive.infolist() if not item.is_dir()
                }
                bad_member = archive.testzip()
            expected_files = {
                "REPORT.md",
                "day1_lab_outputs/IMAGE_ATTRIBUTION.md",
                *(f"day1_lab_outputs/{name}" for name in json_names),
                *(f"day1_lab_outputs/{name}" for name in png_names),
            }
            self.assertIsNone(bad_member)
            self.assertEqual(actual_files, expected_files)
            self.assertEqual(archive_path.name, "K4-DAY01-report.zip")
            self.assertEqual(fake_drive.mount_calls, ["/content/drive"])
            self.assertFalse((test_root / "day1_submission_staging").exists())

    def test_output_records_include_provenance_and_coordinate_context(self):
        for field in (
            "taxonomy_name",
            "model_file",
            "model_sha256",
            "ultralytics_version",
            "coordinate_unit",
            "bbox_format",
        ):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', self.code_source)
        self.assertIn('f"{sample_id}-{index + 1:03d}"', self.code_source)

    def test_notebook_validates_evidence_before_archiving(self):
        validation_position = self.code_source.index("required_json =")
        archive_position = self.code_source.index("shutil.make_archive")
        self.assertLess(validation_position, archive_position)
        self.assertIn("PASS: đủ", self.code_source)
        self.assertIn("for row_number, row in enumerate(rows, start=1)", self.code_source)
        self.assertIn("Image.open(path)", self.code_source)
        self.assertIn("expected_samples", self.code_source)

    def test_colab_badge_opens_frozen_student_release(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        expected = (
            "colab.research.google.com/github/VinUni-AI20k/Day1-Data-Overview-AI-ML-DL-Student/"
            "blob/v1.0.2/notebooks/day1_understand_labels.ipynb"
        )
        self.assertIn(expected, readme)
        self.assertIn("Phiên bản lớp", readme)

    def test_student_repository_is_documented_as_a_template(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        guide = (ROOT / "GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("Use this template", readme)
        self.assertIn("Public template", readme)
        self.assertIn("tạo repository bài làm từ template", guide)
        self.assertNotIn("Fork repository", readme)
        self.assertTrue(
            (ROOT / "docs" / "screenshots" / "01-use-template-repository.png").is_file()
        )
        self.assertFalse(
            (ROOT / "docs" / "screenshots" / "01-fork-repository.png").exists()
        )

    def test_primary_student_flow_is_cross_platform_and_browser_only(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        guide = (ROOT / "GUIDE.md").read_text(encoding="utf-8")
        for operating_system in ("Windows", "macOS", "Ubuntu"):
            with self.subTest(operating_system=operating_system):
                self.assertIn(operating_system, readme)
                self.assertIn(operating_system, guide)
        self.assertIn("không cần mở terminal", readme)
        self.assertIn("Git/terminal chỉ là lựa chọn thêm", guide)

    def test_repository_license_and_third_party_boundary_are_explicit(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("GNU AFFERO GENERAL PUBLIC LICENSE", license_text)
        self.assertIn("Version 3, 19 November 2007", license_text)
        self.assertIn("AGPL-3.0-only", notices)
        self.assertIn("AGPL-3.0-only", readme)
        self.assertIn("COCO", notices)

    def test_day1_has_no_cvat_runtime_or_deliverable(self):
        forbidden = ("cvat.ai", "docker compose", "pip install cvat", "cvat job")
        lowered = self.all_source.lower()
        for value in forbidden:
            with self.subTest(value=value):
                self.assertNotIn(value, lowered)

    def test_student_documents_match_formative_contract(self):
        report = (ROOT / "reports" / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
        rubric = (ROOT / "RUBRIC.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for filename in (
            "classification_predictions.json",
            "detection_predictions.json",
            "segmentation_predictions.json",
        ):
            self.assertIn(filename, report)
        self.assertIn("Không dùng CVAT", readme)
        self.assertIn("formative", rubric)
        self.assertNotIn("100 điểm", rubric)
        self.assertNotIn("59 điểm", rubric)


if __name__ == "__main__":
    unittest.main()
