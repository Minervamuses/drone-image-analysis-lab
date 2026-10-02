"""Checks for creating a fresh evaluation run directory."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from runs import RUN_SUBDIRECTORIES, allocate_run_directory, create_run_directory


class RunDirectoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "runs"
        self.moment = datetime(2026, 9, 19, 1, 23, 45, tzinfo=timezone.utc)

    def test_creates_a_timestamped_directory_with_the_expected_layout(self):
        run = create_run_directory(self.root, moment=self.moment)

        self.assertEqual(run.parent, self.root)
        self.assertEqual(run.name, "20260919T012345Z")
        for name in RUN_SUBDIRECTORIES:
            self.assertTrue((run / name).is_dir(), name)

    def test_refuses_to_reuse_an_existing_directory(self):
        create_run_directory(self.root, moment=self.moment)

        with self.assertRaises(FileExistsError):
            create_run_directory(self.root, moment=self.moment)

    def test_two_moments_produce_two_independent_directories(self):
        first = create_run_directory(self.root, moment=self.moment)
        (first / "marker.txt").write_text("kept", encoding="utf-8")

        second = create_run_directory(self.root, moment=self.moment.replace(second=46))

        self.assertNotEqual(first, second)
        self.assertEqual((first / "marker.txt").read_text(encoding="utf-8"), "kept")


class AllocateRunDirectoryTests(unittest.TestCase):
    """The anti-overwrite policy: suffix on collision, never write into an existing run."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "runs"
        self.moment = datetime(2026, 9, 19, 2, 30, 0, tzinfo=timezone.utc)

    def test_the_first_allocation_uses_the_plain_timestamp(self):
        run = allocate_run_directory(self.root, moment=self.moment)

        self.assertEqual(run.name, "20260919T023000Z")
        for name in RUN_SUBDIRECTORIES:
            self.assertTrue((run / name).is_dir(), name)

    def test_new_suite_can_omit_the_legacy_degradation_directories(self):
        first = allocate_run_directory(self.root, moment=self.moment, subdirectories=())
        second = allocate_run_directory(self.root, moment=self.moment, subdirectories=())
        self.assertNotEqual(first, second)
        self.assertEqual(list(first.iterdir()), [])
        self.assertEqual(list(second.iterdir()), [])

    def test_a_collision_gets_a_suffix_and_leaves_the_earlier_run_untouched(self):
        first = allocate_run_directory(self.root, moment=self.moment)
        marker = first / "report.md"
        marker.write_text("first run", encoding="utf-8")
        before = (marker.read_text(encoding="utf-8"), marker.stat().st_mtime_ns)

        second = allocate_run_directory(self.root, moment=self.moment)
        third = allocate_run_directory(self.root, moment=self.moment)

        self.assertEqual([second.name, third.name], ["20260919T023000Z-2", "20260919T023000Z-3"])
        self.assertEqual((marker.read_text(encoding="utf-8"), marker.stat().st_mtime_ns), before)
        self.assertEqual(list(second.iterdir()) and sorted(p.name for p in second.iterdir()), sorted(RUN_SUBDIRECTORIES))

    def test_a_pre_existing_directory_of_the_same_name_is_never_written_into(self):
        squatter = self.root / "20260919T023000Z"
        squatter.mkdir(parents=True)
        (squatter / "keep.txt").write_text("not ours", encoding="utf-8")
        before = sorted(p.name for p in squatter.iterdir())

        run = allocate_run_directory(self.root, moment=self.moment)

        self.assertEqual(run.name, "20260919T023000Z-2")
        self.assertEqual(sorted(p.name for p in squatter.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
