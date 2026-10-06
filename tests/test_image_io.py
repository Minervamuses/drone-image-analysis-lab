import hashlib
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image, ImageOps

from drone_sr.image_io import read_image, unique_output_path, write_png


class ImageIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.png"
        self.destination = self.root / "output" / "source.png"

    def _oriented_source(self, path, orientation, image_format):
        # Non-square with four distinguishable corners: every orientation maps to
        # a different pixel layout, so rotations and mirrors cannot be confused.
        image = Image.new("RGB", (40, 20), (10, 10, 10))
        for corner, color in (
            ((0, 0), (255, 0, 0)),
            ((39, 0), (0, 255, 0)),
            ((0, 19), (0, 0, 255)),
            ((39, 19), (255, 255, 0)),
        ):
            image.putpixel(corner, color)
        exif = Image.Exif()
        exif[274] = orientation
        image.save(path, format=image_format, exif=exif)

    def test_rgb_channels_range_shape_and_png_round_trip(self):
        source = Image.new("RGB", (2, 1))
        source.putdata([(255, 0, 128), (0, 64, 255)])
        source.save(self.source)
        original = self.source.read_bytes()

        tensor = read_image(self.source)

        self.assertEqual(tensor.shape, (1, 3, 1, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(
            tensor,
            torch.tensor([[[[255, 0]], [[0, 64]], [[128, 255]]]]) / 255,
        )
        write_png(tensor, self.destination, self.source)
        with Image.open(self.destination) as output:
            self.assertEqual(output.format, "PNG")
            self.assertEqual(output.mode, "RGB")
            self.assertEqual(output.size, (2, 1))
            self.assertEqual(output.tobytes(), source.tobytes())
        self.assertEqual(self.source.read_bytes(), original)

    def test_grayscale_becomes_rgb_and_bad_image_raises(self):
        Image.new("L", (1, 1), 64).save(self.source)
        torch.testing.assert_close(read_image(self.source), torch.full((1, 3, 1, 1), 64 / 255))
        self.source.write_bytes(b"not an image")
        with self.assertRaises(OSError):
            read_image(self.source)

    def _write_png(self, path, width, depth, colortype, rows):
        # Hand-written so the source encoding is chosen, not assumed: Pillow cannot
        # save 16-bit colour PNGs, and Image.new() hides the bit depth it picks.
        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        def pack(values):
            if depth >= 8:
                return b"".join(struct.pack(">H" if depth == 16 else ">B", value) for value in values)
            per_byte = 8 // depth
            packed = bytearray()
            for start in range(0, len(values), per_byte):
                byte = 0
                for offset in range(per_byte):
                    if start + offset < len(values):
                        byte |= values[start + offset] << (8 - depth * (offset + 1))
                packed.append(byte)
            return bytes(packed)

        header = struct.pack(">IIBBBBB", width, len(rows), depth, colortype, 0, 0, 0)
        body = zlib.compress(b"".join(b"\x00" + pack(row) for row in rows))
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", body) + chunk(b"IEND", b"")
        )

    def _write_tiff(self, path, width, height, bits, values):
        # Single-strip little-endian TIFF with three samples; Pillow cannot save
        # 16-bit RGB TIFF either.
        pixels = b"".join(struct.pack("<H" if bits == 16 else "<B", value) for value in values)
        body = pixels + struct.pack("<3H", bits, bits, bits)
        entries = [
            (256, 4, 1, width), (257, 4, 1, height), (258, 3, 3, 8 + len(pixels)),
            (259, 3, 1, 1), (262, 3, 1, 2), (273, 4, 1, 8), (277, 3, 1, 3),
            (278, 4, 1, height), (279, 4, 1, len(pixels)), (284, 3, 1, 1),
        ]
        directory = struct.pack("<H", len(entries))
        for tag, kind, count, value in entries:
            directory += struct.pack("<HHLL", tag, kind, count, value)
        path.write_bytes(
            b"II*\x00" + struct.pack("<L", 8 + len(body)) + body + directory + struct.pack("<L", 0)
        )

    def test_high_bit_depth_is_rejected_explicitly(self):
        Image.new("I;16", (2, 2)).save(self.source)
        with self.assertRaisesRegex(ValueError, "Unsupported image mode"):
            read_image(self.source)

        # Pillow maps 16-bit colour PNGs onto RGB/RGBA, so the mode allow-list
        # above cannot see them; these reach read_image() looking like 8-bit.
        truecolour = self.root / "truecolour16.png"
        self._write_png(truecolour, 2, 16, 2, [[256, 257, 511, 300, 301, 302]])
        gray_alpha = self.root / "grayalpha16.png"
        self._write_png(gray_alpha, 2, 16, 4, [[256, 65535, 511, 65535]])
        with_alpha = self.root / "truecolouralpha16.png"
        self._write_png(with_alpha, 2, 16, 6, [[256, 257, 511, 65535, 1, 2, 3, 65535]])
        for source, colortype in ((truecolour, 2), (gray_alpha, 4), (with_alpha, 6)):
            with self.subTest(png_colortype=colortype):
                raw = source.read_bytes()
                self.assertEqual((raw[24], raw[25]), (16, colortype))
                with self.assertRaisesRegex(ValueError, "bit depth"):
                    read_image(source)

        tiff = self.root / "truecolour16.tif"
        self._write_tiff(tiff, 2, 1, 16, [256, 257, 511, 300, 301, 302])
        with Image.open(tiff) as opened:
            self.assertEqual(opened.tag_v2.get(258), (16, 16, 16))
            self.assertEqual(opened.mode, "RGB")
        with self.assertRaisesRegex(ValueError, "bit depth"):
            read_image(tiff)

    def test_low_bit_depth_sources_stay_accepted(self):
        one_bit = self.root / "gray1.png"
        self._write_png(one_bit, 2, 1, 0, [[1, 0]])
        two_bit = self.root / "gray2.png"
        self._write_png(two_bit, 2, 2, 0, [[3, 0]])
        four_bit = self.root / "gray4.png"
        self._write_png(four_bit, 2, 4, 0, [[15, 0]])
        palette = self.root / "palette.png"
        indexed = Image.new("P", (2, 1))
        indexed.putpalette([255, 0, 0, 0, 255, 0])
        indexed.putdata([0, 1])
        indexed.save(palette)
        truecolour = self.root / "truecolour8.png"
        self._write_png(truecolour, 2, 8, 2, [[10, 20, 30, 40, 50, 60]])
        tiff = self.root / "truecolour8.tif"
        self._write_tiff(tiff, 2, 1, 8, [10, 20, 30, 40, 50, 60])

        for source, depth, first_pixel in (
            (one_bit, 1, (255, 255, 255)),
            (two_bit, 2, (255, 255, 255)),
            (four_bit, 4, (255, 255, 255)),
            (palette, 1, (255, 0, 0)),
            (truecolour, 8, (10, 20, 30)),
        ):
            with self.subTest(source=source.name):
                self.assertEqual(source.read_bytes()[24], depth)
                tensor = read_image(source)
                self.assertEqual(tensor.shape, (1, 3, 1, 2))
                self.assertEqual(
                    tuple(round(float(tensor[0, channel, 0, 0]) * 255) for channel in range(3)),
                    first_pixel,
                )
        with self.subTest(source=tiff.name):
            torch.testing.assert_close(
                read_image(tiff),
                torch.tensor([[[[10, 40]], [[20, 50]], [[30, 60]]]], dtype=torch.float32) / 255,
            )

    def test_exif_orientation_is_applied_to_pixels(self):
        for image_format, suffix in (("JPEG", ".jpg"), ("PNG", ".png")):
            for orientation in range(1, 9):
                with self.subTest(format=image_format, orientation=orientation):
                    source = self.root / f"oriented_{orientation}{suffix}"
                    self._oriented_source(source, orientation, image_format)
                    before = hashlib.sha256(source.read_bytes()).hexdigest()
                    with Image.open(source) as opened:
                        stored = opened.convert("RGB").tobytes()
                        expected = ImageOps.exif_transpose(opened).convert("RGB")
                    if orientation != 1:
                        # Guard the fixture: orientations 2-4 keep the size, so a
                        # size-only comparison would pass without any correction.
                        self.assertNotEqual(expected.tobytes(), stored)

                    destination = self.root / "output" / f"oriented_{orientation}.png"
                    write_png(read_image(source), destination, source)

                    with Image.open(destination) as output:
                        self.assertEqual(output.size, expected.size)
                        self.assertEqual(output.tobytes(), expected.tobytes())
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_oriented_tiff_is_not_transposed_twice(self):
        source = self.root / "oriented.tif"
        self._oriented_source(source, 6, "TIFF")
        with Image.open(source) as opened:
            # Pillow applies the tag itself while loading a TIFF.
            expected = opened.convert("RGB")

        tensor = read_image(source)

        self.assertEqual(expected.size, (20, 40))
        write_png(tensor, self.destination, source)
        with Image.open(self.destination) as output:
            self.assertEqual(output.size, expected.size)
            self.assertEqual(output.tobytes(), expected.tobytes())

    def test_mpo_jpeg_uses_oriented_first_frame_and_preserves_source(self):
        source = self.root / "drone.JPG"
        first = Image.new("RGB", (40, 20), (200, 10, 20))
        first.paste((10, 20, 200), (0, 0, 20, 20))
        thumbnail = Image.new("RGB", (10, 5), (10, 200, 20))
        exif = Image.Exif()
        exif[274] = 6
        first.save(source, format="MPO", save_all=True, append_images=[thumbnail], exif=exif)
        original = source.read_bytes()
        with Image.open(source) as opened:
            self.assertEqual(opened.format, "MPO")
            self.assertEqual(opened.n_frames, 2)
            expected = ImageOps.exif_transpose(opened).convert("RGB")
        self.assertEqual(expected.size, (20, 40))

        write_png(read_image(source), self.destination, source)

        with Image.open(self.destination) as output:
            self.assertEqual(output.size, expected.size)
            self.assertEqual(output.tobytes(), expected.tobytes())
        self.assertEqual(source.read_bytes(), original)

    def test_multi_frame_images_stay_rejected(self):
        first = Image.new("RGB", (4, 2), (10, 20, 30))
        second = Image.new("RGB", (4, 2), (40, 50, 60))
        pages = self.root / "pages.tif"
        first.save(pages, save_all=True, append_images=[second])
        animated = self.root / "animated.png"
        first.save(animated, save_all=True, append_images=[second])
        for source in (pages, animated):
            with self.subTest(source=source.name):
                with self.assertRaisesRegex(ValueError, "Only single-frame images are supported"):
                    read_image(source)

    def test_clamp_and_round_before_png_encoding(self):
        Image.new("RGB", (1, 1)).save(self.source)
        tensor = torch.tensor([[[[-0.5]], [[0.5]], [[1.5]]]])
        write_png(tensor, self.destination, self.source)
        with Image.open(self.destination) as output:
            self.assertEqual(output.getpixel((0, 0)), (0, 128, 255))

    def test_numbering_skips_files_directories_symlinks_and_reserved_names(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"existing result")
        (self.destination.parent / "source(2).png").mkdir()
        (self.destination.parent / "source(3).png").symlink_to(self.root / "missing.png")
        reserved = set()
        for name in ("source(4).png", "source(5).png"):
            result = unique_output_path(self.destination, reserved)
            self.assertEqual(result.name, name)
            self.assertFalse(result.exists())
        self.assertEqual(self.destination.read_bytes(), b"existing result")

    def test_nonfinite_output_does_not_replace_existing_result(self):
        Image.new("RGB", (1, 1)).save(self.source)
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"previous result")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            write_png(torch.full((1, 3, 1, 1), float("nan")), self.destination, self.source)
        self.assertEqual(self.destination.read_bytes(), b"previous result")

    def test_encoding_failure_keeps_previous_result_and_cleans_temporary_file(self):
        Image.new("RGB", (1, 1)).save(self.source)
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"previous result")
        with patch.object(Image.Image, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                write_png(torch.zeros(1, 3, 1, 1), self.destination, self.source)
        self.assertEqual(self.destination.read_bytes(), b"previous result")
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_source_and_its_file_aliases_cannot_be_overwritten(self):
        Image.new("RGB", (1, 1), (12, 34, 56)).save(self.source)
        original = self.source.read_bytes()
        symbolic = self.root / "symbolic.png"
        symbolic.symlink_to(self.source)
        hard = self.root / "hard.png"
        hard.hardlink_to(self.source)
        for destination in (self.source, symbolic, hard):
            with self.subTest(destination=destination.name):
                with self.assertRaisesRegex(ValueError, "overwrite input"):
                    write_png(torch.zeros(1, 3, 1, 1), destination, self.source)
        self.assertEqual(self.source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
