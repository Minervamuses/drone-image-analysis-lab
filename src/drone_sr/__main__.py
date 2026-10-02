"""Process a folder through selected, ordered Spandrel stages."""

import argparse
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Process images into PNG; --sr/--deblur appearance determines stage order.", allow_abbrev=False)
    parser.add_argument("--sr", dest="stages", action="append_const", const="sr", help="enable SR at this position")
    parser.add_argument("--deblur", dest="stages", action="append_const", const="deblur", help="enable deblur at this position")
    parser.add_argument("--sr-model", "--model", dest="sr_model", type=Path, help="SR checkpoint; required with --sr")
    parser.add_argument("--deblur-model", type=Path, help="deblur checkpoint; required with --deblur")
    parser.add_argument("--input", type=Path, default=Path("input"), help="Input folder (default: input/)")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output folder (default: output/)")
    args = parser.parse_args()
    if not args.stages:
        parser.error("at least one mode flag is required: --sr or --deblur")
    if len(args.stages) != len(set(args.stages)):
        parser.error("mode flags must not be repeated")
    for role in args.stages:
        if getattr(args, f"{role}_model") is None:
            parser.error(f"required: --{role}-model when --{role} is enabled")
    print("Drone Image Super-Resolution")
    input_directory = args.input
    output_directory = args.output
    try:
        if not input_directory.is_dir():
            raise ValueError(f"Input directory not found or not a directory: {input_directory}/")
        if input_directory.resolve() == output_directory.resolve():
            raise ValueError("Input and output directories must be different")
        if (output_directory.exists() or output_directory.is_symlink()) and not output_directory.is_dir():
            raise ValueError(f"Output path is not a directory: {output_directory}")
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images = sorted(
            (path for path in input_directory.iterdir() if path.is_file() and path.suffix.lower() in extensions),
            key=lambda path: (path.name.casefold(), path.name),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error))
        return 1
    if not images:
        print(f"No supported images found in {input_directory}/")
        return 0

    destinations = [output_directory / f"{source.stem}.png" for source in images]
    counts = Counter(destination.name for destination in destinations)
    errors = {}
    # Check all inputs before any output can replace a path or file alias.
    for source, destination in zip(images, destinations):
        try:
            if counts[destination.name] > 1:
                raise ValueError(f"Multiple input files map to output: {destination.name}")
            if any(
                destination.resolve() == original.resolve()
                or (destination.exists() and destination.samefile(original))
                for original in images
            ):
                raise ValueError(f"Refusing to overwrite input image through output: {destination}")
        except (OSError, RuntimeError, ValueError) as error:
            errors[source] = str(error)

    for source, destination in zip(images, destinations):
        for role in args.stages:
            checkpoint = getattr(args, f"{role}_model")
            try:
                if destination.resolve() == checkpoint.resolve() or (
                    destination.exists() and checkpoint.exists() and destination.samefile(checkpoint)
                ):
                    errors[source] = f"Refusing to overwrite checkpoint: {checkpoint}"
            except (OSError, RuntimeError) as error:
                errors[source] = str(error)

    from .image_io import read_image, write_png
    from .inference import load_model, run_stages

    try:
        stages = [(role, load_model(getattr(args, f"{role}_model"), role=role)) for role in args.stages]
    except Exception as error:
        print(str(error))
        return 1
    descriptor = stages[0][1]
    device_label = str(descriptor.device)
    if descriptor.device.type == "cuda":
        from torch.cuda import get_device_name

        device_label += f" ({get_device_name(descriptor.device)})"
    print(f"Device: {device_label}")
    print(f"Mode: {' -> '.join(args.stages)}")
    for role in args.stages:
        print(f"Model: {getattr(args, f'{role}_model')}")
    print(f"Images: {len(images)}")
    processed = failed = 0
    for index, (source, destination) in enumerate(zip(images, destinations), 1):
        result = None
        try:
            if source in errors:
                raise ValueError(errors[source])
            result = run_stages(read_image(source), stages)
            write_png(result, destination, source)
        except Exception as error:
            print(f"[{index}/{len(images)}] {source.name} — Failed: {error}")
            failed += 1
        else:
            print(f"[{index}/{len(images)}] {source.name}")
            processed += 1
        finally:
            # Release the previous image before starting the next GPU forward.
            del result
    print("Finished")
    print(f"Mode: {' -> '.join(args.stages)}")
    for role in args.stages:
        print(f"Model: {getattr(args, f'{role}_model')}")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Output: {output_directory}/")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
