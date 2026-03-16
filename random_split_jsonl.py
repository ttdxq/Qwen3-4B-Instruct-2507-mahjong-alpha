import argparse
import array
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


def _build_line_offset_index(path: str, *, skip_empty_lines: bool) -> array.array:
    offsets = array.array("Q")
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if skip_empty_lines and not line.strip():
                continue
            offsets.append(pos)
    return offsets


def _write_one_part(
    *,
    input_file: str,
    output_file: str,
    offsets: array.array,
) -> int:
    written = 0
    with open(input_file, "rb") as f_in, open(output_file, "wb") as f_out:
        for off in offsets:
            f_in.seek(off)
            line = f_in.readline()
            if not line:
                continue
            f_out.write(line)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly split a JSONL/NDJSON file into multiple files with a fixed number of lines per file. "
            "The last file contains all remaining lines (< lines_per_file)."
        )
    )
    parser.add_argument("input_file", type=str, help="Path to input .jsonl")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_jsonl",
        help="Output directory (default: output_jsonl)",
    )
    parser.add_argument(
        "--lines_per_file",
        type=int,
        required=True,
        help="Number of lines per output file",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Output filename prefix (default: input filename stem)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Thread workers for writing output files (default: 1)",
    )
    parser.add_argument(
        "--keep_empty_lines",
        action="store_true",
        help="Keep empty/whitespace-only lines (default: skip)",
    )

    args = parser.parse_args()

    if args.lines_per_file <= 0:
        raise SystemExit("--lines_per_file must be > 0")

    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    if not os.path.isfile(args.input_file):
        raise SystemExit(f"Input file not found: {args.input_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    prefix = args.prefix
    if not prefix:
        base = os.path.basename(args.input_file)
        prefix = os.path.splitext(base)[0]

    print("[1/3] Building line offset index...")
    offsets = _build_line_offset_index(
        args.input_file, skip_empty_lines=(not args.keep_empty_lines)
    )
    total = len(offsets)
    if total == 0:
        raise SystemExit("No lines found (after filtering empty lines)")
    print(f"  Indexed lines: {total:,}")

    print("[2/3] Shuffling offsets...")
    rng = random.Random(args.seed)
    rng.shuffle(offsets)

    full_files = total // args.lines_per_file
    remainder = total % args.lines_per_file
    out_files = full_files + (1 if remainder else 0)
    print(
        f"  Plan: {out_files} file(s) (full={full_files}, remainder={remainder}), lines_per_file={args.lines_per_file}"
    )

    print("[3/3] Writing output files...")
    tasks = []
    for i in range(out_files):
        start = i * args.lines_per_file
        end = min(start + args.lines_per_file, total)
        part_offsets = offsets[start:end]
        out_name = os.path.join(args.output_dir, f"{prefix}_{i:05d}.jsonl")
        tasks.append((out_name, part_offsets))

    try:
        from tqdm import tqdm

        use_tqdm = True
    except Exception:
        tqdm = None
        use_tqdm = False

    if args.workers == 1:
        iterable = tasks
        if use_tqdm:
            iterable = tqdm(iterable, total=len(tasks), unit="file")
        for out_name, part_offsets in iterable:
            _write_one_part(
                input_file=args.input_file, output_file=out_name, offsets=part_offsets
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    _write_one_part,
                    input_file=args.input_file,
                    output_file=out_name,
                    offsets=part_offsets,
                ): out_name
                for out_name, part_offsets in tasks
            }
            iterable = as_completed(futures)
            if use_tqdm:
                iterable = tqdm(iterable, total=len(futures), unit="file")
            for fut in iterable:
                fut.result()

    print(f"Done. Output dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
