from __future__ import annotations

import sys
from pathlib import Path

import h5py


def resolve_inputs() -> tuple[list[Path], Path]:
    """Resolve input files and output path.

    Usage:
        python merge_demos.py
        python merge_demos.py output.hdf5 input1.hdf5 input2.hdf5 ...
    """
    args = sys.argv[1:]

    if not args:
        inputs = sorted(Path("./datasets").glob("areumii_pickplace_v9_*.hdf5"))
        inputs += [
        Path("./datasets/areumii_pickplace_v8.hdf5"),
        ]


        output = Path("./datasets/areumii_pickplace_v9.hdf5")
        return inputs, output

    output = Path(args[0])
    inputs = [Path(p) for p in args[1:]]
    if not inputs:
        inputs = sorted(Path("./datasets").glob("areumii_pickplace_v8_*.hdf5"))
    return inputs, output


def demo_sort_key(name: str):
    parts = name.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        return (0, int(parts[1]), name)
    return (1, name)


def merge_demos(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    if not inputs:
        raise FileNotFoundError("No input demo files were found to merge.")

    with h5py.File(output, "w") as fout:
        data_out = fout.create_group("data")

        demo_idx = 0
        total_samples = 0
        env_args = None

        for file in inputs:
            if not file.exists():
                raise FileNotFoundError(f"Input file not found: {file}")
            if file.resolve() == output.resolve():
                continue

            try:
                with h5py.File(file, "r") as fin:
                    if "data" not in fin:
                        continue

                    data_in = fin["data"]

                    if env_args is None and "env_args" in data_in.attrs:
                        env_args = data_in.attrs["env_args"]

                    for demo_name in sorted(data_in.keys(), key=demo_sort_key):
                        fin.copy(data_in[demo_name], data_out, name=f"demo_{demo_idx}")
                        total_samples += data_out[f"demo_{demo_idx}"].attrs.get("num_samples", 0)
                        demo_idx += 1
            except (OSError, ValueError) as exc:
                print(f"Skipping invalid HDF5 file: {file} ({exc})")
                continue

        data_out.attrs["total"] = total_samples
        if env_args is not None:
            data_out.attrs["env_args"] = env_args

    print(f"Merged {demo_idx} demos into {output}")


if __name__ == "__main__":
    inputs, output = resolve_inputs()
    merge_demos(inputs, output)