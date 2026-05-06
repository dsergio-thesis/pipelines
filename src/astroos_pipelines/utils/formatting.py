
from typing import Iterable, Tuple, Any

def ascii_kv_table(
    rows: Iterable[Tuple[str, Any]],
    title: str) -> str:
    rows = [(k, "" if v is None else str(v)) for k, v in rows]

    key_w = max(len(k) for k, _ in rows)
    val_w = max(len(v) for _, v in rows)

    sep = f"+-{'-' * key_w}-+-{'-' * val_w}-+"

    lines = [
        sep,
        f"| {'Setting'.center(key_w)} | {'Value'.center(val_w)} |",
        sep,
    ]

    for k, v in rows:
        lines.append(
            f"| {k.ljust(key_w)} | {v.ljust(val_w)} |"
        )

    lines.append(sep)

    if title:
        lines.append(title + "\n")

    return "\n".join(lines)

def coord_table(
        coords: Iterable[Tuple[str, str, str, float, bool]],
        title: str) -> str:

    rows = [(k, x, y, f"{z:.3f}", w) for k, x, y, z, w in coords]

    key_w = max(max(len(k) + 3 for k, _, _, _, _ in rows), len("key"))
    x_w = max(max(len(x) for _, x, _, _, _ in rows), len("RA_DEC"))
    y_w = max(max(len(y) for _, _, y, _, _ in rows), len("RA_DEC_format"))
    z_w = max(max(len(z) for _, _, _, z, _ in rows), len("radius_arcmiin"))

    sep = f"+-{'-' * key_w}-+-{'-' * x_w}-+-{'-' * y_w}-+-{'-' * z_w}-+"

    lines = [
        sep,
        f"| {'key'.center(key_w)} | {'RA_DEC'.center(x_w)} | {'RA_DEC_format'.center(y_w)} | {'radius_arcmiin'.center(z_w)} |",
        sep,
    ]

    for k, x, y, z, w in rows:
        if w:
            k = f" * {k}"
        else:
            k = f"   {k}"
        lines.append(
            f"| {k.ljust(key_w)} | {x.rjust(x_w)} | {y.rjust(y_w)} | {z.rjust(z_w)} |"
        )

    lines.append(sep)

    if title:
        lines.append(title + "\n")

    return "\n".join(lines)
