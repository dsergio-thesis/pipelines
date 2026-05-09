
from typing import Iterable, Tuple, Any

from typing import Any, Iterable, Tuple
import textwrap

def ascii_config_table(
    rows: Iterable[Tuple[str, str, str, Any]],
    title: str = "",
    max_key_w: int = 24,
    max_arg_w: int = 28,
    max_env_w: int = 28,
    max_val_w: int = 40,
) -> str:
    rows = [(k, arg, env, "" if v is None else str(v)) for k, arg, env, v in rows]

    def col_width(values, max_w):
        return min(max(len(x) for x in values), max_w)

    key_w = col_width([k for k, _, _, _ in rows] + ["Setting"], max_key_w)
    arg_w = col_width([arg for _, arg, _, _ in rows] + ["Arg"], max_arg_w)
    env_w = col_width([env for _, _, env, _ in rows] + ["Env"], max_env_w)
    val_w = col_width([v for _, _, _, v in rows] + ["Value"], max_val_w)

    def wrap_cell(text: str, width: int) -> list[str]:
        return textwrap.wrap(
            text,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]

    sep = f"+-{'-' * key_w}-+-{'-' * arg_w}-+-{'-' * env_w}-+-{'-' * val_w}-+"

    lines = []
    if title:
        lines.append(title)

    lines += [
        sep,
        f"| {'Setting'.center(key_w)} | {'Arg'.center(arg_w)} | {'Env'.center(env_w)} | {'Value'.center(val_w)} |",
        sep,
    ]

    for k, arg, env, v in rows:
        cells = [
            wrap_cell(k, key_w),
            wrap_cell(arg, arg_w),
            wrap_cell(env, env_w),
            wrap_cell(v, val_w),
        ]

        row_h = max(len(c) for c in cells)

        for i in range(row_h):
            lines.append(
                "| "
                + (cells[0][i] if i < len(cells[0]) else "").ljust(key_w)
                + " | "
                + (cells[1][i] if i < len(cells[1]) else "").ljust(arg_w)
                + " | "
                + (cells[2][i] if i < len(cells[2]) else "").ljust(env_w)
                + " | "
                + (cells[3][i] if i < len(cells[3]) else "").ljust(val_w)
                + " |"
            )

    lines.append(sep)

    return "\n".join(lines)
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
