
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
        f"| {'Key'.ljust(key_w)} | {'Val'.ljust(val_w)} |",
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
