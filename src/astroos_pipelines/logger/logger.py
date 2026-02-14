import logging
import sys


# ----------------------------
# CONFIG — change these values
# ----------------------------

# DEBUG / INFO / WARNING / ERROR
LOG_LEVEL = logging.WARNING
LOG_DEST = "file"                 # "stdout" | "file" | "both"
LOG_FILE = "log/debug.log"

# ----------------------------

FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # let handlers filter
    root.handlers.clear()

    formatter = logging.Formatter(FMT)

    if LOG_DEST in ("stdout", "both"):
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(LOG_LEVEL)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if LOG_DEST in ("file", "both"):
        fh = logging.FileHandler(LOG_FILE, mode="a")
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(formatter)
        root.addHandler(fh)
