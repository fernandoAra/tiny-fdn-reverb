#!/usr/bin/env python

import json
import os
import tempfile
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("notebook.ipynb")
REPO_ROOT = NOTEBOOK_PATH.parent.parent
REQUIRED_CELL_INDICES = (0, 3)


def notebook_code(path: Path) -> str:
    notebook = json.loads(path.read_text())
    code_parts = []
    for cell_index in REQUIRED_CELL_INDICES:
        cell = notebook["cells"][cell_index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            code_parts.append(source)
    return "\n\n".join(code_parts)


def main() -> None:
    os.chdir(REPO_ROOT)
    mpl_cache = Path(tempfile.gettempdir()) / "tiny-fdn-reverb-mpl-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None

    namespace = {}
    exec(compile(notebook_code(NOTEBOOK_PATH), str(NOTEBOOK_PATH), "exec"), namespace)
    print(f"Executed notebook cells {REQUIRED_CELL_INDICES} from {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
