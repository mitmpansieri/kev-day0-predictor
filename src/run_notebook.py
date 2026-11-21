"""Run a notebook's code cells in-process and save outputs back into the .ipynb.

Why: the sandbox blocks local ports, so a Jupyter kernel cannot start. This runner executes each code cell in
one shared namespace, captures printed text, DataFrame displays and matplotlib figures, and writes them into the
notebook so it opens in VS Code with results. (In VS Code / Colab the notebook also runs normally.)

Usage: .venv/bin/python src/run_notebook.py notebooks/03_exploration.ipynb
"""
from __future__ import annotations

import base64
import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402


def _figure_outputs() -> list:
    outs = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        outs.append(nbformat.v4.new_output("display_data",
                                           data={"image/png": base64.b64encode(buf.getvalue()).decode()}))
    plt.close("all")
    return outs


def run(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    os.chdir(path.parent)                       # notebooks run from their own folder, like Jupyter
    ns: dict = {"__name__": "__main__"}
    displayed: list = []
    state = {"buf": io.StringIO()}

    def flush_text():
        text = state["buf"].getvalue()
        if text:
            displayed.append(nbformat.v4.new_output("stream", name="stdout", text=text))
            state["buf"].seek(0); state["buf"].truncate()

    def display(obj):                           # minimal IPython display(): DataFrames as text + html
        flush_text()                            # keep printed text and tables in their original order
        data = {"text/plain": repr(obj)}
        if hasattr(obj, "to_html"):
            data["text/html"] = obj.to_html()
        displayed.append(nbformat.v4.new_output("display_data", data=data))
    ns["display"] = display

    count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        cell.outputs, cell.execution_count, displayed[:] = [], count, []
        buf = state["buf"] = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(cell.source, f"<cell {count}>", "exec"), ns)
        except Exception:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue()))
            cell.outputs.append(nbformat.v4.new_output("error", ename="Error", evalue="", traceback=[traceback.format_exc()]))
            nbformat.write(nb, path.name)
            print(f"FAILED in cell {count}:\n{traceback.format_exc()}", file=sys.stderr)
            sys.exit(1)
        flush_text()
        cell.outputs += displayed + _figure_outputs()
    nbformat.write(nb, path.name)
    print(f"OK: ran {count} code cells -> {path}")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
