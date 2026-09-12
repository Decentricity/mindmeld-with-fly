"""Record actual Feather schemas before implementing schema-dependent preprocessing."""
import json
from pathlib import Path
import pyarrow.feather as feather
from .download import ROOT, FILES

def main():
    report = {}
    for name in FILES:
        table = feather.read_table(ROOT / "data/raw" / name, memory_map=True)
        info = {"rows": table.num_rows, "schema": str(table.schema), "sample": table.slice(0, 3).to_pylist()}
        report[name] = info
        print(name, json.dumps(info, indent=2, default=str), flush=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/schemas.json").write_text(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
