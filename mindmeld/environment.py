import json
import platform
import subprocess
from pathlib import Path
import torch
from .download import ROOT

def main():
    free, total = torch.cuda.mem_get_info()
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(),
            "gpu_free_bytes": free, "gpu_total_bytes": total,
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            "system_ram": subprocess.check_output(["free", "-b"], text=True)}
    (ROOT / "reports").mkdir(exist_ok=True)
    text = json.dumps(info, indent=2)
    (ROOT / "reports/environment.txt").write_text(text)
    print(text)

if __name__ == "__main__":
    main()
