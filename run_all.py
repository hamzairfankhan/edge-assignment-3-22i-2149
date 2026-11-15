"""Reproduce all experiments in order. No manual model editing is needed."""
import subprocess,sys
from pathlib import Path
if __name__=='__main__':
    root=Path(__file__).resolve().parent
    for script in ['train_baseline.py','ptq.py','qat.py','prune_unstructured.py','prune_structured.py','benchmark.py','export_onnx.py','verify.py']:
        print('\nRUN',script,flush=True)
        subprocess.run([sys.executable,str(root/script)],cwd=root,check=True)
