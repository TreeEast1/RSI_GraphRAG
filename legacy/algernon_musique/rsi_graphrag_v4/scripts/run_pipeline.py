#!/usr/bin/env python3
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/data2-HDD-SATA-20T/nzq/TAG-experiment/HippoRAG/src")
runpy.run_module("rsi_v4.experiment_runner", run_name="__main__")
