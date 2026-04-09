#!/usr/bin/env python
"""Simple test script to verify experiments work."""

import sys
import os
from pathlib import Path

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Working with experiments directory...")

# Test if we can import the experiments
BASE_DIR = Path('experiments')
print(f"Base directory: {BASE_DIR}")
print(f"Base directory exists: {BASE_DIR.exists()}")

# Try to run exp4
print("\n" + "="*80)
print("TESTANDO EXPERIMENTO 4 - Detecção de Anomalias")
print("="*80 + "\n")

try:
    from experiments.exp4_anomaly_detection import run_anomaly_detection_pipeline
    print("✅ Import OK")
    result = run_anomaly_detection_pipeline()
    print("\n✅ Experimento 4 executado com sucesso!")
    print(f"   Resultados gerados: {list(result.keys())}\n")
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Erro ao executar: {e}")
    import traceback
    traceback.print_exc()
