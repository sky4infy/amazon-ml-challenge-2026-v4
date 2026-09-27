"""
Amazon ML Challenge 2026 - Business Entity Resolution Pipeline (V4 Championship Edition).
Master orchestrator pointing to the peak-scoring V4 architecture (0.8380 Public LB).

Usage:
    python src/pipeline.py                     # Full pipeline
    python src/pipeline.py --step blocking      # Re-run blocking only
    python src/pipeline.py --step train         # Re-train models
    python src/pipeline.py --step predict       # Predict with ensemble
    python src/pipeline.py --step validate      # Validate submission
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_v4 import main

if __name__ == '__main__':
    main()
