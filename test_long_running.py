#!/usr/bin/env python3
"""Test script that runs for a configurable duration."""

import sys
import time

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 15

print(f"Starting long-running script (duration: {duration}s)...")
sys.stdout.flush()

for i in range(duration):
    time.sleep(1)
    print(f"Progress: {i + 1}/{duration}")
    sys.stdout.flush()

print("Long-running script finished!")
