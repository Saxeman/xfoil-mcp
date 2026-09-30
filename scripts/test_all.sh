#!/bin/sh
# Every test, once: host-side tests, then the XFOIL tests in the worker container.
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -m pytest tests/ -q
docker run --rm -v "$(pwd)":/app xfoil-worker sh -c "pip install -q pytest && pytest tests/test_wrapper.py tests/test_worker.py -q"
