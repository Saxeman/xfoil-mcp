FROM python:3.12-slim

# The thermal solver and nothing else: no XFOIL, no harness.
RUN pip install --no-cache-dir "pydantic>=2" numpy scipy scikit-fem

WORKDIR /app
COPY xfoil_mcp/__init__.py xfoil_mcp/schema.py xfoil_mcp/coupling.py \
     xfoil_mcp/thermal.py xfoil_mcp/thermal_stage.py xfoil_mcp/thermal_entry.py \
     xfoil_mcp/
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1

USER 65534:65534

CMD ["python", "-m", "xfoil_mcp.thermal_entry"]
