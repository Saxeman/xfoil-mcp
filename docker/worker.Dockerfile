FROM --platform=linux/amd64 ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    xfoil python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv

COPY pyproject.toml .
COPY xfoil_mcp/ xfoil_mcp/
RUN pip install --no-cache-dir -e .

# Worker has no business on the network; the host passes --network none too,
# but a container that assumes it is offline is easier to reason about.
ENV XFOIL_IMAGE="xfoil-worker"

CMD ["python", "-m", "xfoil_mcp.worker"]

