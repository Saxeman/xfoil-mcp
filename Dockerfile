FROM --platform=linux/amd64 ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    xfoil \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv 

COPY . .
# can use below syntax to include optional additional libraries
RUN pip install -e ".[dev]"

CMD ["python", "-m", "xfoil_mcp.server"]
