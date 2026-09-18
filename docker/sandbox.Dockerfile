FROM python:3.12-slim

# Only what a campaign is allowed to import. No xfoil, no harness, no fastmcp.
RUN pip install --no-cache-dir "pydantic>=2" numpy scipy

WORKDIR /app
# Copy only the two modules the sandbox needs. wrapper.py, harness.py, and
# server.py are deliberately absent, so a campaign cannot import them even
# though they exist in the repo.
COPY xfoil_mcp/__init__.py xfoil_mcp/schema.py xfoil_mcp/sandbox_entry.py xfoil_mcp/
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1

# Run as nobody. The host also passes --user; belt and braces.
USER 65534:65534

CMD ["python", "-m", "xfoil_mcp.sandbox_entry"]

