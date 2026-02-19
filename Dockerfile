FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY tiktok_mcp_service/ tiktok_mcp_service/

# Set environment variables
ENV PYTHONPATH=/app

# Run the server
CMD ["python", "-m", "tiktok_mcp_service.main"] 