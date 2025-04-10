import os
from dotenv import load_dotenv

# Try to load from different locations
print("Trying to load from .env in current directory...")
load_dotenv()
print(f"ms_token from current directory: {os.environ.get('ms_token')}")

print("\nTrying to load from tiktok_mcp_service/.env...")
load_dotenv("tiktok_mcp_service/.env")
print(f"ms_token from tiktok_mcp_service/.env: {os.environ.get('ms_token')}")

# Print environment variables
print("\nAll environment variables:")
for key, value in os.environ.items():
    if 'token' in key.lower() or 'tiktok' in key.lower():
        print(f"{key}: {value[:10]}..." if value else f"{key}: {value}") 