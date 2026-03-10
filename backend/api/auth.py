"""
auth.py — Authentication stub
Currently a placeholder. Can be extended with JWT tokens later.
"""
from fastapi import Header, HTTPException
from typing import Optional

# For now, no auth is required — this is a research prototype
# To add API key auth in future, set API_KEY in .env and uncomment below

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Optional API key verification.
    Currently disabled — all requests are allowed.
    To enable: set REQUIRE_AUTH=true in .env
    """
    pass  # No auth required for research prototype
