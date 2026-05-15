"""
client.py — Safe Supabase Connection Layer
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Singleton instance
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        supabase = None
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY is missing from environment variables. Database operations will fail safely.")

def get_supabase() -> Optional[Client]:
    """Safe getter for the Supabase client."""
    if supabase is None:
        logger.warning("Attempted to get Supabase client, but it is not initialized.")
    return supabase
