import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

def get_engine() -> Engine:
    postgres_uri = os.getenv("POSTGRES_URI")
    if not postgres_uri:
        raise ValueError("POSTGRES_URI not found in environment variables.")
    return create_engine(postgres_uri, pool_pre_ping=True)