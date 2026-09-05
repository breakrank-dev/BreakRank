import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)