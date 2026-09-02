import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)