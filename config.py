"""API keys loaded from .env.

The numbered keys let parallel extraction runs use separate rate limits;
scripts select one with --key N.
"""

import os

from dotenv import load_dotenv

load_dotenv()

API_KEYS = {
    0: os.getenv("OPENAI_API_KEY"),
    1: os.getenv("OPENAI_API_KEY1"),
    2: os.getenv("OPENAI_API_KEY2"),
    3: os.getenv("OPENAI_API_KEY3"),
    4: os.getenv("OPENAI_API_KEY4"),
    5: os.getenv("OPENAI_API_KEY5"),
    6: os.getenv("OPENAI_API_KEY6"),
}
