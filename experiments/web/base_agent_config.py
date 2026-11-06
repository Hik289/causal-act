"""Configuration for benchmark sites; credentials are supplied by the caller."""
import os

DEFAULT_MODEL = os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol")
OPENAI_MODEL = DEFAULT_MODEL
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VWA_SITES = {
    "CLASSIFIEDS": "http://127.0.0.1:9980",
    "SHOPPING": "http://127.0.0.1:7770",
    "REDDIT": "http://127.0.0.1:9999",
    "WIKIPEDIA": "http://127.0.0.1:8888",
    "HOMEPAGE": "http://127.0.0.1:4399",
}
SITE_URLS = VWA_SITES
BASE_MODEL = DEFAULT_MODEL

def inject_env():
    os.environ.setdefault("DATASET", "visualwebarena")
    for key, value in VWA_SITES.items():
        os.environ.setdefault(key, value)
