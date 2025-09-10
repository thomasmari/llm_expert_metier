
import os
import getpass
from dotenv import load_dotenv

def setup_env_variables(auto:bool=False, verbose:bool=False):
    """Set up environment variable if not already set !"""

    if auto :
        load_dotenv()
        if verbose :
            print("Environment variables loaded from .env file.")

    if not os.environ.get("LANGSMITH_TRACING"):
        os.environ["LANGSMITH_TRACING"] = "true"
    elif verbose:
        print("Environment variable LANGSMITH_TRACING : ✔️")

    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for Langsmith: ")
    elif verbose:
        print("Environment variable LANGSMITH_API_KEY : ✔️")

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    elif verbose:
        print("Environment variable GOOGLE_API_KEY : ✔️")