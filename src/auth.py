from dotenv import load_dotenv
import requests
import os


# --------------------------------------------------------------------------------------------------
# secrets fetch


load_dotenv()  # reads .env into environment variables

CLIENT_ID     = os.getenv("OSU_CLIENT_ID")
CLIENT_SECRET = os.getenv("OSU_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("OSU_CLIENT_ID and OSU_CLIENT_SECRET must be set in .env")


# --------------------------------------------------------------------------------------------------


def get_token() -> str:

    r = requests.post("https://osu.ppy.sh/oauth/token", 

    json={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "public"
    })
    
    r.raise_for_status()
    return r.json()["access_token"]


# --------------------------------------------------------------------------------------------------
