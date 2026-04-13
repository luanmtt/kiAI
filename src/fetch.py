import requests
import time

BASE_URL = "https://osu.ppy.sh/api/v2"


# --------------------------------------------------------------------------------------------------


def get_beatmap_batch(ids: list, token: str) -> list:

    headers = { "Authorization": f"Bearer {token}" }
    params = [("ids[]", bid) for bid in ids[:50]] # máximo de 50 requests por chamada.
    r = requests.get(f"{BASE_URL}/beatmaps", headers=headers, params=params)

    r.raise_for_status()
    return r.json().get("beatmaps", [])


# --------------------------------------------------------------------------------------------------

def get_beatmaps_bulk(ids: list, token:str, delay: float = 0.5) -> list:

    results = []
    chunks = [ids [i : i+50] for i in range(0, (len(ids)), 50)] # grupos de 50.
    
    for i, chunk in enumerate(chunks):

        print(f"Fetching batch {i+1}/{len(chunks)}...")
        results.extend(get_beatmap_batch(chunk, token))
        time.sleep(delay)

    return results


# --------------------------------------------------------------------------------------------------

def get_ids_from_collector(collection_id:int) -> list[int]:


    ids = []
    cursor = 0
    while True:
        r = requests.get(
            f"https://osucollector.com/api/collections/{collection_id}/beatmapsv2",
            params={"cursor": cursor, "perPage": 100}
        )
        r.raise_for_status()
        data = r.json()

        beatmaps = data.get("beatmaps", [])
        ids += [b["id"] for b in beatmaps]

        if not data.get("hasMore"):
            break
        
        cursor = data["nextPageCursor"]

    return ids


# --------------------------------------------------------------------------------------------------
