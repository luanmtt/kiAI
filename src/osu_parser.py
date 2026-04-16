from features import feature_engineering
import os


# --------------------------------------------------------------------------------------------------
# parsing file:


def parse_osu_file(filepath:str) -> dict:
    
    # lendo linha-a-linha cada .osu.

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    sections = {}
    current = None

    for line in content.splitlines():

        line = line.strip()
        if not line or line.startswith("//"):
            continue

        if line.startswith("[") and line.endswith("]"):

            current = line[1:-1]
            sections[current] = []
        
        elif current:
            sections[current].append(line)
    
    return feature_engineering(sections)
        
   
# --------------------------------------------------------------------------------------------------
# teste se necessário


'''
if __name__ == "__main__":

    # quick test on a single file
    import json
    sample = next(
        os.path.join("data/beatmaps", f)
        for f in os.listdir("data/beatmaps")
        if f.endswith(".osu")
    )
    print(f"Parsing: {sample}")
    result = parse_osu_file(sample)
    print(json.dumps(result, indent=2))
'''


# --------------------------------------------------------------------------------------------------
