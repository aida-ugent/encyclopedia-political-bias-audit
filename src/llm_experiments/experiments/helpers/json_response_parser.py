import json
import re

class JsonResponseParser:
    def parse(self, raw: str, default={}) -> dict:
        try:
            cleaned = raw.strip()
            m = re.search(r'\{.*\}', cleaned, flags=re.S)
            cleaned = m.group(0)
            data = json.loads(cleaned)
        except Exception:
            # Last resort – empty result
            print(f"Unable to process json: {raw}")
            data = default
        return data
