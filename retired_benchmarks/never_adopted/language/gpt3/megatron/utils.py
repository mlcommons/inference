import json
import io

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        with open(f, mode=mode) as f:
            return json.load(f)
    else:
        return json.load(f)