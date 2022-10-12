import json
from functools import wraps
from pathlib import Path


def file_cache(filename):
    """Decorator to cache the output of a function to disk."""

    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(directory) / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out

        return decorated

    return decorator
