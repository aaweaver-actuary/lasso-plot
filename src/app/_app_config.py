from sys import argv
from dash import Dash
from app.element_ids import ElementIds
from app.data import Data
from app.config import cache


# Initialize the input data
data = Data()

# Initialize the element IDs as blank unless the --eids flag is passed with a json file
__HAS_EIDS_FLAG = "--eids" in argv
__HAS_PASSED_EIDS_FILE = __HAS_EIDS_FLAG and (argv.index("--eids") + 1 < len(argv))
__NEXT_ARG_IS_JSON_FILE = (
    __HAS_EIDS_FLAG
    and (argv.index("--eids") + 1 < len(argv) - 1)
    and argv[argv.index("--eids") + 1].endswith(".json")
)
if __HAS_PASSED_EIDS_FILE and __NEXT_ARG_IS_JSON_FILE:
    try:
        eids = ElementIds.from_json(argv[argv.index("--eids") + 1])
    except Exception as e:
        print(f"Error loading the element IDs: {e}")  # noqa: T201
        eids = ElementIds()
        eids.to_json()
elif __HAS_PASSED_EIDS_FILE and not __NEXT_ARG_IS_JSON_FILE:
    # Try to read in the default file
    try:
        eids = ElementIds.from_json()
    except Exception as e:
        print(f"Error loading the element IDs: {e}")  # noqa: T201
        eids = ElementIds()
        eids.to_json()
else:
    eids = ElementIds()
    eids.to_json()

# Initialize the Dash app
app = Dash(__name__)


__all__ = ["app", "data", "eids", "cache"]
