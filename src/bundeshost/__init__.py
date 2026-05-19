"""bundeshost package.

We load the project's .env file at package import time so that environment
variables (e.g. MLFLOW_TRACKING_URI, POSTGRES_*) are available to every
module without each one having to call load_dotenv() manually.

This is a deliberate side-effect: load_dotenv() does nothing if no .env
file is found, so it is safe in environments that pass env vars directly
(e.g. Docker, CI).
"""

from dotenv import load_dotenv

load_dotenv()
