"""Entry point that serves the API on the configured port."""

import uvicorn

from src.config import HOST, PORT


def main() -> None:
    """Start the web server listening on ``HOST`` at ``PORT``."""
    uvicorn.run("src.api.server:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
