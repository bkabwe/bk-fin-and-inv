from __future__ import annotations


def load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "python-dotenv is required to load environment variables from a .env file. "
            "Install dependencies from requirements.txt or requirements-api.txt."
        ) from exc

    load_dotenv()
