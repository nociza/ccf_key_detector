"""Core package for the realtime tonal-center probability estimator."""

from importlib import metadata


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("ccf-key-detector")
        except metadata.PackageNotFoundError:  # pragma: no cover - during dev
            return "0.0.0"
    raise AttributeError(name)


__all__ = ["__version__"]
