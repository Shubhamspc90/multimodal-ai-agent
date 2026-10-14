
import logging


def configure_logging(log_level: str) -> None:
    """Configure application-wide logging."""

    resolved_level = getattr(logging, log_level.upper(), None)

    if not isinstance(resolved_level, int):
        raise TypeError(f"Invalid logging level: {log_level}")

    logging.basicConfig(
        level=resolved_level,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )
