"""
Defines the logging settings for the package.
"""

import logging


logging.getLogger("fontTools.subset").setLevel(logging.WARN)


logger = logging.getLogger("metatopia")
logger.setLevel(logging.INFO)

# Create a handler
handler = logging.StreamHandler()

# Create a formatter
formatter = logging.Formatter(
    "[%(levelname)s %(asctime)s] %(name)s: %(message)s",
    datefmt="%d-%m %H:%M:%S",
)  # noqa

# Add the formatter to the handler
handler.setFormatter(formatter)

# Check if the logger already has handlers. If the logger doesn't have any
# handlers, add the new handler and set propagate to False to prevent the
# log messages from being passed to the root logger and possibly being
# duplicated.

if not logger.handlers:
    logger.addHandler(handler)
    logger.propagate = False
