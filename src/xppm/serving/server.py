"""Deprecated: use xppm.serve.server instead."""

import warnings

warnings.warn(
    "xppm.serving.server is deprecated and will be removed in a future version. "
    "Use xppm.serve.server instead.",
    DeprecationWarning,
    stacklevel=2,
)

from xppm.serve.server import app, load_bundle, set_bundle_dir  # noqa: F401, E402
