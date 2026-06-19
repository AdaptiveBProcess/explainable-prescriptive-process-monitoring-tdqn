"""Deprecated: use xppm.serve.guard instead."""

import warnings

warnings.warn(
    "xppm.serving.guard is deprecated and will be removed in a future version. "
    "Use xppm.serve.guard instead.",
    DeprecationWarning,
    stacklevel=2,
)

from xppm.serve.guard import PolicyGuard  # noqa: F401, E402
