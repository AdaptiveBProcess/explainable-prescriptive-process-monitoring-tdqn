"""Deprecated: use xppm.serve.schemas instead."""

import warnings

warnings.warn(
    "xppm.serving.schemas is deprecated and will be removed in a future version. "
    "Use xppm.serve.schemas instead.",
    DeprecationWarning,
    stacklevel=2,
)

from xppm.serve.schemas import (  # noqa: F401, E402
    CaseFeatures,
    DecisionRequest,
    DecisionResponse,
    PolicyVersions,
)
