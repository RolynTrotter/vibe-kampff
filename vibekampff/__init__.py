"""vibekampff — a Voight-Kampff-style introspection test for language models.

Puts a model inside a fabricated first-person transcript, then reads its
j-space (the Jacobian-lens readout of the mid-layer residual stream) to see
whether the workspace carries content the model never says out loud.

See the tracking issue for the architecture and the decisions behind it.
"""

from vibekampff.models import LENS_REPO, LENS_REVISION, ModelSpec, get_model
from vibekampff.readout import LensReader, Readout, Tokenized

__all__ = [
    "LENS_REPO",
    "LENS_REVISION",
    "LensReader",
    "ModelSpec",
    "Readout",
    "Tokenized",
    "get_model",
]
