"""Model factory for diffusion runners.

This small helper lets you swap in alternative model implementations
(such as discrete diffusion models) without touching the runner logic.
Set ``config.model.name`` to select a model type.
"""

from typing import Any

from .diffusion import Model


def create_model(config: Any):
    """Create a diffusion model instance based on ``config.model.name``.

    Parameters
    ----------
    config:
        Namespace-like object containing a ``model`` attribute. When
        ``config.model.name`` is ``"discrete"``, the factory will build
        :class:`DiscreteDiffusionModel` defined in ``discrete_diffusion.py``.
        Any other value defaults to the standard continuous diffusion
        :class:`Model` used across the original codebase.
    """

    model_name = getattr(getattr(config, "model", object()), "name", "continuous")

    if model_name == "discrete":
        from .discrete_diffusion import DiscreteDiffusionModel

        return DiscreteDiffusionModel(config)

    return Model(config)
