# -*- coding: utf-8 -*-

"""Top-level package for vivarium-ReaDDy."""

__author__ = "Blair Lyons"
__email__ = "blair208@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.0"


def get_module_version():
    return __version__


from .processes.readdy_process import ReaddyProcess  # noqa: F401
from vivarium.core.registry import emitter_registry  # noqa: F401
from .processes.simularium_emitter import SimulariumEmitter  # noqa: F401

emitter_registry.register("simularium", SimulariumEmitter)
