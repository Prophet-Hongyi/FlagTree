from triton.flagtree_spec import spec_path

spec_path(__path__)

from . import blackwell
from . import hopper

__all__ = ["blackwell", "hopper"]
