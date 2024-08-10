from .automorpher import (
    AutomorpherNet,
    WeightedAutomorpherNet,
)
from .depthwise_automorpher import DepthwiseAutomorpherNet

__all__ = [AutomorpherNet.__name__, WeightedAutomorpherNet.__name__, DepthwiseAutomorpherNet.__name__]
