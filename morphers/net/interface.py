import abc

from torch import nn


class Net(nn.Module):
    @abc.abstractmethod
    def get_input_shape(self) -> tuple[int]:
        # TODO - change this to input & output transform functions
        # This can contain l2 normalizations, reshaping to sketch form, permutations etc.
        pass
