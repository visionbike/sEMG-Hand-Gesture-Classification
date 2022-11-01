from abc import abstractmethod, ABC

__all__ = ['BaseProcessor']


class BaseProcessor(ABC):
    """
    The abstract preprocessor implementation
    """
    def __init__(self):
        self.data = None

    @abstractmethod
    def load_data(self) -> None:
        pass

    @abstractmethod
    def process_data(self) -> None:
        pass

    @abstractmethod
    def split_data(self, mode: str) -> None:
        pass
