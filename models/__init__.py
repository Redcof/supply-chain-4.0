import logging
from abc import abstractmethod, ABC

logger = logging.getLogger(__file__)


class BaseModel(ABC):

    @staticmethod
    @abstractmethod
    def fit(x, y):
        ...

    def __init__(self, *args):
        ...