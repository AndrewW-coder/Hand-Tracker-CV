from abc import ABC, abstractmethod
import numpy as np

class Effect(ABC):
    @abstractmethod
    def apply(self, frame: np.ndarray, hands: list) -> np.ndarray:
        ...