from abc import ABC, abstractmethod

class Retrieve(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup_retrieve(self):
        pass

    @abstractmethod
    def search(self):
        pass
