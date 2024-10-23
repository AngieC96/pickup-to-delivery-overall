from abc import ABC, abstractmethod

# Estimator will be an abstract class, and then we will define our different estimators like the baseline model or the linear model as subclasses of Estimator.
class Estimator(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass