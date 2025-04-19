from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma
        self.current_milestone_idx = 0

    def step(self):
        self.step_count += 1
        while (self.current_milestone_idx < len(self.milestones) and
               (self.step_count >= self.milestones[self.current_milestone_idx])):
            self.optimizer.init_lr *= self.gamma

            self.current_milestone_idx += 1



class ExponentialLR(scheduler):
    def __init__(self, optimizer,update_count=2000, gamma=0.999) -> None:

        super().__init__(optimizer)
        self.gamma = gamma
        self.step_count=0
        self.update_count=update_count

    def step(self):
        self.step_count += 1
        if self.step_count >= self.update_count:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0