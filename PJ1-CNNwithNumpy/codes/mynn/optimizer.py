from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                if layer.weight_decay:
                        layer.W *= (1 - self.init_lr * layer.weight_decay_lambda)

                layer.W -= self.init_lr * layer.grads['W']
                layer.b -= self.init_lr * layer.grads['b']





class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu=mu
        self.velocities = {}

        for layer in self.model.layers:
            if layer.optimizable:
                self.velocities[layer] = {
                    'W': np.zeros_like(layer.W),
                    'b': np.zeros_like(layer.b)
                }

    """公式：(W+)=W-lr*grad+mu*(W-(W-))
    
    转换：设置V=W-(W-)
            V+=(W+)-W=W-lr*grad+mu*(W-(W-))-W=mu*V-lr*grad
            W+=W+(V+)
    """
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                # 当前速度
                v_W = self.velocities[layer]['W']
                v_b = self.velocities[layer]['b']
                # v = β * v - lr * grad
                v_W = self.mu * v_W - self.init_lr * layer.grads['W']
                v_b = self.mu * v_b - self.init_lr * layer.grads['b']
                # w = w + v
                layer.W += v_W
                layer.b += v_b
                self.velocities[layer]['W'] = v_W
                self.velocities[layer]['b'] = v_b

