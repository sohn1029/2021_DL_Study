# libraries
import numpy as np

# Optimizers : SGD, Momentum, AdaGrad, RMSProp, Adam

class SGD():
    """Stochastic Gradient Descent Optimizer"""

    def __init__(self, lr=0.01):
        """
        lr : The learning rate (default : 0.01)
        """
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum():
    """Momentum Optimizer"""

    def __init(self, lr=0.01, a=0.9):
        """
        lr : The learning rate (default : 0.01)
        a : Acceleration (default : 0.9)
        v : Velocity
        """
        self.lr = lr
        self.a = a
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                # init
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.a * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad():
    """AdaGrad Optimizer"""

    def __init__(self, lr=0.01):
        """
        lr : The learning rate (default : 0.01)
        h : A adaptive hyper parameter
        epsilon : A small number to avoid zero-division
        """
        self.lr = lr
        self.h = None
        self.epsilon = 1e-7

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # init
                self.h[key] = np.zeros_like(val)

        for key in params, keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.epsilon)

class RMSProp():
    """RMSProp Optimizer"""

    def __init__(self, lr=0.01, decay_rate=0.99):
        """
        lr : The learning rate (default : 0.01)
        decay_rate : The decay rate (default : 0.99)
        h : A adaptive hyper parameter
        epsilon : A small number to avoid zero-division
        """
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        self.epsilon = 1e-7
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # init
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.epsilon)

class Adam():
    """Adam(Momentum + RMSProp) Optimizer"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """
        lr : The learning rate (default : 0.001)
        beta1 : The exponential decay rate for the 1st moment estimates (default : 0.9)
        beta2 : The exponential decay rate for the 2nd moment estimates (default : 0.999)
        iter : The timestep
        m, v : Momentum vectors
        epsilon : A small number to avoid zero-division
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0 # init
        self.m = None
        self.v = None
        self.epsilon = 1e-7
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                # init
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - np.power(self.beta2, self.iter)) / (1.0 - np.power(self.beta1, self.iter))
        
        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (np.power(grads[key], 2) - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + self.epsilon)
            
            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)