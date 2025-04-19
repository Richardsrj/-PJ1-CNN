from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    # def train(self):
    #     """设置为训练模式"""
    #     for layer in self.layers:
    #         if isinstance(layer, Dropout):
    #             layer.train()  # 启用 Dropout
    #
    # def eval(self):
    #     """设置为评估模式"""
    #     for layer in self.layers:
    #         if isinstance(layer, Dropout):
    #             layer.eval()  # 禁用 Dropout

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    def __init__(self):
        super().__init__()
        self.layers = [
            conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2),  # 输出 [batch, 6, 28, 28]
            max_pool(), #6*14*14
            conv2D(in_channels=6, out_channels=16, kernel_size=3, padding=0), #16*12*12
            max_pool(),

            Flatten(),

            Dropout(p=0.2),

            Linear(in_dim=576,out_dim=10) , # 输出 [batch, 10]

        ]
        self.params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                self.params.extend(layer.params.values())

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Reshape输入为 [batch, 1, 28, 28]
        if X.ndim == 2:
            X = X.reshape(-1, 1, 28, 28)
        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def train(self):
        """设置为训练模式"""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train()  # 启用 Dropout

    def eval(self):
        """设置为评估模式"""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.eval()  # 禁用 Dropout

    def save_model(self, save_path):
        param_list = []
        # 遍历所有层，仅保存可优化层的参数
        for layer in self.layers:
            if isinstance(layer, (conv2D, Linear)):
                layer_params = {
                    'W': layer.W,
                    'b': layer.b,
                    'weight_decay': layer.weight_decay,
                    'weight_decay_lambda': layer.weight_decay_lambda
                }
                param_list.append(layer_params)
        # 保存到文件
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

    def load_model(self, param_list_path):
        # 加载参数列表
        with open(param_list_path, 'rb') as f:
            param_list = pickle.load(f)
        param_idx = 0
        # 按顺序恢复参数到对应层
        for layer in self.layers:
            if isinstance(layer, (conv2D, Linear)):
                if param_idx >= len(param_list):
                    raise ValueError("参数列表长度与模型层数不匹配")
                params = param_list[param_idx]
                # 检查参数形状是否匹配
                if layer.W.shape != params['W'].shape:
                    raise ValueError(f"权重形状不匹配: 预期 {layer.W.shape}, 实际 {params['W'].shape}")
                if layer.b.shape != params['b'].shape:
                    raise ValueError(f"偏置形状不匹配: 预期 {layer.b.shape}, 实际 {params['b'].shape}")
                # 恢复参数
                layer.W = params['W']
                layer.b = params['b']
                layer.weight_decay = params['weight_decay']
                layer.weight_decay_lambda = params['weight_decay_lambda']
                param_idx += 1

