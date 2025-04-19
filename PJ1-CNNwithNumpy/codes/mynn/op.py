from abc import abstractmethod
import numpy as np
from click.core import batch


class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        Z = np.dot(X, self.W) + self.b  # Z=x*W+b
        return Z

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        dZ = grad
        batch_size = dZ.shape[0]
        # print(batch_size)  输出：32
        if not self.weight_decay:
            dW = np.dot(self.input.T, dZ)  # in*out
        else:
            dW = np.dot(self.input.T, dZ) +  self.weight_decay_lambda * self.W
        db = np.sum(dZ, axis=0, keepdims=True)  #sum or mean
        self.grads['W'] = dW
        self.grads['b'] = db/batch_size
        doutput = np.dot(dZ, self.W.T)
        return doutput

    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size, self.kernel_size))
        # [out_channels, in_channels, kernel, kernel]
        self.b = np.zeros((out_channels,))

        self.grads = {'W': None, 'b': None}
        self.input = None  # Record the input for backward process.

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay  # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda  # control the intensity of weight decay

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """

        self.input = X
        batch_size, in_channels, H_in, W_in = X.shape
        k = self.kernel_size

        # padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding,) * 2, (self.padding,) * 2))
        else:
            X_padded = X

        H_out = (H_in + 2 * self.padding - k) // self.stride + 1
        W_out = (W_in + 2 * self.padding - k) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # 计算卷积
        for i in range(H_out):
            h_start = i * self.stride
            h_end = h_start + k
            for j in range(W_out):
                w_start = j * self.stride
                w_end = w_start + k

                window = X_padded[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.tensordot(
                    window, self.W, axes=([1, 2, 3], [1, 2, 3])
                ) + self.b

        return output



    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """

        batch_size, out_channels, H_out, W_out = grads.shape
        k = self.kernel_size

        # 初始化梯度
        dX = np.zeros_like(self.input)
        dfilters = np.zeros_like(self.W)
        dbias = np.zeros_like(self.b)

        # 旋转卷积核（关键步骤）
        # rotated_filters = np.rot90(self.W, 2, axes=(2, 3))
        filters=self.W

        # 处理padding
        if self.padding > 0:
            X_padded = np.pad(self.input,
                              ((0, 0), (0, 0), (self.padding,) * 2, (self.padding,) * 2))
            dX_padded = np.pad(dX,
                               ((0, 0), (0, 0), (self.padding,) * 2, (self.padding,) * 2))
        else:
            X_padded = self.input
            dX_padded = dX

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                w_start = j * self.stride
                window = X_padded[:, :, h_start:h_start + k, w_start:w_start + k]


                # 1. 计算滤波器梯度
                dfilters += np.tensordot(
                    grads[:, :, i, j],  # shape: (batch, out_ch)
                    window,  # shape: (batch, in_ch, k, k)
                    axes=([0], [0])  # 沿着batch维度做点积
                )  # 结果形状: (out_ch, in_ch, k, k)

                # 2. 计算输入梯度
                grad_slice = grads[:, :, i, j][:, :, np.newaxis, np.newaxis,
                             np.newaxis]  # shape: (batch, out_ch, 1, 1, 1)
                dX_padded[:, :, h_start:h_start + k, w_start:w_start + k] += np.sum(
                    grad_slice * filters[np.newaxis, :, :, :, :],  # shape: (1, out_ch, in_ch, k, k)
                    axis=1  # 沿out_ch维度求和
                )  # 结果形状: (batch, in_ch, k, k)

                # 3. 计算偏置梯度
                dbias += np.sum(grads[:, :, i, j], axis=0)  # 向量化计算

        # 去除padding
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # 权重衰减
        if self.weight_decay:
            dfilters +=  self.weight_decay_lambda * self.W

        self.grads['W'] = dfilters/batch_size
        self.grads['b'] = dbias/batch_size
        return dX



    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class Flatten(Layer):
    """将多维输入展平为一维
    CNN--->fc的连接
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """

    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False



    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts

        batch_size = predicts.shape[0]
        log_probs = -np.log(probs[np.arange(batch_size), labels] + 1e-8)
        loss = np.mean(log_probs)

        # 保存变量用于反向传播
        self.probs = probs  # 预测
        self.labels = labels  # 标签（真）

        return loss

    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size = self.labels.shape[0]

        # 计算梯度
        if self.has_softmax:
            grad = self.probs.copy()
            grad[np.arange(batch_size), self.labels] -= 1
        else: #不会用
            grad = np.zeros_like(self.probs)
            grad[np.arange(batch_size), self.labels] = -1.0 / (self.probs[np.arange(batch_size), self.labels] + 1e-8)

        grad /= batch_size  #归一化，否则模型训不动

        self.grads = grad
        if self.model is not None:
            self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self



class max_pool(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size  # 池化窗口大小，例如2x2
        self.stride = stride            # 步长
        self.optimizable = False        # 池化层无可训练参数
        self.mask = None                # 记录最大值位置的掩码

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        前向传播：计算最大池化
        输入形状：[batch_size, channels, height, width]
        输出形状：[batch_size, channels, new_height, new_width]
        """
        self.input = X
        batch_size, channels, height, width = X.shape
        k = self.kernel_size
        stride = self.stride

        # 计算输出尺寸
        new_height = (height - k) // stride + 1
        new_width = (width - k) // stride + 1

        # 初始化输出和掩码
        output = np.zeros((batch_size, channels, new_height, new_width))
        self.mask = np.zeros_like(X)  # 与输入同形的掩码，用于反向传播

        for b in range(batch_size):
            for c in range(channels):
                for i in range(new_height):
                    for j in range(new_width):
                        h_start = i * stride
                        h_end = h_start + k
                        w_start = j * stride
                        w_end = w_start + k

                
                        window = X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        output[b, c, i, j] = max_val


                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.mask[b, c, h_start + max_idx[0], w_start + max_idx[1]] = 1

        return output

    def backward(self, dout):
        """
        反向传播：将梯度传递到前向传播中最大值的位置
        输入梯度形状：[batch_size, channels, new_height, new_width]
        输出梯度形状：[batch_size, channels, height, width]
        """
        dx = np.zeros_like(self.input)  # 初始化梯度张量

        batch_size, channels, new_height, new_width = dout.shape
        k = self.kernel_size
        stride = self.stride

        # 遍历每个池化窗口
        for b in range(batch_size):
            for c in range(channels):
                for i in range(new_height):
                    for j in range(new_width):
                        h_start = i * stride
                        h_end = h_start + k
                        w_start = j * stride
                        w_end = w_start + k

                        # 将梯度分配到前向传播中最大值的位置
                        dx[b, c, h_start:h_end, w_start:w_end] += \
                            dout[b, c, i, j] * self.mask[b, c, h_start:h_end, w_start:w_end]

        return dx


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """

    def __init__(self, model, lambda_=1e-5):
        super().__init__()
        self.model = model
        self.lambda_ = lambda_  # 正则化系数λ
        self.optimizable = False

    def __call__(self, inputs=None, labels=None):
        return self.forward()

    def forward(self):
        """
        计算模型的L2正则化损失，并启用各层的权重衰减。
        """
        l2_loss = 0.0
        for layer in self.model.layers:
            if hasattr(layer, 'weight_decay') and hasattr(layer, 'params') and 'W' in layer.params:
                # 启用该层的权重衰减，并设置λ值
                layer.weight_decay = True
                layer.weight_decay_lambda = self.lambda_
                # 累加权重平方和
                l2_loss += np.sum(layer.params['W'] ** 2)
        return 0.5 * self.lambda_ * l2_loss


import numpy as np

class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        self.p = p                  # 神经元保留概率
        self.is_training = True
        self.mask = None
        self.optimizable=False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.is_training:
            self.mask = (np.random.rand(*X.shape) < self.p) / self.p
            return X * self.mask
        else:
            return X

    def backward(self, dout):
        if self.is_training:
            return dout * self.mask
        else:
            return dout

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition