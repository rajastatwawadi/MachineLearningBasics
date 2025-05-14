import numpy as np

def convolution(image, kernels, stride=1, padding=0, Relu=False):
    num_channels, h, w = image.shape
    num_filters, kernel_depth, kh, kw = kernels.shape  # Ensure depth matches num_channels
    
    if kernel_depth != num_channels:
        raise ValueError(f"Kernel depth {kernel_depth} must match image channels {num_channels}")
    
    image_padded = pad_image(image, padding)
    
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    output = np.zeros((num_filters, out_h, out_w))
    
    for f in range(num_filters):
        kernel = kernels[f]
        for i in range(out_h):
            for j in range(out_w):
                region = image_padded[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
                output[f, i, j] = np.sum(region * kernel)

    if(Relu):
        return relu(output)
    return output


def pooling(image, kernel_size=2, stride=2, padding=0, pool_type="max"):
    if(padding):
        image = pad_image(image, padding)
    num_channels, h, w = image.shape
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    output = np.zeros((num_channels, out_h, out_w))
    
    for k in range(num_channels):
        for i in range(out_h):
            for j in range(out_w):
                region = image[k, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
                if pool_type == "max":
                    output[k, i, j] = np.max(region)
                elif pool_type == "average":
                    output[k, i, j] = np.mean(region)
    
    return output

def apply_kernel(region, kernel):
    return np.sum(region * kernel)

def pad_image(image, padding):
    if padding > 0:
        return np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    return image

def relu(matrix):
    return np.maximum(0, matrix)

def inception_layer(image, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, num_pool_proj):
    
    num_channels, h, w = image.shape  
    
    filters = {
        '1x1': np.random.rand(num_1x1, num_channels, 1, 1),
        '3x3_reduce': np.random.rand(num_3x3_reduce, num_channels, 1, 1),
        '3x3': np.random.rand(num_3x3, num_3x3_reduce, 3, 3),
        '5x5_reduce': np.random.rand(num_5x5_reduce, num_channels, 1, 1),
        '5x5': np.random.rand(num_5x5, num_5x5_reduce, 5, 5),
        '1x1_pool': np.random.rand(num_pool_proj, num_channels, 1, 1)
    }
    
    conv_1x1 = convolution(image, filters['1x1'], stride=1, padding=0)

    conv_3x3_reduce = convolution(image, filters['3x3_reduce'], stride=1, padding=0)
    conv_3x3 = convolution(conv_3x3_reduce, filters['3x3'], stride=1, padding=1)

    conv_5x5_reduce = convolution(image, filters['5x5_reduce'], stride=1, padding=0)
    conv_5x5 = convolution(conv_5x5_reduce, filters['5x5'], stride=1, padding=2)

    pool_proj = pooling(image, kernel_size=3, stride=1, padding=1, pool_type='max')
    pool_proj_conv = convolution(pool_proj, filters['1x1_pool'], stride=1, padding=0)

    output = np.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj_conv], axis=0)
    
    return output


def fully_connected_layer(input_tensor, num_outputs):
    num_inputs = input_tensor.shape[0]
    weights = np.random.randn(num_outputs, num_inputs) * 0.01  # Small random weights
    bias = np.zeros((num_outputs,))
    
    return np.dot(weights, input_tensor) + bias  # Linear transformation


def dropout_layer(input_tensor, dropout_rate=0.4):
    mask = np.random.rand(*input_tensor.shape) > dropout_rate
    return input_tensor * mask  # Apply dropout


def softmax(logits):
    exp_vals = np.exp(logits - np.max(logits))  # Shift for numerical stability
    return exp_vals / np.sum(exp_vals)


if __name__=='__main__':

    #Input
    image = np.random.rand(3, 224, 224)
    print(f"Input layer Dimension: {image.shape}")


    #Conv Layer 1
    Conv1 = convolution(image, np.random.rand(64, 3, 7, 7), stride=2, padding=3,Relu=True)
    print(f"Conv1 Ouput Dimension: {Conv1.shape}")

    MaxPool1=pooling(Conv1, kernel_size=3, stride=2, padding=1, pool_type="max")
    print(f"MaxPool1 Ouput Dimension: {MaxPool1.shape}")
    print()




    # Conv Layer 2a
    Conv2a = convolution(MaxPool1, np.random.rand(64, 64, 1, 1), stride=1, padding=0, Relu=True)
    print(f"Conv2a Output Dimension: {Conv2a.shape}")

    # Conv Layer
    Conv2b = convolution(Conv2a, np.random.rand(192, 64, 3, 3), stride=1, padding=1, Relu=True)
    print(f"Conv2b Output Dimension: {Conv2b.shape}")

    # MaxPool 2
    MaxPool2 = pooling(Conv2b, kernel_size=3, stride=2, padding=1, pool_type="max")
    print(f"MaxPool2 Output Dimension: {MaxPool2.shape}")
    print()




    #Inception layer 3
    Inception3a = inception_layer(MaxPool2, 
                            num_1x1=64, 
                            num_3x3_reduce=32, 
                            num_3x3=128, 
                            num_5x5_reduce=32, 
                            num_5x5=32, 
                            num_pool_proj=32)
    print(f"Inception Layer3a Output Dimension: {Inception3a.shape}")

    #Inception Layer 3b
    Inception3b = inception_layer(Inception3a, 
                            num_1x1=128, 
                            num_3x3_reduce=64, 
                            num_3x3=192, 
                            num_5x5_reduce=64, 
                            num_5x5=96,
                            num_pool_proj=64)
    print(f"Inception Layer3b Output Dimension: {Inception3b.shape}")

    # MaxPool 3
    MaxPool3 = pooling(Inception3b, kernel_size=3, stride=2, padding=1, pool_type="max")
    print(f"MaxPool3 Output Dimension: {MaxPool3.shape}")
    print()




    # Inception Layer 4a
    Inception4a = inception_layer(MaxPool3, 
                            num_1x1=192, 
                            num_3x3_reduce=96, 
                            num_3x3=208, 
                            num_5x5_reduce=16, 
                            num_5x5=48, 
                            num_pool_proj=64)
    print(f"Inception Layer 4a Output Dimension: {Inception4a.shape}")

    # Inception Layer 4b
    Inception4b = inception_layer(Inception4a, 
                            num_1x1=160, 
                            num_3x3_reduce=112, 
                            num_3x3=224, 
                            num_5x5_reduce=24, 
                            num_5x5=64, 
                            num_pool_proj=64)
    print(f"Inception Layer 4b Output Dimension: {Inception4b.shape}")

    # Inception Layer 4c
    Inception4c = inception_layer(Inception4b, 
                            num_1x1=128, 
                            num_3x3_reduce=128, 
                            num_3x3=256, 
                            num_5x5_reduce=24, 
                            num_5x5=64, 
                            num_pool_proj=64)
    print(f"Inception Layer 4c Output Dimension: {Inception4c.shape}")

    # Inception Layer 4d
    Inception4d = inception_layer(Inception4c, 
                            num_1x1=112, 
                            num_3x3_reduce=144, 
                            num_3x3=288, 
                            num_5x5_reduce=32, 
                            num_5x5=64, 
                            num_pool_proj=64)
    print(f"Inception Layer 4d Output Dimension: {Inception4d.shape}")

    # Inception Layer 4e
    Inception4e = inception_layer(Inception4d, 
                            num_1x1=256, 
                            num_3x3_reduce=160, 
                            num_3x3=320, 
                            num_5x5_reduce=32, 
                            num_5x5=128, 
                            num_pool_proj=128)
    print(f"Inception Layer 4e Output Dimension: {Inception4e.shape}")

    # MaxPool 4
    MaxPool4 = pooling(Inception4e, kernel_size=3, stride=2, padding=1, pool_type="max")
    print(f"MaxPool4 Output Dimension: {MaxPool4.shape}")
    print()




    # Inception Layer 5a
    Inception5a = inception_layer(MaxPool4, 
                            num_1x1=256, 
                            num_3x3_reduce=160, 
                            num_3x3=320, 
                            num_5x5_reduce=32, 
                            num_5x5=128, 
                            num_pool_proj=128)
    print(f"Inception Layer 5a Output Dimension: {Inception5a.shape}")

    # Inception Layer 5b
    Inception5b = inception_layer(Inception5a, 
                            num_1x1=384, 
                            num_3x3_reduce=192, 
                            num_3x3=384, 
                            num_5x5_reduce=48, 
                            num_5x5=128, 
                            num_pool_proj=128)
    print(f"Inception Layer 5b Output Dimension: {Inception5b.shape}")
    print()




    # Avg Pool
    AvgPool = pooling(Inception5b, kernel_size=7, stride=1, padding=0, pool_type="max")
    print(f"Average Pool Output Dimension: {AvgPool.shape}")

    #Dropout
    Dropped = dropout_layer(AvgPool.reshape(1024,), dropout_rate=0.4)
    print(f"Dropped Output Shape: {Dropped.shape}")

    # Fully Connected Layer
    FC_output = fully_connected_layer(Dropped, num_outputs=1000)
    print(f"Fully Connected Layer Output Shape: {FC_output.shape}")

    # Softmax layer
    Softmax_output = softmax(FC_output)
    print(f"Softmax Output Shape: {Softmax_output.shape}")
    print()


