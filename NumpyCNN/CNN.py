import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    # Compute the dimensions of the CONV output volume using the formula given above.
    n_H = int((n_H_prev + 2*pad - f) / stride) + 1
    n_W = int((n_W_prev + 2*pad - f) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    # padding
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = h * stride
                    horiz_end = h * stride + f
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[..., c], b[..., c])

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    assert mode in ('max', 'average')

    f = hparameters['f']
    stride = hparameters['stride']
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_pre = A_prev[i]
        for h in range(n_W):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                    a_pre_slice = a_pre[vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes.
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_pre_slice)
                    if mode == 'average':
                        A[i, h, w, c] = np.average(a_pre_slice)

    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    Arguments:
    x -- Array of shape (f, f)
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """

    mask = x == np.max(x)

    return mask


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    # Retrieve dimensions from shape
    (n_H, n_W) = shape
    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)

    # Create a matrix where every entry is the "average" value
    a = np.ones(shape) * average

    return a


def pool_backward(dA, cache, mode="max"):
    """
    Implements the backward pass of the pooling layer
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the corners of the current "slice"
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        #  Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "average":
                        # Get the value a from dA (≈1 line)
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da.
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)

    assert (dA_prev.shape == A_prev.shape)

    return dA_prev


if __name__ == "__main__":
    # test padding function
    # np.random.seed(1)
    # x = np.random.randn(4, 3, 3, 2)
    # x_pad = zero_pad(x, 2)
    # print("x.shape =", x.shape)
    # print("x_pad.shape =", x_pad.shape)
    # print("x[1, 1] =", x[1, 1])
    # print("x_pad[1, 1] =", x_pad[1, 1])
    #
    # fig, axarr = plt.subplots(1, 2)
    # axarr[0].set_title('x')
    # axarr[0].imshow(x[0, :, :, 0])
    # axarr[1].set_title('x_pad')
    # axarr[1].imshow(x_pad[0, :, :, 0])
    # plt.show()

    # test conv_single_step
    # np.random.seed(1)
    # a_slice_prev = np.random.randn(4, 4, 3)
    # W = np.random.randn(4, 4, 3)
    # b = np.random.randn(1, 1, 1)
    #
    # Z = conv_single_step(a_slice_prev, W, b)
    # print("Z =", Z)

    # test conv_forward
    # np.random.seed(1)
    # A_prev = np.random.randn(10, 4, 4, 3)
    # W = np.random.randn(2, 2, 3, 8)
    # b = np.random.randn(1, 1, 1, 8)
    # hparameters = {"pad": 2,
    #                "stride": 1}
    #
    # Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    # print("Z's mean =", np.mean(Z))
    # print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])

    # test pool_forward
    # np.random.seed(1)
    # A_prev = np.random.randn(2, 4, 4, 3)
    # hparameters = {"stride": 1, "f": 4}
    #
    # A, cache = pool_forward(A_prev, hparameters)
    # print("mode = max")
    # print("A =", A)
    # print()
    # A, cache = pool_forward(A_prev, hparameters, mode="average")
    # print("mode = average")
    # print("A =", A)

    # test conv_backward
    # np.random.seed(1)
    # dA, dW, db = conv_backward(Z, cache_conv)
    # print("dA_mean =", np.mean(dA))
    # print("dW_mean =", np.mean(dW))
    # print("db_mean =", np.mean(db))

    # test pool_backward
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride": 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode="max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()
    dA_prev = pool_backward(dA, cache, mode="average")
    print("mode = average")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])