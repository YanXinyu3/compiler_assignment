from __future__ import absolute_import as _abs

import tvm
import numpy as np
'''
The names of fucntions, parameters, return values and their orders are essential
to testing, so do not modify them.
The meaning of parameters are put in comments, and the order of return values are
specified in comments by 
'
Returns
-------
    Return val_1 : ...
    Return val_2 : ...
'
Here we also provide an example of conv2d to show how to build functions for the operators:
--------------------------------------------------------------------------
i_shape = (batch_size, image_height, image_width, in_channels)
f_shape = (out_channels, in_channels, kernel_height, kernel_width)
c_shape = (batch_size, image_height - kernel_height + 1, image_width - kernel_width + 1, out_channels)

Image = tvm.placeholder(i_shape, name='Image')
Filter = tvm.placeholder(f_shape, name='Filter')

Conv = conv2d(Image, Filter)
s =  tvm.create_schedule(Conv.op)

ctx = tvm.cpu(0)
f = tvm.build(s, [Image, Filter, Conv], 'llvm')

im = tvm.nd.array(np.random.uniform(size=i_shape).astype(Image.dtype), ctx)
fi = tvm.nd.array(np.random.uniform(size=f_shape).astype(Filter.dtype), ctx)
conv = tvm.nd.array(np.zeros(c_shape, dtype = Conv.dtype), ctx)
f(im, fi, conv)
---------------------------------------------------------------------------
You can use tvm.testing.assert_allclose function to test if the outputs are right
'''

def conv2d(Image, Filter):
    """
    Convolution operator in NHWC layout

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: tvm.tensor.Tensor
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]

    Returns
    -------
    Output: tvm.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, out_channels]
    """
    batch_size, image_height, image_width, in_channels = Image.shape
    out_channels, in_channels, kernel_height, kernel_width = Filter.shape
    out_height = image_height - kernel_height + 1
    out_width = image_width - kernel_width + 1
    rx = tvm.reduce_axis((0, kernel_height), name='rx')
    ry = tvm.reduce_axis((0, kernel_width), name='ry')
    rc = tvm.reduce_axis((0, in_channels), name='rc')

    conv = tvm.compute((batch_size, out_height, out_width, out_channels), lambda n, h, w, o: \
        tvm.sum(Image[n, h + rx, w + ry, rc] * Filter[o, rc, rx, ry], axis=[rx, ry, rc]))
    return conv

def conv2db(Image, Filter, POutput):
    """
    convolution with NHWC layout backward

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Filter: tvm.tensor.Tensor
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, out_height, out_width, out_channels]

    Returns
    -------
    PImage :tvm.tensor.Tensor, gradient of Image
        4-D with shape (Image.shape)
    PFilter:tvm.tensor.Tensor, gradient of Filter
        4-D with shape (Filter.shape)
    """
    batch_size, image_height, image_width, in_channels = Image.shape
    out_channels, in_channels, kernel_height, kernel_width = Filter.shape
    batch_size, out_height, out_width, out_channels = POutput.shape

    #TODO: zero padding
    ZPoutput = tvm.compute((batch_size, out_height + 2, out_width + 2, out_channels), lambda n, i, j, o: \
        tvm.if_then_else(i == 0 or i == out_height + 1 or j==0 or j == out_width + 1, \
             tvm.const(0,POutput.dtype), POutput[n,i-1,j-1,o] ))
    rx0 = tvm.reduce_axis((0,kernel_height))
    ry0 = tvm.reduce_axis((0, kernel_width))
    ro = tvm.reduce_axis((0,out_channels))
    PImage = tvm.compute(Image.shape, lambda n, h, w, c: \
        tvm.sum(Filter[ro, c, kernel_height - rx0 - 1, kernel_width - ry0 - 1] * \
            ZPoutput[n,h+rx0,w+ry0,ro], axis=[rx0,ry0,ro]))

    rn = tvm.reduce_axis((0,batch_size))
    rx = tvm.reduce_axis((0,image_height-kernel_height+1))
    ry = tvm.reduce_axis((0,image_width-kernel_width+1))
    PFilter = tvm.compute(Filter.shape, lambda o, c, h, w: \
        tvm.sum(Image[rn,h+rx,w+ry,c] * POutput[rn,rx,ry,o], axis=[rn,rx,ry]))
    return PImage, PFilter

def relu(Image):
    """
    ReLU operator

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: tvm.tensor.Tensor
        4-D with shape (Image.shape)
    """

    return tvm.compute(Image.shape,lambda n,h,w,c: tvm.max(Image(n,h,w,c), tvm.const(0,Image.dtype)))

def relub(Image, POutput):
    """
    ReLU operator backward

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    PImage: tvm.tensor.Tensor
        4-D with shape (Image.shape)
    """
    return tvm.compute(Image.shape, lambda n, i, j, c: tvm.if_then_else(\
        Image(n,i,j,c) > 0, POutput(n,i,j,c), tvm.const(0,Image.dtype)))

def pooling(Image):
    """
    2*2 max pooling adapted to the revised poolingb2

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: tvm.tensor.Tensor
        4-D with shape [batch_size, out_height, out_width, in_channels]
    """
    batch_size, image_height, image_width, in_channels = Image.shape
    out_height = image_height / 2
    out_width = image_width / 2
    rx = tvm.reduce_axis((0, 2), name='rx')
    ry = tvm.reduce_axis((0,2),name='ry')
    return tvm.compute((batch_size, out_height, out_width, in_channels), lambda n, i, j, c: \
        tvm.max( Image[n, i*2+rx, j*2+ry,c],axis = [rx, ry]))

def poolingb(Image, Index, POutput):
    """
    reverse 2*2 max pooling revised

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    Index : tvm.tensor.Tensor, specify where Output[i,j,k,l] is from, this follows the convention of 
        Numpy and PyTorch. You will need this tensor to compute the gradient.
        ------------------------------------------
        For example, if Image is of shape [1, 4, 4, 1] (batch 1 and channel 1), then the slice
        Image[0, :, :, 0] is

        [[0.7243, 0.3236, 0.0124, 0.4314],
        [0.4104, 0.3997, 0.4534, 0.1791],
        [0.0973, 0.2673, 0.6907, 0.9207],
        [0.9268, 0.6590, 0.0312, 0.2364]]

        and Index is of shape [1, 2, 2, 1] and the slice Index[0, :, :, 0] is 

        [[ 0,  6],
        [12, 11]]

        because 0 = 0 * 4 + 0 (0, 0)
                6 = 1 * 4 + 2 (1, 2)
                12= 3 * 4 + 0 (3, 0)
                11= 2 * 4 + 3 (2, 3)
        --------------------------------------------
        4-D with shape [batch_size, out_height, out_width, in_channels]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, out_height, out_width, in_channels]

    Returns
    -------
    PImage: tvm.tensor.Tensor, gradient of Image
        4-D with shape (Image.shape)
    """
    batch_size, image_height, image_width, in_channels = Image.shape
    batch_size, out_height, out_width, in_channels = Index.shape
    poolingb = tvm.compute(Image.shape, lambda n, i, j, c: \
        tvm.if_then_else(i * image_width + j == Index[n, i // 2, j // 2, c], \
             POutput[n, i // 2, j // 2, c], tvm.const(0, Image.dtype)))
    return poolingb

def flatten(Image):
    """
    flatten layer

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]

    Returns
    -------
    Output: tvm.tensor.Tensor
        2-D with shape [batch_size, out_size]
    """
    batch_size, image_height, image_width, in_channels = Image.shape
    flatten = tvm.compute((batch_size, image_height*image_width*in_channels), lambda n, i: \
        Image[n, i // (image_width * in_channels), \
            (i//in_channels) % image_width, i % in_channels])
    return flatten

def flattenb(Image, POutput):
    """
    reverse flatten

    Parameters
    ----------
    Image : tvm.tensor.Tensor
        4-D with shape [batch_size, image_height, image_width, in_channels]
    POutput:tvm.tensor.Tensor, gradient of Output
        4-D with shape [batch_size, out_size]

    Returns
    -------
    PImage: tvm.tensor.Tensor
        4-D with shape (Image.shape)
    """
    batch_size, image_height, image_width, in_channels = Image.shape
    PImage = tvm.compute(Image.shape, lambda n, i, j, c: POutput[n, (i * image_width + j) *  \
                          in_channels + c])
    return PImage
    

def fullyconn(Input, Weight):
    """
    Fully Connected Layer

    Parameters
    ----------
    Input : tvm.tensor.Tensor
        2-D with shape [batch_size, input_size]
    Weight: tvm.tensor.Tensor
        2-D with shape [input_size, out_size]

    Returns
    -------
    Output: tvm.tensor.Tensor
        2-D with shape [batch_size, out_size]
    """
    batch_size, input_size = Input.shape
    _input_size, out_size = Weight.shape
    k = tvm.reduce_axis((0, input_size), name='k')
    fconn = tvm.compute((batch_size,out_size),lambda n, i: tvm.sum(Input[n,k] * Weight[k,i], axis = k), name='fconn')
    return fconn

def fullyconnb(Input, Weight, POutput):
    """
    reverse fully connected

    Parameters
    ----------
    Input : tvm.tensor.Tensor
        2-D with shape [batch_size, input_size]
    Weight: tvm.tensor.Tensor
        2-D with shape [input_size, out_size]
    POutput:tvm.tensor.Tensor, gradient of Output
        2-D with shape [batch_size, out_size]

    Returns
    -------
    PWeight:tvm.tensor.Tensor, gradient of Weight
        2-D with shape (Weight.shape)
    PInput: tvm.tensor.Tensor, gradient of Input
        2-D with shape (Input.shape)
    """
    batch_size, input_size = Input.shape
    input_size, out_size = Weight.shape
    batch_size, out_size = POutput.shape
    n = tvm.reduce_axis((0,input_size), name='n')
    PWeight = tvm.compute((input_size,out_size),lambda k,i:tvm.sum(Input[n,k] * POutput[n,i], axis=n), name='PWeight')
    i = tvm.reduce_axis((0,out_size), name='i')
    PInput = tvm.compute((batch_size,input_size),lambda n,k:tvm.sum(POutput[n,i] * Weight[k,i], axis=i), name='PInput')
    return PWeight, PInput
