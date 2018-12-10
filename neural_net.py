import conv_layer
import pool
import activation
import utils

import numpy as np

inputs = np.random.random((3, 28, 28))
end_points = {}
NUM_CLASSES = 3

##### ConvLayer1 #####
#layer1_kernels = np.array([128, 3, 3, 3])
layer1_kernels = np.random.random((32, 3, 3, 3))
layer1_stride = 1
layer1_padding = 2
layer1 = ConvLayer(1, inputs, layer1_kernels, layer1_stride, layer1_padding)
layer1_out = layer1.forward_pass()
end_points['convlayer1'] = layer1_out
print("convlayer1 output ready")

##### Act1 #####
act1_inputs = layer1_out
act1 = Activation(act1_inputs, 0)
act1_outs = act1.forward_pass()
end_points['act1'] = act1_outs

##### ConvLayer2 #####
layer2_inputs = act1_outs
#layer2_kernels = np.array([64, 128, 3, 3])
layer2_kernels = np.random.random((64, 32, 3, 3))
layer2_stride = 1
layer2_padding = 1
layer2 = ConvLayer(2, layer2_inputs, layer2_kernels, layer2_stride, layer2_padding)
layer2_out = layer2.forward_pass()
end_points['convlayer2'] = layer2_out

##### Act2 #####
act2_inputs = layer2_out
act2 = Activation(act2_inputs, 0)
act2_outs = act2.forward_pass()
end_points['act2'] = act2_outs

##### FC #####
layer3_inputs = act2_outs
#layer3_weights = np.array([64, NUM_CLASSES])
layer3_weights = np.random.rand(64, NUM_CLASSES)
layer3_bias = np.random.rand(NUM_CLASSES)
#layer3 = fully_connected.FClayer(3, layer3_inputs, NUM_CLASSES)
#layer3_out = layer3.forward_pass()
layer3_out = fullyconnected(layer3_inputs, layer3_weights, )
end_points['FC'] = layer3_out

##### final Activation #####
act3_inputs = layer3_outs
act3 = Activation(act3_inputs, 1)
act3_outs = act3.forward_pass()
end_points['act3'] = act3_outs

print(end_points['act3'])
