import os
import torch
from models.SingleCNN import SingleCNNModel
from tools.ONNXGenerator import generator_onnx


root_path = 'onnx_repo'
if not os.path.exists(root_path): os.mkdir(root_path)

for i in [32]:
    for j in range(1):
        for k in [1,3]:
            input_channel   = i
            output_channel  = j+1
            kennel_size     = k

            model_path = f'{root_path}/single_cnn_{input_channel}_{output_channel}_{kennel_size}.onnx'
            dummy_input = torch.randn(1,input_channel,224,224)

            model = SingleCNNModel(input_channel, output_channel, kennel_size)
            generator_onnx(model, dummy_input, model_path, opset_version=14)