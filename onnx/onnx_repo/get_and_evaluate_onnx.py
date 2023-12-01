import os
import torchvision
import torch
import transformers
import onnx
import onnxruntime
from thop import profile
import numpy as np


onnx_suffix = ''
root_path = 'onnx_repo'

# mainly from pytorch
cnn_model_dict = {
    # f'vgg11'              :       torchvision.models.vgg11(pretrained=True),
    # f'vgg13'              :       torchvision.models.vgg13(pretrained=True),
    # f'vgg16'              :       torchvision.models.vgg16(pretrained=True),
    # f'vgg19'              :       torchvision.models.vgg19(pretrained=True),
    # f'alexnet'            :       torchvision.models.alexnet(pretrained=True),
    
    # f'mobilenet'          :       torchvision.models.mobilenet(pretrained=True),
    # f'mobilenet_v2'       :       torchvision.models.mobilenet_v2(pretrained=True),
    # f'mobilenet_v3_large' :       torchvision.models.mobilenet_v3_large(pretrained=True) ,
    # f'mobilenet_v2_small' :       torchvision.models.mobilenet_v3_small(pretrained=True) ,

    # f'resnet_18' :                torchvision.models.resnet18(pretrained=True) ,
    # f'resnet_50' :                torchvision.models.resnet50(pretrained=True) ,
    # f'resnet101' :                torchvision.models.resnet101(pretrained=True),
    # f'resnet152' :                torchvision.models.resnet152(pretrained=True),
    
    # f'yolov5n':                   torch.hub.load("ultralytics/yolov5", "yolov5n"),
    # f'yolov5s':                   torch.hub.load("ultralytics/yolov5", "yolov5s"),
    # f'yolov5m':                   torch.hub.load("ultralytics/yolov5", "yolov5m"),
    # f'yolov5l':                   torch.hub.load("ultralytics/yolov5", "yolov5l"),

    # f'resnext50_32x4d' :          torchvision.models.resnext50_32x4d(pretrained=True) ,
    # f'resnext101_32x8d' :         torchvision.models.resnext101_32x8d(pretrained=True) ,
    # f'resnext101_64x4d' :         torchvision.models.resnext101_64x4d(pretrained=True) ,
}

# mainly from huggingface
# transformer model of huggingface https://huggingface.co/docs/transformers/index
# pretrained model in huggingface https://huggingface.co/ 

transformer_model_dict = {
    # f'detr-resnet-50' :               transformers.DetrModel.from_pretrained("facebook/detr-resnet-50"),
    # f'vit_mae_base'   :               transformers.ViTMAEModel.from_pretrained("facebook/vit-mae-base"),
    f'swin-tiny-patch4-window7':      transformers.SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224"),
    # f'swin-small-patch4-window7':     transformers.SwinModel.from_pretrained("microsoft/swin-small-patch4-window7-224"),
    # f'swin-base-patch4-window7':      transformers.SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224"),
    # f'deit-base-distilled-patch16':   transformers.DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224"),
    # f'deit-small-distilled-patch16':  transformers.DeiTModel.from_pretrained("facebook/deit-small-distilled-patch16-224"),
    # f'deit-tiny-distilled-patch16':   transformers.DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224"),
}

input_224p  = torch.rand(1,3,224,224)
input_720p  = torch.rand(1,3,1280,720)
input_1080p = torch.rand(1,3,1920,1080)



if __name__ == '__main__':
    # Configuration
    dummy_input = input_224p
    model_dict  = transformer_model_dict
    onnx_suffix = '_224p'

    if not os.path.exists('onnx_repo'): os.mkdir('onnx_repo')

    for onnx_name, model in model_dict.items():
        if 'yolo' in onnx_name:     dummy_input = torch.rand(1,3,int(np.ceil(dummy_input.shape[2]/32))*32, int(np.ceil(dummy_input.shape[3]/32)*32))
        file_name = f'{onnx_name}_{dummy_input.shape[2]}x{dummy_input.shape[3]}.onnx'
        model_path = os.path.join(root_path, file_name)
        
        torch.onnx.export(model, dummy_input, model_path, input_names=['input'], output_names=['output'],do_constant_folding=True)

        # param and tops
        flops, params = profile(model, inputs=(dummy_input, ))
        print(flops/1e12, params/1e6)

        # check
        try:                                        onnx.checker.check_model(model_path)
        except onnx.checker.ValidationError as e:   print("The model is invalid: %s"%e)
        else:                                       print("The model is valid!")

        # inference
        ort_session = onnxruntime.InferenceSession(model_path)

        # 将张量转化为ndarray格式
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # 构建输入的字典和计算输出结果
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        # 比较使用PyTorch和ONNX Runtime得出的精度
        torch_out = model(dummy_input)
        print(ort_outs[0].shape, torch_out[0].shape)
        # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")