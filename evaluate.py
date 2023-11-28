import torch
import onnx
import onnxruntime
import transformers
import torchvision
from thop import profile
import numpy as np

shape_list = [
    torch.rand(1,3,224,224),
    torch.rand(1,3,1280,720),
    torch.rand(1,3,1920,1080),
]

transformer_model_dict = {

    f'detr-resnet-50' :               transformers.DetrModel.from_pretrained("facebook/detr-resnet-50"),
    f'vit_mae_base'   :               transformers.ViTMAEModel.from_pretrained("facebook/vit-mae-base"),
    f'swin-tiny-patch4-window7':      transformers.SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224"),
    f'swin-small-patch4-window7':     transformers.SwinModel.from_pretrained("microsoft/swin-small-patch4-window7-224"),
    f'swin-base-patch4-window7':      transformers.SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224"),
    f'deit-base-distilled-patch16':   transformers.DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224"),
    f'deit-small-distilled-patch16':  transformers.DeiTModel.from_pretrained("facebook/deit-small-distilled-patch16-224"),
    f'deit-tiny-distilled-patch16':   transformers.DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224"),

    
}

cnn_model_dict = {
    f'mobilenet_v2'       :           torchvision.models.mobilenet_v2(pretrained=True),
    f'mobilenet_v3_large' :           torchvision.models.mobilenet_v3_large(pretrained=True) ,
    f'mobilenet_v2_small' :           torchvision.models.mobilenet_v3_small(pretrained=True) ,
    f'resnet_18' :                torchvision.models.resnet18(pretrained=True) ,
    f'resnet_50' :                torchvision.models.resnet50(pretrained=True) ,
    f'yolov5n':                   torch.hub.load("ultralytics/yolov5", "yolov5n"),
    f'yolov5s':                   torch.hub.load("ultralytics/yolov5", "yolov5s"),
    f'yolov5m':                   torch.hub.load("ultralytics/yolov5", "yolov5m"),
    f'yolov5l':                   torch.hub.load("ultralytics/yolov5", "yolov5l"),
    f'resnext50_32x4d' :          torchvision.models.resnext50_32x4d(pretrained=True) ,
}

f = open('res.csv','w')

def write(string):
    f.write(f'{string}\n')
    print(string)

def evaluate_model(name, model, resolution):
    if 'yolo' in name:     resolution = torch.rand(1,3,int(np.ceil(resolution.shape[2]/32))*32, int(np.ceil(resolution.shape[3]/32)*32))
    flops, params = profile(model, inputs=(resolution, ))
    write(f'{name}_{resolution.shape[2]}x{resolution.shape[3]},{resolution.shape[2]}x{resolution.shape[3]},{flops/1e12},{params/1e6}')

    # pretrained_path = f'{name}_{resolution.shape[2]}x{resolution.shape[3]}.onnx'
    # model_path = pretrained_path.split('/')[-1]
    # torch.onnx.export(model, resolution, model_path, opset_version=14)

    # # check
    # try:                                        onnx.checker.check_model(model_path)
    # except onnx.checker.ValidationError as e:   print("The model is invalid: %s"%e)
    # else:                                       print("The model is valid!")

    # # inference
    # ort_session = onnxruntime.InferenceSession(model_path)

    # # 将张量转化为ndarray格式
    # def to_numpy(tensor):
    #     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # # 构建输入的字典和计算输出结果
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(resolution)}
    # ort_outs = ort_session.run(None, ort_inputs)

    # # 比较使用PyTorch和ONNX Runtime得出的精度
    # torch_out = model(resolution)
    # print(ort_outs[0].shape, torch_out[0].shape)
    # # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")


for name,model in cnn_model_dict.items():
    for shape in shape_list:
        evaluate_model(name, model, shape)

for name,model in transformer_model_dict.items():
    evaluate_model(name, model, shape_list[0])

f.close()

# flops, params = profile(model, inputs=(torch.rand(1,3,224,224 ),))
# print(flops/float(1e12), params/float(1e6))
# print(flops,params)
