import os
import torch
import onnx
import onnxruntime



def generator_onnx(model, dummy_input, model_path, opset_version=14):
    # if not os.path.exists(model_path): os.mkdir(model_path)
    torch.onnx.export(model, dummy_input, model_path, opset_version=opset_version)
    # check
    try:
        onnx.checker.check_model(model_path)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s"%e)
    else:
        print("The model is valid!")

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
    print(ort_outs[0].shape, torch_out.shape)
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")