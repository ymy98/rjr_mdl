from ONNXParser import *


if __name__=='__main__':
    single_onnx = OnnxParser(model_path='onnx_model',output_path='csv', single_mode=False)
    single_onnx.parseBatchOnnx()
    # opt_global_dict()
    # write_csv(single_onnx.output_path)
    single_onnx.save_csv()
    single_onnx.save_sqlite()

   

        