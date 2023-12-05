import os
import onnx 
import onnxruntime
import numpy as np
import copy
from collections import OrderedDict
from GlobalDict import *



class OnnxParser(object):
    def __init__(self, model_path='', output_path='', single_mode=True):
        self.name           = '' 
        self.model_path     = model_path
        self.output_path    = output_path
        self.single_mode    = single_mode
        self.name_list      = []
        self.model_list     = [] 
        self.model          = None 

        self.Node           = None 
        self.input_name     = None 
        self.output_name    = None
        self.input_info     = None
        self.attr_info      = None
        self.output_info    = None 
        self.weight_info    = None

        self.model_input    = None
        self.ort_outs       = None 
        self.ort_input      = None
        self.init_weight    = {}

        self.onnx_dict      = {}
        self.op_type_list   = []
        self.shape_dict     = {}

        self.onnx_sqlite    = {}
    

    def load_model(self):
        if self.single_mode:
            self.name = os.path.basename(self.model_path)
            self.model = onnx.load(self.model_path)
            self.check_model(self.model_path)
            self.onnx_dict = {self.name:{}}
        else:
            onnx_list = os.listdir(self.model_path)
            for onnx_model in onnx_list:
                self.name_list.append(onnx_model)
                onnx_full_path = os.path.join(self.model_path, onnx_model)
                self.model_list.append(onnx.load(onnx_full_path))
                self.check_model(onnx_full_path)
            self.onnx_dict = {name:{} for name in self.name_list}


    def check_model(self, path):
        print('Step: checkModel: %s'% path)
        try:                                        onnx.checker.check_model(path)
        except onnx.checker.ValidationError as e:   print("The model is invalid: %s"%e)
        else:                                       print("The model is valid!")


    def preParse(self):
        self.getAllOutputShape()
        self.getInitializer()
        self.ort_input = onnxruntime.InferenceSession(self.model.SerializeToString()).get_inputs()
        

    def parseSingleOnnx(self):
        print(f'\nStart Parsing: {os.path.join(self.model_path, self.name)}')
        if self.model == None: self.load_model()
        if self.onnx_dict[self.name]=={} : self.init_onnx_dict()
        self.preParse()
        for node in self.getNodeNameList:
            print(f'############# Parse:{node} #############')
            self.parseNode(node)
            # self.update_onnx_dict()
            self.update_sum_dict()
            update_global_dict(self.Node, self.input_info, self.output_info, self.attr_info, self.weight_info)


    def parseBatchOnnx(self):
        self.load_model()
        for i in range(len(self.model_list)):
            self.model = self.model_list[i]
            self.name  = self.name_list[i] 
            self.parseSingleOnnx()


    def parseNode(self, node):
        self.getNodeAndIOname(node)
        self.getInputTensorValueInfo()
        self.getOutputTensorValueInfo()
        self.getAttrValueInfo(self.Node.name) 
        self.getWeightValueInfo()


    def update_sum_dict(self):
        self.op_type_list.append(self.Node.op_type)
        if self.Node.op_type not in self.shape_dict: self.shape_dict[self.Node.op_type] = {
            'input shape' : [],
            'weight shape': []
        }
        if self.input_info in [(),{}]: pass
        elif len(self.input_info) == 1:
            self.shape_dict[self.Node.op_type]['input shape'].append([i for i in self.input_info.values()])
        elif len(self.input_info) == 2:
            shape_0 = list(self.input_info.values())[0]
            shape_1 = list(self.input_info.values())[1]    
            if len(shape_0) == len(shape_1):
                self.shape_dict[self.Node.op_type]['input shape'].append(shape_0)
                self.shape_dict[self.Node.op_type]['input shape'].append(shape_1)
            else:
                self.shape_dict[self.Node.op_type]['input shape'].append([i for i in self.input_info.values()])
        
        if self.Node.op_type in ['Conv', 'Relu', 'Gemm', 'MaxPool']:
            self.shape_dict[self.Node.op_type]['weight shape'].append([i for i in self.weight_info.values()])  


    def save_csv(self):
        df = pd.DataFrame(columns=['op type', 'freq', 'input shape', 'weight shape'])

        freq_cntv = pd.Series(self.op_type_list).value_counts()
        df['op type']   = freq_cntv.index.tolist()
        df['freq']      = freq_cntv.values.tolist()
        
        i = 0
        for op in df['op type']:
            input_str = self.flatten_serials_to_string(pd.Series(self.shape_dict[op]['input shape']).value_counts())
            weight_str = self.flatten_serials_to_string(pd.Series(self.shape_dict[op]['weight shape']).value_counts())
            df.loc[i, 'input shape']  = input_str
            df.loc[i, 'weight shape'] = weight_str
            i = i + 1

        df.to_csv(os.path.join(self.output_path,'sum.csv'), index=False, sep=',')


    def save_sqlite(self):
        df = pd.DataFrame(columns=['op type', 'freq', 'input shape', 'weight shape'])

        freq_cntv = pd.Series(self.op_type_list).value_counts()
        df['op type']   = freq_cntv.index.tolist()
        df['freq']      = freq_cntv.values.tolist()
        df.to_csv('test.csv', index=False, sep=',')
        
        i = 0
        for op in df['op type']:
            input_str = self.flatten_serials_to_string(pd.Series(self.shape_dict[op]['input shape']).value_counts())
            weight_str = self.flatten_serials_to_string(pd.Series(self.shape_dict[op]['weight shape']).value_counts())
            df.loc[i, 'input shape']  = input_str
            df.loc[i, 'weight shape'] = weight_str
            i = i + 1

        conn = sqlite3.connect(os.path.join(self.output_path, 'sum.db'))
        df.to_sql('onnx_op_summary', conn, if_exists='replace')









    def flatten_serials_to_string(self, serials):
        serials_str = ''
        for index, value in serials.items():
            serials_str = serials_str + f'{index}: {value}\n'
        return serials_str


    def update_onnx_dict(self):
        self.onnx_dict[self.name]['op type'].append(self.Node.op_type)
        if self.Node.op_type not in self.onnx_dict[self.name]: self.onnx_dict[self.name][self.Node.op_type] = {
            'input shape' : [],
            'weight shape': []
        }
        if self.input_info in [(),{}]: pass
        elif len(self.input_info) == 1:
            self.onnx_dict[self.name][self.Node.op_type]['input shape'].append([i for i in self.input_info.values()])
        elif len(self.input_info) == 2:
            shape_0 = list(self.input_info.values())[0]
            shape_1 = list(self.input_info.values())[1]    
            if len(shape_0) == len(shape_1):
                self.onnx_dict[self.name][self.Node.op_type]['input shape'].append(shape_0)
                self.onnx_dict[self.name][self.Node.op_type]['input shape'].append(shape_1)
            else:
                self.onnx_dict[self.name][self.Node.op_type]['input shape'].append([i for i in self.input_info.values()])
        
        if self.Node.op_type in ['Conv', 'Relu', 'Gemm', 'MaxPool']:
            self.onnx_dict[self.name][self.Node.op_type]['weight shape'].append([i for i in self.output_info.values()])


    def init_onnx_dict(self):
        self.onnx_dict[self.name] = {
            'op type'       : [],
            'input shape'   : [],
            'weight shape'  : [],
        }


    def getNodeAndIOname(self, nodename):
        print('Step: getNodeAndIOname')
        for i in range(len(self.model.graph.node)):
            if self.model.graph.node[i].name == nodename:
                self.Node = self.model.graph.node[i]
                self.input_name = self.model.graph.node[i].input
                self.output_name = self.model.graph.node[i].output
    
    def getInputTensorValueInfo(self):
        print('Step: getInputTensorValueInfo')
        in_tvi = {}
        for name in self.input_name:
            for params_input in self.ort_input:
                if params_input.name == name:
                    in_tvi[params_input.name] = tuple(params_input.shape)
            if name in self.ort_outs.keys():
                in_tvi[name] = self.ort_outs[name] 
        self.input_info = in_tvi

    def getOutputTensorValueInfo(self):
        print('Step: getOutputTensorValueInfo')
        out_tvi = {}
        for name in self.output_name:
            out_tvi[name] = self.ort_outs[name] 
        self.output_info = out_tvi


    def getAttrValueInfo(self, node_name):
        print('Step: getAttrValueInfo')
        attr_avi = {}
        for node in self.model.graph.node:
            if node_name == node.name:
                for attr in node.attribute:
                    if hasattr(attr, 'ints') and attr.ints!=[]:
                        attr_avi[attr.name] = [attr.ints]
        self.attr_info = attr_avi
    

    def getWeightValueInfo(self):
        print('Step: getWeightValueInfo')
        weight_vi = {}
        for name in self.input_name:
            if name in self.init_weight.keys():
                weight_vi[name] = self.init_weight[name]
        self.weight_info = weight_vi


    def getAllOutputShape(self):
        print('Step: getAllOutputShape')
        input_shape = onnxruntime.InferenceSession(self.model.SerializeToString()).get_inputs()[0].shape
        input_data = np.random.rand(input_shape[0],input_shape[1],input_shape[2],input_shape[3]).astype(np.float32)
        ori_output = copy.deepcopy(self.model.graph.output)
        for node in self.model.graph.node:
            for output in node.output:
                self.model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        
        ort_session = onnxruntime.InferenceSession(self.model.SerializeToString())
        ort_inputs = {}

        for i, input_ele in enumerate(ort_session.get_inputs()):
            ort_inputs[input_ele.name] = input_data
        outputs = [x.name for x in ort_session.get_outputs()]
        self.ort_outs = ort_session.run(outputs, ort_inputs)
        self.ort_outs = OrderedDict(zip(outputs, [outs.shape for outs in self.ort_outs]))
        del self.model.graph.output[:]
        self.model.graph.output.extend(ori_output)


    def getInitializer(self):
        self.init_weight = {}
        for initializer in self.model.graph.initializer:
            weight = onnx.numpy_helper.to_array(initializer)
            self.init_weight[initializer.name] = weight.shape

    @property
    def getNodeNum(self):
        return len(self.model.graph.node)
    @property
    def getNodeNameList(self):
        NodeNameList = []
        for i in range(self.getNodeNum):
            NodeNameList.append(self.model.graph.node[i].name)
        return NodeNameList

    @property
    def getModelInputInfo(self):
        return self.model.graph.input[0]

    @property
    def getModelOutputInfo(self):
        return self.model.graph.output[0]
    

    
if __name__=='__main__':
    
    onnx_class = OnnxParser(name='test', model_path='bydModel',output_path='.', single_mode=False)
    onnx_class.parseBatchOnnx()
    opt_global_dict()
    write_csv()
    
