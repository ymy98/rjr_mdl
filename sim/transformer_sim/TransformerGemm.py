from SystolicArray import *
from tools.ParseFunction import *
from functools import reduce

class TransformerGemm(Systolic_Array):

    def __init__(self, name='trans', shape=[16,16], stationary_type=OutputStationary, pipe_col=True, pipe_row=True, bit_width=32, weight_width=8, activation_width=8, frequency=1.0):
        super().__init__(name=name, shape=shape,\
                         stationary_type=stationary_type, \
                         pipe_col=pipe_col, \
                         pipe_row=pipe_row, \
                         bit_width=bit_width, \
                         weight_width=weight_width, \
                         activation_width=activation_width, \
                         frequency=frequency)
        
        self.model = {}
        self.model_config = {}
        
        
    def load_model(self, model):
        if not isinstance(model, dict): raise Exception('model must be a dict type.') 
        self.model = model 

    def show_model(self):
        for key, value in self.model.items():
            print('layer name: %s  input shape: [%s, %s], hidden shape: [%s, %s]'% (key, value[0], value[1], value[1], value[2]))


    def collect_cycles(self):
        self.parse_model_sequence()
        utilization_list=[]
        layer_cycle_list=[]
        for layer, layer_config in self.model_config.items():
            layer_total_cycles = layer_config['sequence_length']
            layer_cycle_list.append(layer_total_cycles)

            total_calculate = layer_total_cycles*self.array_height*self.array_width
            used_calculate  = reduce(lambda x,y: x*y, layer_config['layer_shape'])
            compute_unit_utilization = used_calculate/total_calculate
            utilization_list.append(compute_unit_utilization)

            print(f'Collect Cycles| layer: {layer}  layer cycles: {layer_total_cycles}  compute unit utilization: {round(compute_unit_utilization,3)}')
        
        total_cycles = reduce(lambda x,y: x+y, layer_cycle_list)
        compute_utilization = reduce(lambda x,y: x+y, utilization_list)/len(utilization_list)
        print(f'Total Collect Cycles| total cycles: {total_cycles}  average compute unit utilization: {round(compute_utilization,3)}')

    # def collect_bandwidth(self):
    #     [weight_buf_bw, activation_bw, output_bw] = self.collect_wire_width()
    #     self.parse_model_sequence()

    #     # min_col = min(element.index[2] for element in self.array.values())
    #     # array_col_edge = sorted([element for element in self.array.values() if element.index[2] == min_col], key=lambda x: x.index[1])

    #     # min_row = min(element.index[1] for element in self.array.values())
    #     # array_row_edge = sorted([element for element in self.array.values() if element.index[1] == min_row], key=lambda x: x.index[2])

    #     for layer, layer_config in self.model_config.items(): 
    #         #TODO peak bw, avg bw


    def report_res(self):
        print("################## Result: Systolic Array ####################")
        self.collect_buf()
        self.collect_bandwidth()
        print("###################### Result: Model #########################")
        self.collect_cycles()


    def parse_model_sequence(self, parse_function=BasicParse):
        for key, value in self.model.items():
            use_shape, sequence_length, layer_shape = parse_function(self, key, value)
            self.model_config[key] = {'use_shape':use_shape, 'sequence_length':sequence_length, 'layer_shape':layer_shape}



if __name__=='__main__':
    sa_0 = Systolic_Array(name='sa_0', shape=[8,16], \
                        core_num=1, stationary_type=OutputStationary, \
                        bit_width=32, weight_width=8, activation_width=8)

    sa_0.generate()
    # sa_0._show()
    sa_0.report_res(sequence_length=100, use_shape=[8,7])

    input_seq   = 100
    hidden_size_encoder = 192
    hidden_size_decoder = 768

    model_transformer = {
        'test_layer':                   [16, 12, 15],
        'key'   :                       [input_seq, hidden_size_encoder, hidden_size_encoder],
        'query' :                       [input_seq, hidden_size_encoder, hidden_size_encoder],
        'value' :                       [input_seq, hidden_size_encoder, hidden_size_encoder],
        'Matmul(kxq)':                  [input_seq, hidden_size_encoder, input_seq],
        'Matmul(pxv)':                  [input_seq, input_seq, hidden_size_encoder],
        'Matmul(Linear projection)':    [input_seq, hidden_size_encoder, hidden_size_encoder],
        'hidden_0':                     [input_seq, hidden_size_encoder, hidden_size_decoder],
        'hidden_1':                     [input_seq, hidden_size_decoder, hidden_size_encoder]
    }


    transim_0 = TransformerGemm(name='trans', shape=[16,16], \
                                stationary_type=OutputStationary, \
                                pipe_col=True, pipe_row=True, \
                                bit_width=32, weight_width=8, activation_width=8)
    
    transim_0.generate()
    transim_0.load_model(model_transformer)
    transim_0.show_model()
    # transim_0.report_res()
    transim_0.collect_cycles()
