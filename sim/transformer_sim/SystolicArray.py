from tools.GlobalValue import *
from NetworkNode import *



class Systolic_Array(object):
    def __init__(self, name='SA_0', shape=[4,4], core_num=1, stationary_type=OutputStationary, pipe_col=True, pipe_row=True, bit_width=16, weight_width=8, activation_width=8, frequency=1.0):
        self.name               = name
        self.core_num           = core_num
        self.array_height       = shape[0]
        self.array_width        = shape[1]
        self.stationary_type    = stationary_type
        self.frequency          = frequency
        self.pipe_col           = pipe_col    # weight direction
        self.pipe_row           = pipe_row    # activation direction
        self.array              = {}
        self.bit_width          = bit_width
        self.weight_width       = weight_width
        self.activation_width   = activation_width
        self.pos                = {}
        self.nx                 = NetworkShow(name)

        if self.stationary_type == OutputStationary:
            self.pipe_col         = True
            self.pipe_row         = True
            # self.bit_width        = 32

    def generate(self, quanti=True, array={}, array_param={}):
        if array!={}:
            self.array              = array
            self.core_num           = array_param['core_num'] if 'core_num' in array_param.keys() else self.core_num
            self.array_height       = array_param['array_height']
            self.array_width        = array_param['array_width']
            self.stationary_type    = array_param['stationary_type'] if 'stationary_type' in array_param.keys() else self.stationary_type
            self.bit_width          = array_param['bit_width'] if 'bit_width' in array_param.keys() else self.bit_width
            self.weight_width       = array_param['weight_width'] if 'weight_width' in array_param.keys() else self.weight_width
            self.activation_width   = array_param['activation_width'] if 'activation_width' in array_param.keys() else self.activation_width

        else:
            for core in range(self.core_num):
                for height in range(self.array_height):
                    for width in range(self.array_width):
                        array_index = [core, height, width]
                        name        = f'pe_{core}_{height}_{width}'
                        if quanti:
                            self.array[name]=PEUnit(name=name, \
                                                    bit_width=self.bit_width, index=array_index, \
                                                    pipe_row=self.pipe_row, pipe_col=self.pipe_col, \
                                                    width_col=self.weight_width, width_row=self.activation_width)
                            if height>0: self.array[name].add_link(Top,self.array[f'pe_{core}_{height-1}_{width}'])
                            if width>0 : self.array[name].add_link(Left,self.array[f'pe_{core}_{height}_{width-1}'])
                        else:
                            raise Exception("should increase this case")
                
        # print('Array: ',self.array)

    def report_res(self, sequence_length=0, use_shape=None):
        print("################## Result: Systolic Array ####################")
        self.collect_buf()
        self.collect_cycles(sequence_length, use_shape)
        self.collect_wire_width()
        print("##############################################################")


    def collect_buf(self):
        buffer = 0
        for element in self.array.values():
            buffer += element.bit_width
        buffer *= self.core_num
        print('Systolic Array Buffer: %s'% size_format(buffer))
        return buffer
    

    def collect_cycles(self, sequence_length, use_shape=None):
        input_height = self.array_height if use_shape==None else use_shape[0]
        weight_width = self.array_width  if use_shape==None else use_shape[1]
        print('input_height: %s  sequence_length: %s  weight_width: %s'% (input_height, sequence_length, weight_width))
        total_cycles = self.collect_start_cycle(use_shape=use_shape)+sequence_length
        print('Input sequence: %s  Total Cycle: %s'% (sequence_length, total_cycles))
        return total_cycles
    

    def collect_wire_width(self):
        if self.stationary_type==WeightStationary:
            weight_buf_bw = self.weight_width*self.array_height*self.array_width*self.core_num
            activation_bw = self.activation_width*self.array_height*self.core_num
            output_bw     = self.bit_width*self.array_height*self.array_width*self.core_num
        elif self.stationary_type==OutputStationary:
            weight_buf_bw = self.weight_width*self.array_width*self.core_num
            activation_bw = self.activation_width*self.array_height*self.core_num
            output_bw     = self.bit_width*(self.array_height if self.array_height<self.array_width else self.array_width)*self.core_num

        else:
            raise Exception()

        print('weight_wire_width: %s  activation_wire_width: %s  output_wire_width:%s'% (size_format(weight_buf_bw), \
                                                                     size_format(activation_bw), \
                                                                     size_format(output_bw)))
        return [weight_buf_bw, activation_bw, output_bw]
    
    
    def collect_start_cycle(self, use_shape=None):
        if self.stationary_type==OutputStationary:
            if use_shape==None:
                weight_depth = self.prepare_weight()
                input_depth  = self.prepare_input()
            else:
                weight_depth = self.prepare_weight(use_shape[0])
                input_depth  = self.prepare_input(use_shape[1])  

            print('Start_cycle: %s  Weight_pipe_depth: %s  Input_pipe_depth: %s'% (weight_depth+input_depth-1, weight_depth, input_depth))
            return weight_depth+input_depth-1
        else:
            raise Exception('need add this case')
        
    
    def prepare_weight(self, input_height=None, use_input_height=False):
        if input_height==None:
            min_col = min(element.index[2] for element in self.array.values())
            array_edge = sorted([element for element in self.array.values() if element.index[2] == min_col], key=lambda x: x.index[1])
            weight_cycle=0
            
            for element in array_edge:
                # print(element.name)
                weight_cycle+=element.pipe_col

            return weight_cycle
        
        else:
            if use_input_height: return input_height
            if input_height > self.array_height: raise Exception()

            min_col = min(element.index[2] for element in self.array.values())
            min_row = min(element.index[1] for element in self.array.values())
            array_edge = sorted([element for element in self.array.values() if (element.index[2]==min_col and element.index[1]<min_row+input_height)], key=lambda x: x.index[1])
            weight_cycle=0
            
            for element in array_edge:
                # print(element.name)
                weight_cycle+=element.pipe_col

            return weight_cycle
        

    def prepare_input(self, weight_width=None, use_weight_width=False):
        if weight_width==None:
            min_row = min(element.index[1] for element in self.array.values())
            array_edge = sorted([element for element in self.array.values() if element.index[1] == min_row], key=lambda x: x.index[2])
            input_cycle=0
            
            for element in array_edge:
                # print(element.name)
                input_cycle+=element.pipe_row

            return input_cycle
        else:
            if use_weight_width: return weight_width
            if weight_width > self.array_width: raise Exception()

            min_row = min(element.index[1] for element in self.array.values())
            min_col = min(element.index[2] for element in self.array.values())
            array_edge = sorted([element for element in self.array.values() if (element.index[1]==min_row and element.index[2]<min_col+weight_width)], key=lambda x: x.index[2])
            input_cycle=0
            
            for element in array_edge:
                input_cycle+=element.pipe_row

            return input_cycle


    def _show(self):
        for node in self.array.values():
            
            self.pos[node] = ((node.col+node.core*self.array_height+2)*10, -node.row*10)
            self.nx.add(node)
            if node.link_dict[Right]!=None: self.nx.link(node,node.link_dict[Right]) 
            if node.link_dict[Down]!=None: self.nx.link(node,node.link_dict[Down]) 
        
        self.nx.pos = self.pos
        self.nx._show(self.array_height, self.array_width)

        


    

############# weight stationary function ###############
# def 



############# output stationary function ###############



if __name__=='__main__':
    # pe_0_0 = PEUnit(name='pe_0_0',index=[0,0,0])
    # pe_0_1 = PEUnit(name='pe_0_1',index=[0,0,1])
    # pe_0_2 = PEUnit(name='pe_0_2',index=[0,0,2])
    # pe_0_3 = PEUnit(name='pe_0_3',index=[0,0,3])
    # pe_0_4 = PEUnit(name='pe_0_4',index=[0,0,4])
    # pe_0_5 = PEUnit(name='pe_0_5',index=[0,0,5])
    # pe_0_0.add_link(Right, pe_0_1)
    # pe_0_1.add_link(Right, pe_0_2)
    # pe_0_2.add_link(Right, pe_0_3)
    # pe_0_3.add_link(Right, pe_0_4)
    # pe_0_4.add_link(Right, pe_0_5)

    # array = {}
    # array_param = {
    #     'core_num' : 1,
    #     'array_height' : 1,
    #     'array_width'  : 6,
    # }

    # array['pe_0_0'] = pe_0_0
    # array['pe_0_1'] = pe_0_1
    # array['pe_0_2'] = pe_0_2
    # array['pe_0_3'] = pe_0_3
    # array['pe_0_4'] = pe_0_4
    # array['pe_0_5'] = pe_0_5

    # sa_0 = Systolic_Array(name='SA_External_input')
    # sa_0.generate(array=array,array_param=array_param)
    # sa_0.report_res(128)
    # sa_0._show()



    sa_0 = Systolic_Array(shape=[4,4])
    sa_0.generate()
    accu_0 = Accumulator(name='acc_0',index=[0,4,0])
    accu_1 = Accumulator(name='acc_1',index=[0,4,1])
    accu_2 = Accumulator(name='acc_2',index=[0,4,2])
    accu_3 = Accumulator(name='acc_3',index=[0,4,3])
    accu_0.add_link(Top,sa_0.array['pe_0_3_0'])
    accu_1.add_link(Top,sa_0.array['pe_0_3_1'])
    accu_2.add_link(Top,sa_0.array['pe_0_3_2'])
    accu_3.add_link(Top,sa_0.array['pe_0_3_3'])
    sa_0.array['acc_0'] = accu_0
    sa_0.array['acc_1'] = accu_1
    sa_0.array['acc_2'] = accu_2
    sa_0.array['acc_3'] = accu_3
    sa_0.report_res(128, use_shape=[3,3])
    sa_0._show()
