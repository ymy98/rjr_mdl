import os
import networkx as nx
import matplotlib.pyplot as plt
from tools.Logger import NetworkLogger
from tools.GlobalValue import *


class Node(object):
    color = 'gray'
    def __init__(self,name='', operation_type=Null, bit_width=0, index=[0,0,0], pipe_col=False, pipe_row=False, width_col=0, width_row=0):
        self._name              = name
        self.layer              = 0
        self.operation_type     = operation_type
        self.bit_width          = bit_width
        self.index              = index
        self.core               = index[0]
        self.row                = index[1]
        self.col                = index[2]
        self.pipe_col           = pipe_col
        self.pipe_row           = pipe_row
        self.width_row          = width_row
        self.width_col          = width_col

        self.link_dict      = {
                Left  : None, 
                Right : None,
                Top   : None,
                Down  : None
            }

    def add_link(self, direction, unit):
        if direction not in self.link_dict.keys():
            raise Exception()
        if not isinstance(unit, Node):
            raise Exception()
        
        #TODO add constraint
        self.link_dict[direction] = unit
        unit.link_dict[direction.reverse] = self

    
    def show_link(self, direction):
        if direction not in self.link_dict.keys():
            raise Exception()
        # if self.link_dict[direction]==None:
        #     print()
        return self.link_dict[direction]

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return self._name


class PEUnit(Node):
    def __init__(self, name, operation_type=Both, bit_width=16, index=[0,0,0], pipe_col=True, pipe_row=True, width_col=8, width_row=8):
        super().__init__(name, operation_type, bit_width, index, pipe_col, pipe_row, width_col, width_row)
        # self.operation_type     = operation_type
        # self.bit_width          = bit_width
        # self.index              = index
        # self.core               = index[0]
        # self.row                = index[1]
        # self.col                = index[2]
        # self.pipe_col           = pipe_col
        # self.pipe_row           = pipe_row
        # self.width_row          = width_row
        # self.width_col          = width_col


class Accumulator(Node):
    def __init__(self, name, operation_type=Add, bit_width=16, index=[0,0,0], width_col=8, width_row=8):
        super().__init__(name=name, operation_type=operation_type, bit_width=bit_width, index=index, width_col=width_col, width_row=width_row)
        





    
class NetworkShow(object):
    def __init__(self, name='systolic_array_0'):
        self.name     = name
        self.G        = nx.MultiDiGraph()
        self.logger   = NetworkLogger(name)
        self.pe_list  = []
        self.acc_list = []
        self.pos      = {}

    def add(self, node):
        if isinstance(node, Node):
            self.pe_list.append(node)
        # elif isinstance(node, Accumulator):
        #     self.acc_list.append(node)
        else:
            raise Exception()
        self.logger.info('add %s to network.' % node)
        self.G.add_node(node)

    def link(self, src, dst):
        self.logger.info('link %s to %s.' % (src, dst))
        self.G.add_node(src)
        self.G.add_node(dst)
        self.G.add_edge(src, dst)

    def _show(self, height, width):
        self.logger.info('%s generate and show.'% self.name)
        for node in self.G.nodes():
            self.G.nodes[node]['layer'] = node.layer
        
        pos =self.pos

        plt.figure(figsize=(width*1.5, height*1.5))

        # nx.draw(self.G)
        color = [x.color for x in self.pe_list]
        nx.draw_networkx_nodes(self.G, pos, node_color=color, nodelist=self.pe_list, node_size = 2000, node_shape ='s')


        nx.draw_networkx_edges(self.G, pos, node_size=1600, node_shape='s')
        nx.draw_networkx_labels(self.G, pos, font_size=8, font_color="black")
        plt.axis("equal")
        #plt.show()
        if not os.path.exists('result'): os.makedirs('result')
        plt.savefig("result/%s.network.png" % self.name, format='PNG')


# nx_0 = NetworkShow()
# node_0 = PE(name='pe_0_0')
# node_1 = PE(name='pe_0_1')
# node_2 = PE(name='pe_1_0')
# node_3 = PE(name='pe_1_1')
# node_4 = PE(name='pe_2_0')
# node_5 = PE(name='pe_2_1')
# nx_0.add(node_0)
# nx_0.add(node_1)
# nx_0.add(node_2)
# nx_0.add(node_3)
# nx_0.add(node_4)
# nx_0.add(node_5)
# nx_0.link(node_0,node_1)
# nx_0.link(node_0,node_2)
# nx_0.link(node_2,node_3)
# nx_0.link(node_1,node_3)
# nx_0.link(node_3,node_5)
# nx_0.link(node_2,node_4)
# nx_0.link(node_4,node_5)
# nx_0.pos= {
#     node_0: (0,0),
#     node_1: (0,1),
#     node_2: (1,0),
#     node_3: (1,1),
#     node_4: (2,0),
#     node_5: (2,1)
# }
# nx_0._show()

