import pandas as pd
import sqlite3

global_dict = {}

def update_global_dict(Node, input_info, output_info, attr_info, weight_info):
    if Node.op_type in global_dict.keys():
        value_dict = global_dict[Node.op_type]
        input_shape_dict = [tuple(max(input_shape[i],input_shape[i]) for i in range(len(input_shape))) for input_shape in input_info.values() ]
        value_dict['input_shape'] = input_shape_dict

        if Node.op_type == 'Gemm':
            for i in list(weight_info.values())[0]:
                if i not in value_dict['attribute']['weight']: value_dict['attribute']['weight'].append(i)
        if Node.op_type == 'Conv':
            value_dict['attribute']['output_channel'] = [min(list(weight_info.values())[0][0], value_dict['attribute']['output_channel'][0]), max(list(weight_info.values())[0][0], value_dict['attribute']['output_channel'][1])]
            value_dict['attribute']['input_channel']  = [min(list(weight_info.values())[0][1], value_dict['attribute']['input_channel'][0]), max(list(weight_info.values())[0][1], value_dict['attribute']['input_channel'][1])]
            # attr_info['output_channel'] = [list(weight_info.values())[0][0]]
            # attr_info['input_channel']  = [list(weight_info.values())[0][1]]

        for info_key, info_value in attr_info.items():
            if info_value[0] not in value_dict['attribute'][info_key]:
                value_dict['attribute'][info_key].append(info_value[0])
    else:
        if Node.op_type == 'Gemm':
            if not hasattr(attr_info,'weight'): attr_info['weight'] = [list(weight_info.values())[0][0]] 
            for i in list(weight_info.values())[0]:
                if list(weight_info.values())[0][1] not in attr_info['weight']: attr_info['weight'].append(i)
        if Node.op_type == 'Conv':
            attr_info['output_channel'] = [list(weight_info.values())[0][0], list(weight_info.values())[0][0]]
            attr_info['input_channel']  = [list(weight_info.values())[0][1], list(weight_info.values())[0][1]]
        global_dict[Node.op_type] = {'input_shape':[i for i in input_info.values()], \
                                        'attribute':attr_info}


def opt_global_dict():
    for key, value in global_dict.items():
        for key_inner, value_inner in value.items():
            if value_inner in [{},[]]:
                global_dict[key][key_inner] = 'Null'


def write_csv(path):
        print(global_dict)
        dataframe = pd.DataFrame(columns=['Op Type', 'Input Shape', 'Attribute'])
        dataframe['Op Type'] = global_dict.keys()
        # print(dataframe['Op Type'].value_counts())
        shape_list = [val['input_shape'] for val in global_dict.values()]
        attr_list  = [val['attribute'] for val in global_dict.values()]
        dataframe['Input Shape'] = shape_list
        dataframe['Attribute']   = attr_list
        dataframe.to_csv(path,index=False, sep=',')
        print("Write Finish")  
        return dataframe


def summary_to_csv(inst):
    pass
    
