

def BasicParse(sim_model, layer_name, layer_shape):
    array_height = sim_model.array_height
    array_width  = sim_model.array_width
    # print(f'Shape of Systolic Array: {array_height}x{array_width}')
    input_sequence = layer_shape[0]
    hidden_size    = layer_shape[1]
    output_size    = layer_shape[2]

    use_shape_dict={
        'full_use':         [None,0],
        'input_partial':    [None,0],
        'hidden_partial':   [None,0],
        'both_partial':     [None,0]
    }

    input_tile =  0 if input_sequence<array_height else int(input_sequence/array_height)
    hidden_tile = 0 if output_size<array_width else int(output_size/array_width)

    remain_input_tile_width = input_sequence if input_sequence<array_height else input_sequence - input_tile*array_height
    remain_hidden_tile_width = output_size if output_size<array_width else output_size - hidden_tile*array_width

    if remain_hidden_tile_width!=0 and remain_input_tile_width!=0:
        use_shape_dict['both_partial'] = [[remain_input_tile_width, remain_hidden_tile_width], hidden_size]
    if input_tile>0 and remain_hidden_tile_width!=0:
        use_shape_dict['hidden_partial'] = [[array_height, remain_hidden_tile_width], input_tile*hidden_size]
    if hidden_tile>0 and remain_input_tile_width!=0:
        use_shape_dict['input_partial'] = [[remain_input_tile_width, array_width], hidden_tile*hidden_size]
    if hidden_tile>0 and input_tile>0:
        use_shape_dict['full_use'] = [[array_height, array_width], input_tile*hidden_tile*hidden_size]

    full_use_start       = sim_model.collect_start_cycle(use_shape_dict['full_use'][0])
    input_partial_start  = sim_model.collect_start_cycle(use_shape_dict['input_partial'][0]) if use_shape_dict['input_partial'][0] != None else 0
    hidden_partial_start = sim_model.collect_start_cycle(use_shape_dict['hidden_partial'][0]) if use_shape_dict['hidden_partial'][0] != None else 0
    both_partial_start   = sim_model.collect_start_cycle(use_shape_dict['both_partial'][0]) if use_shape_dict['both_partial'][0] != None else 0

    cycle_list = []
    cycle_list.append(full_use_start + use_shape_dict['full_use'][1])
    cycle_list.append(input_partial_start + use_shape_dict['full_use'][1] + use_shape_dict['input_partial'][1] + use_shape_dict['hidden_partial'][1])
    cycle_list.append(hidden_partial_start + use_shape_dict['full_use'][1] + use_shape_dict['input_partial'][1] + use_shape_dict['hidden_partial'][1])
    cycle_list.append(both_partial_start + use_shape_dict['full_use'][1] + use_shape_dict['input_partial'][1] + use_shape_dict['hidden_partial'][1] + use_shape_dict['both_partial'][1])

    cycle_list.append(use_shape_dict['full_use'][1] + use_shape_dict['input_partial'][1] + use_shape_dict['hidden_partial'][1] + use_shape_dict['both_partial'][1])

    total_cycle = max(cycle_list)

    print(f'Parse layer: {layer_name}  input shape: [{input_sequence},{hidden_size}]  hidden shape: [{hidden_size}, {output_size}]  use_shape: {use_shape_dict}  sequence length: {total_cycle}')

    return use_shape_dict, total_cycle, layer_shape


def OtherParse(sim_model, layer_name, layer_shape):
    pass


