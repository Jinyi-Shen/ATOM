import os


def get_node_info(x):
    name = []
    add_node = False
    reverse_node = False
    if x == 0:
        pass
    elif x == 2:  # R
        name = ['C']
    elif x == 1:  # RC series
        name = ['R', 'C']
        add_node = True
    elif x == 3 or x == 4:  # rev Av&c series + r_gnd
        reverse_node = True
        add_node = True
        name = ['G', 'Rprs', 'Cprsin', 'Cprsout', 'C', 'R']
    elif x == 5 or x == 6:  # Av
        name = ['G', 'Rprs', 'Cprsin', 'Cprsout']
    else:
        raise NotImplementedError
    return name, add_node, reverse_node


def write_netlist_conf(block_type, node_pair, node_names, netlist, conf, c_output=True, c_input=True, ff=True,
                       n_p_n=True):
    if block_type == 0:
        return
    if ff:
        if n_p_n:
            if node_pair[0] == 0 and node_pair[1] == 3:
                block_type = 6
            else:
                block_type = 5
        else:
            block_type = 5

    name, add_node, reverse_node = get_node_info(block_type)

    if add_node:
        if ff:
            mid_name = node_names[node_pair[0]] + '_' + node_names[node_pair[1]] + '_ff'
        else:
            mid_name = node_names[node_pair[0]] + '_' + node_names[node_pair[1]]
        suffix1 = '_' + node_names[node_pair[0]] + '_' + mid_name
        suffix2 = '_' + mid_name + '_' + node_names[node_pair[1]]
        suffix = [suffix1, suffix2]
    else:
        if ff:
            suffix = ['_' + node_names[node_pair[0]] + '_' + node_names[node_pair[1]] + '_ff']
        else:
            suffix = ['_' + node_names[node_pair[0]] + '_' + node_names[node_pair[1]]]

    for i, _name in enumerate(name):
        if c_output == False and i == len(name) - 1:
            continue
        if c_input == False and i == 2:
            continue
        if add_node:
            if block_type == 1:
                if i == 0:
                    _suffix = suffix[0]
                else:
                    _suffix = suffix[1]
            else:
                if i < 4:
                    _suffix = suffix[0]
                elif i == 4:
                    _suffix = suffix[1]
                else:
                    _suffix = '_' + mid_name + '_0'
        else:
            _suffix = suffix[0]
        block_name = _name + _suffix
        if i == 0:
            block_name0 = block_name
        if add_node:
            if len(name) >= 5:  # serial Av&R/C
                if i == 4:  # R1/C1
                    pin_name = mid_name + ' ' + node_names[node_pair[1]]
                elif i == 2 or i == 5:
                    pin_name = mid_name + ' gnd'
                elif i == 0:  # G
                    pin_name = node_names[node_pair[0]] + ' gnd ' + mid_name + ' gnd'
                else:
                    pin_name = node_names[node_pair[0]] + ' gnd'
            else:
                if i == 1:  # R1/C1
                    pin_name = mid_name + ' ' + node_names[node_pair[1]]
                else:
                    pin_name = node_names[node_pair[0]] + ' ' + mid_name
        elif len(name) == 4:  # single Av
            if i == 0:  # g
                pin_name = node_names[node_pair[1]] + ' gnd ' + node_names[node_pair[0]] + ' gnd'
            elif i == 2:  # Cin_prs
                pin_name = node_names[node_pair[0]] + ' gnd'
            else:
                pin_name = node_names[node_pair[1]] + ' gnd'
        else:
            pin_name = node_names[node_pair[0]] + ' ' + node_names[node_pair[1]]
        if pin_name == 'in gnd':
            continue
        write_var = True
        if len(name) < 4:
            val_name = block_name
        else:
            if i == 2:
                write_var = False
                val_name = '\'{}/6.28*5n\''.format(block_name0)
            elif i == 1:
                val_name = '\'{}/{}\''.format(block_name, block_name0)
            elif i == 0 and block_type in [4, 6]:
                val_name = '\'-1*{}\''.format(block_name)
            elif i == 5:
                write_var = False
                val_name = '\'1/{}\''.format(block_name0)
            else:
                val_name = block_name
        netlist.writelines(' '.join([block_name, pin_name, val_name]) + '\n')
        if write_var:
            conf.writelines('des_var ' + block_name + '\n')
    return


def amp_generator(topo_vector, circuit_dir):
    conf_file = os.path.join(circuit_dir, 'conf')
    netlist_file = os.path.join(circuit_dir, 'opamp.sp')
    conf = open(conf_file, 'w')
    netlist = open(netlist_file, 'w')

    netlist.writelines('.subckt opamp in vo gnd\n')

    node_pairs = [[0, 1], [1, 2], [2, 3]]
    node_pairs_ff = [[0, 2], [0, 3], [1, 3]]
    node_pairs_gnd = [[1, 4], [2, 4]]
    node_names = ['in', '1', '2', 'vo', 'gnd']
    if topo_vector[0]:  # -+-
        write_netlist_conf(5, node_pairs[0], node_names, netlist, conf, c_output=False, c_input=False, ff=False)
        write_netlist_conf(6, node_pairs[1], node_names, netlist, conf, c_output=False, ff=False)
        write_netlist_conf(5, node_pairs[2], node_names, netlist, conf, c_output=False, ff=False)
        node_pairs_fb = [[1, 3], [2, 3]]
        n_p_n = True
    else:  # +-+
        write_netlist_conf(6, node_pairs[0], node_names, netlist, conf, c_output=False, c_input=False, ff=False)
        write_netlist_conf(5, node_pairs[1], node_names, netlist, conf, c_output=False, ff=False)
        write_netlist_conf(6, node_pairs[2], node_names, netlist, conf, c_output=False, ff=False)
        node_pairs_fb = [[1, 3], [1, 2]]
        n_p_n = False
    for i in range(1, 4):
        write_netlist_conf(topo_vector[i], node_pairs_ff[i - 1], node_names, netlist, conf, ff=True, n_p_n=n_p_n)
    for i in range(4, 6):
        write_netlist_conf(topo_vector[i], node_pairs_fb[i - 4], node_names, netlist, conf, ff=False, n_p_n=n_p_n)
    for i in range(6, 8):
        write_netlist_conf(topo_vector[i], node_pairs_gnd[i - 6], node_names, netlist, conf, ff=False, n_p_n=n_p_n)
    netlist.writelines('C_L vo gnd c=10n\n') #10nf
    # netlist.writelines('C_L vo gnd c=10p\n') #10pf
    netlist.writelines(".ends")
    netlist.close()
    conf.close()
    return