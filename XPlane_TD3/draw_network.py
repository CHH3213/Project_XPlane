'''

'''

import numpy as np
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow


def draw_feed_forward(ax, num_node_list):
    '''
    draw a feed forward neural network.

    Args:
        num_node_list (list<int>): 每层节点数组成的列表
    '''
    num_hidden_layer = len(num_node_list) - 2  # 隐藏层数
    token_list = ['\sigma^z'] + \
        ['y^{(%s)}' % (i + 1) for i in range(num_hidden_layer)] + ['\psi']
    kind_list = ['nn.input'] + ['nn.hidden'] * num_hidden_layer + ['nn.output']
    radius_list = [0.005] + [0.001] * num_hidden_layer + [0.005]   # 半径大小
    y_list = - 1.5 * np.arange(len(num_node_list))  # 每一层节点所在的位置的纵轴坐标，全取负值说明网络是自顶而下的

    seq_list = []
    for n, kind, radius, y in zip(num_node_list, kind_list, radius_list, y_list):
        b = NodeBrush(kind, ax)
        seq_list.append(node_sequence(b, n, center=(0, y)))

    eb = EdgeBrush('-->', ax)
    for st, et in zip(seq_list[:-1], seq_list[1:]):
        connecta2a(st, et, eb)
    #for i, layer_nodes in enumerate(seq_list):
        #[node.text('$z_%i^{(%i)}$'%(j, i), 'center', fontsize=16) for j, node in enumerate(layer_nodes)]
    return seq_list


def real_bp():
    with DynamicShow((6, 6), '_feed_forward.png') as d:  # 隐藏坐标轴
        seq_list = draw_feed_forward(d.ax, num_node_list=[8, 4, 4,4])
        # for i, layer_nodes in enumerate(seq_list):
        #     [node.text('$z_{%i}^{(%i)}$'%(j, i), 'center', fontsize=10) for j, node in enumerate(layer_nodes)]


if __name__ == '__main__':
    real_bp()
