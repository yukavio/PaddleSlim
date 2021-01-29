import logging
import numpy as np
import paddle
from ..common import get_logger
from .var_group import *
from .pruning_plan import *
from .filter_pruner import FilterPruner

__all__ = ['HRANKFilterPruner']

_logger = get_logger(__name__, logging.INFO)


class HRANKFilterPruner(FilterPruner):
    def __init__(self, model, inputs, sen_file=None):
        super(HRANKFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file)
        self.model = model
        self.hooks = []

    def cal_mask(self, var_name, pruned_ratio, group):
        for _item in group[var_name]:
            if _item['pruned_dims'] == [0]:
                value = _item['value']
                layer = _item['layer']
                pruned_dims = _item['pruned_dims']
        reduce_dims = [
            i for i in range(len(value.shape)) if i not in pruned_dims
        ]
        ranks_of_filter = np.array(layer.total_rank)
        sorted_idx = ranks_of_filter.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]
        mask_shape = [value.shape[i] for i in pruned_dims]
        mask = np.ones(mask_shape, dtype="int32")
        mask[pruned_idx] = 0
        return mask

    def print_statistic(self):
        for m in self.model.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                print(m.total_rank)

    def cal_act_rank(self, samples):
        self.model.apply(self.add_hooks)
        for i in range(len(samples)):
            data = samples[i].reshape([1] + list(samples[i].shape))
            print('data shape:' + str(data.shape))
            out = self.model(paddle.to_tensor(data).astype('float32'))

    def add_hooks(self, m):
        if (isinstance(m, paddle.nn.Conv2D)):
            m.register_buffer(
                'total_rank', paddle.zeros(
                    [m._out_channels], dtype='int64'))
            handle = m.register_forward_post_hook(cal_rank)
            self.hooks.append(handle)

    def remove_handle(self):
        for handle in self.hooks:
            handle.remove()
        for m in self.model.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                m._buffers.pop('total_rank')


def cal_rank(m, x, y):
    ranks = []
    y = y.numpy()
    for i in range(m._out_channels):
        ranks.append(np.linalg.matrix_rank(y[0][i], tol=1e-3))
    m.total_rank = paddle.add(paddle.to_tensor(
        ranks, dtype='int64'),
                              m.total_rank)
