# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append("../")
import unittest
import paddle
import paddle.fluid as fluid
import numpy as np
from paddleslim.prune import Pruner
from layers import conv_bn_layer
from paddleslim.prune import SensitivePruner
from paddleslim.analysis import flops
sys.path.append("../demo")
from models import MobileNet


class TestPrune(unittest.TestCase):
    def test_prune(self):
        image = fluid.layers.data(
            name='image', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        model = MobileNet()
        out = model.net(input=image, class_dim=10)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
        optimizer = fluid.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            regularization=fluid.regularizer.L2Decay(4e-5))
        optimizer.minimize(avg_cost)
        main_prog = fluid.default_main_program()
        val_prog = main_prog.clone(for_test=True)

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        feeder = fluid.DataFeeder([image, label], place, program=main_prog)
        train_reader = paddle.fluid.io.batch(
            paddle.dataset.mnist.train(), batch_size=64)
        eval_reader = paddle.fluid.io.batch(
            paddle.dataset.mnist.test(), batch_size=64)

        def train(program):
            iter = 0
            for data in train_reader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print(
                        'train iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                        format(iter, cost, top1, top5))
                break

        def test(program):
            iter = 0
            result = [[], [], []]
            for data in eval_reader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print('eval iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                          format(iter, cost, top1, top5))
                result[0].append(cost)
                result[1].append(top1)
                result[2].append(top5)
                break
            print(' avg loss {}, acc_top1 {}, acc_top5 {}'.format(
                np.mean(result[0]), np.mean(result[1]), np.mean(result[2])))
            return np.mean(result[1])

        train(main_prog)
        test(val_prog)

        pruner = SensitivePruner(place, test)

        start = 0
        end = 1
        for param in main_prog.global_block().all_parameters():
            #print(param.name)
            break
        base_flops = flops(val_prog)

        for iter in range(start, end):
            main_prog, val_prog = pruner.greedy_prune(
                main_prog, val_prog, [param.name], 0.03, topk=1)


if __name__ == '__main__':
    unittest.main()
