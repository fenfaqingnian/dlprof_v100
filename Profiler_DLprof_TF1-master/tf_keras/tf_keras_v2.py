# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import numpy as np
import tensorflow
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from nvtx.plugins.tf.keras.layers import NVTXStart, NVTXEnd
from nvtx.plugins.tf.keras.callbacks import NVTXCallback
import time

NUM_EPOCHS = 200

# load pima indians dataset
dataset = np.loadtxt('data.csv', delimiter=',')
features = dataset[:,0:8]
labels = dataset[:,8]


def DenseBinaryClassificationNet(input_shape=(8,)):
    inputs = Input(input_shape)

    x = inputs
    x, marker_id, domain_id = NVTXStart(message='Dense 1',
                                        domain_name='forward',
                                        trainable=True)(x)
    x = Dense(8*1024, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 1 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])


    x, marker_id, domain_id = NVTXStart(message='Dense 2',
                                        domain_name='forward')(x)
    x = Dense(8*512, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 2 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])


    x, marker_id, domain_id = NVTXStart(message='Dense 3',
                                        domain_name='forward')(x)
    x = Dense(8*1024, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 3 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])


    x, marker_id, domain_id = NVTXStart(message='Dense 4',
                                        domain_name='forward')(x)
    x = Dense(8*512, activation='relu')(x)
    x = NVTXEnd(grad_message='Dense 4 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])

    x, marker_id, domain_id = NVTXStart(message='Dense 5',
                                        domain_name='forward')(x)
    x = Dense(1, activation='sigmoid')(x)
    x = NVTXEnd(grad_message='Dense 5 grad',
                grad_domain_name='backwards')([x, marker_id, domain_id])

    predictions = x
    model = Model(inputs=inputs, outputs=predictions)
    return model


nvtx_callback = NVTXCallback()

startTime=time.time()
tensorflow.config.optimizer.set_jit(True) # Enable XLA
with tensorflow.device('/gpu:0'):
    dataset=tensorflow.data.Dataset.from_tensor_slices((features, labels))
    train_data=dataset.cache().batch(8*8).repeat()
    
    opt = tensorflow.keras.optimizers.Adam(1e-4)     
    
    model = DenseBinaryClassificationNet()
    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    #add mixed precision
    opt = tensorflow.train.experimental.enable_mixed_precision_graph_rewrite(sgd)
    model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
#
    
    model.fit(train_data, epochs=100,steps_per_epoch=8*8,
              callbacks=[nvtx_callback])
print('Using GPU for training with XLA + MixedPrecision + tf.data pipeline took {0} seconds'.format(time.time() - startTime))



