# net: "itracker_train_val.prototxt"
# test_iter: 1402
# test_interval: 1000
# base_lr: 0.001
# momentum: 0.9
# weight_decay: 0.0005
# lr_policy: "step"
# gamma: 0.1
# stepsize: 75000
# display: 20
# max_iter: 150000
# snapshot: 1000
# snapshot_prefix: "snapshots/itracker"
# solver_mode: GPU

from caffe import layers as L, params as P, to_proto


def itracker_net(train=True):
    n = caffe.NetSpec()

    # Data layer
    if train:
        n.data, n.label = L.Data(
            source="train_data.txt", backend=P.Data.LMDB, batch_size=64, ntop=2)
    else:
        n.data = L.Data(source="test_data.txt",
                        backend=P.Data.LMDB, batch_size=64, ntop=1)

    # Define the network architecture
    n.conv1 = L.Convolution(n.data, kernel_size=3,
                            num_output=64, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    n.conv2 = L.Convolution(n.pool1, kernel_size=3,
                            num_output=128, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.relu2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    # Add more layers as needed

    # Fully connected layers
    n.fc1 = L.InnerProduct(n.pool2, num_output=256,
                           weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.fc1, in_place=True)

    # Output layer
    n.score = L.InnerProduct(
        n.relu3, num_output=NUM_CLASSES, weight_filler=dict(type='xavier'))

    # Loss layer
    if train:
        n.loss = L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()


# Set solver parameters
solver_param = {
    'test_iter': 1402,
    'test_interval': 1000,
    'base_lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'lr_policy': "step",
    'gamma': 0.1,
    'stepsize': 75000,
    'display': 20,
    'max_iter': 150000,
    'snapshot': 1000,
    'snapshot_prefix': "snapshots/itracker",
    'solver_mode': P.Solver.GPU
}

# Write solver file
with open('solver.prototxt', 'w') as f:
    f.write(str(to_proto(solver_param)))

# Write train and test net files
with open('itracker_train_val.prototxt', 'w') as f:
    f.write(str(itracker_net(train=True)))
