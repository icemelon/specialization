import os
import random
import caffe
import collections
import numpy as np
from caffe.proto import caffe_pb2

import env


def prepare_input(net, inputs, oversample=False, batch=None):
    # Scale to standardize input dimensions.
    input_ = np.zeros((len(inputs),
        net.image_dims[0], net.image_dims[1], inputs[0].shape[2]),
        dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = caffe.io.resize_image(in_, net.image_dims)

    if oversample:
        # Generate center, corner, and mirrored crops.
        input_ = caffe.io.oversample(input_, net.crop_dims)
    else:
        # Take center crop.
        center = np.array(net.image_dims) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([
            -net.crop_dims / 2.0,
            net.crop_dims / 2.0
        ])
        crop = crop.astype(int)
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

    # Classify
    if batch is None:
      caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                          dtype=np.float32)
    else:
      caffe_in = np.zeros([batch, input_.shape[3], input_.shape[1],
                           input_.shape[2]], dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = net.transformer.preprocess(net.inputs[0], in_)

    return caffe_in


def init_net(config, section, batch=1):
    model_name = config.get(section, "name")
    model_def = os.path.join(env.pretrained_model_dir, "%s.prototxt" % model_name)
    pretrained = os.path.join(env.pretrained_model_dir, "%s.caffemodel" % model_name)
    dim = config.getint(section, "image_dim")
    raw_scale = config.getfloat(section, "raw_scale")
    mean_opt = config.get(section, "mean")
    if mean_opt.endswith("npy"):
        mean = np.load(mean_opt)
    elif mean_opt.endswith("binaryproto"):
        with open(mean_opt, 'rb') as f:
            binary = f.read()
        mean_blob = caffe_pb2.BlobProto()
        mean_blob.ParseFromString(binary)
        mean = caffe.io.blobproto_to_array(mean_blob)
        if len(mean.shape) == 4:
            mean = mean[0]
        assert len(mean.shape) == 3
    else:
        mean = np.array([float(v) for v in mean_opt.split(',')])
    net = caffe.Classifier(model_def, pretrained,
                           channel_swap=(2,1,0),
                           mean=mean,
                           raw_scale=raw_scale,
                           image_dims=(dim, dim),
                           batch=batch)
    caffe.set_mode_gpu()
    return net


def init_special_net(config, special_path, batch=1):
    model_def = special_path + '.prototxt'
    pretrained = special_path + '.caffemodel'
    dim = config.getint("compact_model", "image_dim")
    raw_scale = config.getfloat("compact_model", "raw_scale")
    mean_opt = config.get("compact_model", "mean")
    if mean_opt.endswith("npy"):
        mean = np.load(mean_opt)
    elif mean_opt.endswith("binaryproto"):
        with open(mean_opt, 'rb') as f:
            binary = f.read()
        mean_blob = caffe_pb2.BlobProto()
        mean_blob.ParseFromString(binary)
        mean = caffe.io.blobproto_to_array(mean_blob)
        if len(mean.shape) == 4:
            mean = mean[0]
        assert len(mean.shape) == 3
    else:
        mean = np.array([float(v) for v in mean_opt.split(',')])
    net = caffe.Classifier(model_def, pretrained,
                           channel_swap=(2,1,0),
                           mean=mean,
                           raw_scale=raw_scale,
                           image_dims=(dim, dim),
                           batch=batch)
    caffe.set_mode_gpu()
    return net


def gen_skewed_train_data(train_dir, training_label, target_list, nitem_per_class, incontext_percent):
    nclasses = len(target_list)
    initem = collections.defaultdict(list)
    outitem = []

    total_incontext = 0
    with open(training_label) as f:
        for line in f:
            fn, label  = line.strip().split()
            fn = os.path.join(train_dir, fn)
            label = int(label)
            if label in target_list:
                initem[label].append( (fn, label) )
                total_incontext += 1
            else:
                outitem.append( (fn, nclasses) )

    reverse_index = {}
    for i in range(nclasses):
        reverse_index[target_list[i]] = i

    if nitem_per_class == 0:
        # we need all incontext items from the training data
        num_outcontext = int(total_incontext / incontext_percent * (1 - incontext_percent))
        train_data = random.sample(outitem, num_outcontext)
        for i in target_list:
            for j in initem[i]:
                train_data.append( (j[0], reverse_index[j[1]]) )
    else:
        in_data = []
        for i in target_list:
            if len(initem[i]) < nitem_per_class:
                target = initem[i]
            else:
                target = random.sample(initem[i], nitem_per_class)
            for j in target:
                in_data.append( (j[0], reverse_index[j[1]]) )
        num_outcontext = int(len(in_data) / incontext_percent * (1 - incontext_percent))
        out_data = random.sample(outitem, num_outcontext)
        train_data = in_data + out_data

    random.shuffle(train_data)
    return train_data, num_outcontext


def gen_skewed_test_data(val_dir, val_label, target_classes, incontext_percent):
    initems = []
    outitems = []
    with open(val_label) as f:
        for line in f:
            fn, label = line.strip().split()
            fn = os.path.join(val_dir, fn)
            label = int(label)
            if label in target_classes:
                initems.append([fn, label])
            else:
                outitems.append([fn, label])
    if incontext_percent == 1.0:
        num_incontext = len(initems)
        ret = initems
    elif incontext_percent == 0:
        num_incontext = 0
        ret = outitems
    else:
        num_incontext = len(initems)
        num_outcontext = int(num_incontext / incontext_percent * (1-incontext_percent))
        ret = initems + random.sample(outitems, num_outcontext)
    random.shuffle(ret)
    return ret, num_incontext


def stitch_model(origin_path, retarget_path, retarget_layer, output_path):
    os.system('%s %s %s %s %s' % (env.stitch_network_bin, origin_path,
                                  retarget_path, retarget_layer, output_path))
