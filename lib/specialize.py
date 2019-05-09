import os
import random
import time
import numpy as np
import shutil
import plyvel
import cPickle as pickle
import google.protobuf.text_format
import caffe
from caffe.proto import caffe_pb2

import dbg
import env
import utils

BATCH = 128

def generate_db(net, lst, layer_name, dbpath=None, blobKey=None, batch=1):
    assert len(lst) > 0, "Target list is empty"
    if dbpath is None:
        dbpath = "_train"
    if os.path.exists(dbpath):
        shutil.rmtree(dbpath)

    db = plyvel.DB(dbpath, create_if_missing=True, error_if_exists=True, write_buffer_size=268435456)
    wb = db.write_batch()
    count = 0
    for i in range(len(lst)/batch+1):
        images = []
        labels = []
        for item in lst[i*batch : (i+1)*batch]:
            path = item[0]
            labels.append(item[1])
            im = caffe.io.load_image(path)
            images.append(im)

        if len(images) == 0: break
        prepared = utils.prepare_input(net, images, False, batch)

        if blobKey is not None:
            out = net.forward(end=layer_name, blobKey=blobKey, **{net.inputs[0]: prepared})[layer_name]
        else:
            out = net.forward(end=layer_name, **{net.inputs[0]: prepared})[layer_name]
        for j in range(len(images)):
            cur = out[j]
            if cur.ndim < 3:
                dim = [1] * (3 - cur.ndim)
                dim.extend(cur.shape)
                cur = cur.reshape(dim)
            datum = caffe.io.array_to_datum(np.array(cur).astype(float), labels[j])
            wb.put('%08d' % count, datum.SerializeToString()) 
            count += 1
            if count % 1000 == 0:
                wb.write()
                wb = db.write_batch()
    wb.write()
    db.close()
    

def retarget(nitem, top, nclasses, snapshot_prefix, traindb=None, step_epoch=15,
             max_epoch=40, base_lr=0.1, gamma=0.1):
    import tempfile
    topnet = caffe_pb2.NetParameter()
    with open(top) as f:
        google.protobuf.text_format.Merge(f.read(), topnet)

    layer = topnet.layer[1]
    layer.inner_product_param.num_output = nclasses
    if traindb is None:
        traindb = "_train"
    topnet.layer[0].data_param.source = traindb
    
    topfile = tempfile.NamedTemporaryFile(delete=False)
    topfile.write(google.protobuf.text_format.MessageToString(topnet))  
    topfile.close()

    solver_param = caffe_pb2.SolverParameter() 
    with open(os.path.join(env.template_dir,  "solver_template.prototxt")) as f:
        google.protobuf.text_format.Merge(f.read(), solver_param)
    solver_param.net = topfile.name
    solver_param.base_lr = base_lr
    solver_param.momentum = 0.9
    solver_param.gamma = gamma
    solver_param.weight_decay = 0.0005
    solver_param.display = 20
    iter_per_epoch = nitem / 256 + 1
    dbg.dbg("#iterations per epoch: %s" % iter_per_epoch)
    solver_param.stepsize = iter_per_epoch * step_epoch
    solver_param.max_iter = iter_per_epoch * max_epoch
    solver_param.snapshot_prefix = snapshot_prefix
    dbg.dbg("train for %s iterations, step size: %s iterations" % 
            (solver_param.max_iter, solver_param.stepsize))

    f = tempfile.NamedTemporaryFile(delete = False)
    f.write(google.protobuf.text_format.MessageToString(solver_param))
    name = f.name
    f.close()

    solver = caffe.SGDSolver(f.name)
    solver.solve()

    os.remove(name)
    os.remove(topfile.name)

    return str(solver_param.snapshot_prefix) + "_iter_" + str(solver_param.max_iter) + ".caffemodel"


def specialize(config, dbpath=None):
    global BATCH
    nitem_per_class = config.getint("train", "item_per_class")
    incontext_percentage = config.getfloat("train", "incontext_percent")
    
    train_root = config.get("data", "train_image_root")
    train_label = config.get("data", "train_label")
    target_list = [int(lbl) for lbl in config.get("train", "target_labels").split(",")]
    nclasses = len(target_list)+1 if incontext_percentage < 1.0 else len(target_list)
    dbg.dbg("# item per cls: %d, # cls: %d, percentage: %.2f", nitem_per_class, nclasses, incontext_percentage)

    # generate list
    #random.seed(tuple(target_list))
    lst, num_outcontext = utils.gen_skewed_train_data(train_root, train_label, target_list, nitem_per_class, incontext_percentage)
    dbg.dbg("training set is picked. length: %d, # outcontext: %d", len(lst), num_outcontext)
    net = utils.init_net(config, 'compact_model', batch=BATCH)
    return _specialize(config, net, lst, nclasses, batch=BATCH, dbpath=dbpath)


def _specialize(config, net, lst, nclasses, batch=1, dbpath=None):
    image_root = config.get("data", "train_image_root")
    target_layer = config.get("train", "target_layer")
    if config.has_option("train", "target_blob"):
        target_blob = config.get("train", "target_blob")
    else:
        target_blob = target_layer
    bottom_out_blob = net.blobs[target_blob]
    try:
        step_epoch = config.getint("train", "step_epoch")
    except:
        step_epoch = 15
    try:
        max_epoch = config.getint("train", "max_epoch")
    except:
        max_epoch = 40
    try:
        base_lr = config.getfloat("train", "base_lr")
    except:
        base_lr = 0.1
    try:
        gamma = config.getfloat("train", "gamma")
    except:
        gamma = 0.1

    beg = time.time()
    generate_db(net, lst, target_layer, blobKey=target_blob, batch=batch, dbpath=dbpath)
    end = time.time()
    gen_db_time = end - beg

    top = config.get("train", "top")
    snapshot_prefix = config.get("train", "snapshot_prefix")
    retarget_file = retarget(len(lst), top, nclasses, snapshot_prefix, step_epoch=step_epoch,
                             max_epoch=max_epoch, base_lr=base_lr, gamma=gamma, traindb=dbpath)
    end2 = time.time()
    train_time = end2 - end
    dbg.dbg("forward: %f, retarget: %f", end-beg, end2-end)

    test_proto = config.get("train", "test")
    testnet = caffe_pb2.NetParameter()
    with open(test_proto) as f:
        google.protobuf.text_format.Merge(f.read(), testnet)
    testnet_input_shape = testnet.layer[0].input_param.shape[0]
    testnet_input_shape.dim[1] = bottom_out_blob.channels
    testnet_input_shape.dim[2] = bottom_out_blob.height
    testnet_input_shape.dim[3] = bottom_out_blob.width
    testnet.layer[1].inner_product_param.num_output = nclasses

    target_test = config.get("train", "target_test")
    with open(target_test, "w") as f:
        f.write(google.protobuf.text_format.MessageToString(testnet))

    shutil.copyfile(retarget_file, config.get("train", "target_model"))
    return gen_db_time, train_time

