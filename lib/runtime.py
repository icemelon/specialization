import os
import random
import time
import numpy as np
import cPickle as pickle
import tempfile
import shutil
import caffe

import dbg
import env
import utils
from default_config import *
from specialize import specialize
from proto import trace_pb2 as tpb


# runtime specs
train_default_config = {
  'obj': obj_default_config,
  'scene': scene_default_config,
  'face': face_default_config,
}

task_primary_accuracy = {
  'obj': .6893,
  'scene': .5810,
  'face': .9651
}

task_best_train_skew = {
  'obj': 0.6,
  'scene': 0.7,
  'face': 0.5,
}

task_total_class = {
  'obj': 1000,
  'scene': 205,
  'face': 201,
}  


""" Runtime classifier """
class RuntimeClassifier:
  def __init__(self, task, **config):
    self.task = task
    self.train_config = train_default_config[task]()
    self.total_class = task_total_class[task]

    # init config
    # cpu: using cpu (default: False)
    self.cpu = config['cpu'] if 'cpu' in config else False
    # max_ncls: max number of classes allowed for specialization (default: 30)
    self.max_ncls = config['max_ncls'] if 'max_ncls' in config else 30
    # min_skew: min skew for dominant classes to enter specialization mode (default: 0.5)
    self.min_skew = config['min_skew'] if 'min_skew' in config else 0.5
    # error_tolerance: error tolerance for special model compared to primary model
    #     acc of special model >= acc of primary model + error_tolerance (can be negative, default: 0.05)
    self.error_tolerance = config['error_tolerance'] if 'error_tolerance' in config else 0.05
    # minimin length of exploiting special model 
    self.min_exploit_length = config['min_exploit_length'] if 'min_exploit_length' in config else 90
    # maximum distance between two dominant set to determine whether merge them
    self.max_distance = config['max_distance'] if 'max_distance' in config else 0.5
    # epsilon probility to invoke oracle model
    self.epsilon = 0.01

    # init neural network
    self.primary_net = utils.init_net(self.train_config, 'best_model')
    self.primary_accuracy = task_primary_accuracy[task]
    self.compact_model = None # compact model name
    self.special_net = None # Caffe classifier of specialize model
    self.special_model_info = None
    self.model_dir = env.special_model_dir
    if not os.path.exists(self.model_dir):
        os.mkdir(self.model_dir)
    if self.cpu: caffe.set_mode_cpu()

    # init policy (only effective if window_explore is True)
    self.min_window = 30
    self.max_window = 180
    self.window = self.min_window
    self.window_init = True
    self.conf_lookup = {
      30: 2, 45: 2, 60: 2, 90: 3, 120: 3, 135: 3, 150: 3, 180: 3
    }

    # init classifier state
    self.state = tpb.NORMAL  # state
    self.frame = 0           # current frame index
    self.time = 0.0          # current time
    self.train_finish_time = 0.0  # specialization finish time
    self.history = []        # history trace
    self.current_epoch = 0   # start index in the history for current epoch
    self.sample = []         # current sample from oracle classifier
    self.sample_history = [] # sample history
    self.special_length = 0  # last specialize length
    self.incontext = []      # current special incontext
    self.epsilon_results = {}


  def set_compact_model(self, net):
    self.compact_model = net
    self.train_config.set('compact_model', 'name', net)
    # load specialize table
    with open(os.path.join(env.model_info_dir, '%s_special_table.p' % net), 'rb') as f:
      self.special_model_info = pickle.load(f)

  def get_avg_latency(self):
    avg_prepare_time = np.mean([e.prepare_time for e in self.history])
    avg_forward_time = np.mean([e.forward_time for e in self.history])
    return avg_prepare_time, avg_forward_time

  def predict(self, frame, img):
    self.frame = frame
    self.time = frame / 24.0
    pt = tpb.DataPoint()
    pt.frame = frame
    pt.mode = self.state
    if pt.mode == tpb.SPECIAL:
      self.classify_special(pt, img)
    else: # tpb.NORMAL | tpb.TRAIN
      self.classify_primary(pt, img)
    self.history.append(pt)
    # check if we need to update mode
    self.update()
    return pt.predict

  def classify_primary(self, pt, img):
    beg = time.time()
    prepared = utils.prepare_input(self.primary_net, [img], False)
    end = time.time()
    out = self.primary_net.forward_all(**{self.primary_net.inputs[0]: prepared})[self.primary_net.outputs[0]]
    end2 = time.time()
    pt.predict = pt.oracle_predict = out.argmax()
    pt.prepare_time = end - beg
    pt.forward_time = end2 - end
    pt.cascaded = False

  def classify_special(self, pt, img):
    beg = time.time()
    prepared = utils.prepare_input(self.special_net, [img], False)
    end = time.time()
    special_out = self.special_net.forward_all(**{self.special_net.inputs[0]: prepared})[self.special_net.outputs[0]]
    end2 = time.time()
    special_predict = special_out.argmax()
    prepare_time = end - beg
    forward_time = end2 - end
    if special_predict == len(self.incontext):
      # cascade to the non-special model
      pt.cascaded = True
      pt.special_predict = -1
      prepared = utils.prepare_input(self.primary_net, [img], False)
      beg = time.time()
      out = self.primary_net.forward_all(**{self.primary_net.inputs[0]: prepared})[self.primary_net.outputs[0]]
      end = time.time()
      pt.oracle_predict = out.argmax()
      forward_time += end - beg
      pt.predict = pt.oracle_predict
    else:
      pt.cascaded = False
      pt.special_predict = self.incontext[special_predict]
      pt.predict = pt.special_predict
      if random.random() <= self.epsilon:
        # call oracle to double-check the specialized model's prediction
        pt.mode = tpb.SPECIAL_EPSILON
        prepared = utils.prepare_input(self.primary_net, [img], False)
        beg = time.time()
        out = self.primary_net.forward_all(**{self.primary_net.inputs[0]: prepared})[self.primary_net.outputs[0]]
        end = time.time()
        pt.oracle_predict = out.argmax()
        if pt.oracle_predict not in self.epsilon_results: 
          self.epsilon_results[pt.oracle_predict] = [0, 0]
        self.epsilon_results[pt.oracle_predict][1] += 1
        if pt.oracle_predict != pt.special_predict:
          # add error count
          self.epsilon_results[pt.oracle_predict][0] += 1
        # we compare with oracle not in the critical path, thus 
        # invoking oracle won't increase the forward time
        #pt.predict = pt.oracle_predict
        #forward_time += end - beg
    pt.prepare_time = prepare_time
    pt.forward_time = forward_time
    pt.incontext = pt.predict in self.incontext

  def update(self):
    if self.compact_model is None: return
    if self.state == tpb.SPECIAL or self.state == tpb.TRAIN:
      # check if we need to exit special mode
      window = min(self.window, len(self.history) - self.current_epoch)
      skew = self.compute_incontext_skew(self.history[-window:], self.incontext)
      accuracy = self.estimate_accuracy(len(self.incontext), skew)
      self.history[-1].estimate_accuracy = accuracy
      dbg.dbg('[frame %s] skew %.3f (length: %s), accuracy: %.3f' % (self.frame, skew, window, accuracy))
      if self.state == tpb.TRAIN:
        if accuracy < self.primary_accuracy + self.error_tolerance:
          self.special_length = 0
          self.state = tpb.NORMAL
          dbg.dbg('[frame %s] TRAIN -> NORMAL' % (self.frame))
        elif self.time >= self.train_finish_time:
          # check if the training finishes
          self.state = tpb.SPECIAL
          dbg.dbg('[frame %s] TRAIN -> SPECIAL' % self.frame)
      else: #tpb.SPECIAL
        if self.exit_special(window, skew):
          self.special_length = self.count_special_length()
          self.state = tpb.NORMAL
          self.current_epoch = len(self.history)
          dbg.dbg('[frame %s] SPECIAL -> NORMAL, special length: %s, next epoch: %s' %
                  (self.frame, self.special_length, self.current_epoch))
    else:
      self.sample.append(self.history[-1])
      # we need to explore at least min_window samples
      if len(self.sample) < self.min_window: return
      if not self.window_init:
        merge = False
        # we consider merge curr exploration with last one only when length of 
        # specialization mode is less than min_exploit_length
        if self.special_length < self.min_exploit_length:
          # init the window size
          prev_domset = self.dominant_set(self.sample_history[-1])
          curr_domset = self.dominant_set(self.sample)
          dist = self.domset_distance(curr_domset, prev_domset)
          dbg.dbg('[frame %s] prev domset %s, curr domset %s, distance %s' %
                  (self.frame, prev_domset, curr_domset, dist))
          #if float(dist) / len(curr_domset) <= self.max_distance:
          if dist <= 2:
            # merge the current context with previous context
            merge = True
            self.sample = self.sample_history[-1] + self.sample
        self.window = len(self.sample)
        if self.window > self.max_window:
          dbg.dbg('[frame %s] window size %s too large, reduce to %s' % (self.frame, self.window, self.max_window))
          self.window = self.max_window
          self.sample = self.sample[-self.max_window:]
        self.window_init = True
        dbg.dbg('[frame %s] window size %s' % (self.frame, self.window))
        # decide the dominant set
        incontext = self.dominant_set(self.sample)
        if merge:
          incontext = list(set(incontext) | set(curr_domset))
      else:
        # keep the sample size same as window
        if len(self.sample) > self.window: self.sample.pop(0)
        incontext = self.dominant_set(self.sample)
      # check if we need to specialize
      skew = self.compute_incontext_skew(self.sample, incontext)
      sp_accuracy = self.estimate_accuracy(len(incontext), skew)
      if sp_accuracy > self.primary_accuracy + self.error_tolerance:
        dbg.dbg('[frame %s] NORMAL -> SPECIAL, window %s, incontext %s, skew %.2f, acc %s' % 
                (self.frame, self.window, incontext, skew, sp_accuracy))
        self.state = tpb.TRAIN
        self.incontext = incontext
        self.sample_history.append(self.sample)
        self.sample = []
        self.window_init = False
        self.epsilon_results = {}
        self.specialize(incontext)
  
  def exit_special(self, window, skew):
    # check if last 3 estimate accuracy is lower than oracle
    bad_length = 3
    bad = True
    for i in range(1, bad_length+1):
      pt = self.history[-i]
      if pt.estimate_accuracy >= self.primary_accuracy + self.error_tolerance:
        bad = False
        break
    if bad:
      dbg.dbg('[frame %s] last %s estimate accuracy lower than oracle, exit SPECIAL' %
              (self.frame, bad_length))
      return True
    # check if special model makes too many mistakes on any class
    epsilon_count = sum([self.epsilon_results[cl][1] for cl in self.epsilon_results])
    dbg.dbg('[frame %s] total epsilon %s, %s' % (self.frame, epsilon_count, str(self.epsilon_results)))
    max_error = max(2, float(epsilon_count) / len(self.incontext) * 0.5)
    for cl in self.epsilon_results:
      wrong, total = self.epsilon_results[cl]
      if wrong >= max_error:
        dbg.dbg('[frame %s] Wrong prediction for class %s: %s (%s) >= %s times, total epsilon count: %s, exit SPECIAL' %
                (self.frame, cl, wrong, total, max_error, epsilon_count))
        return True
    # otherwise, stay in SPECIAL mode
    return False

  def count_special_length(self):
    cnt = 0
    for pt in reversed(self.history):
      if pt.mode in [tpb.SPECIAL, tpb.SPECIAL_EPSILON]:
        cnt += 1
      else:
        break
    return cnt

  def specialize(self, target_list):
    global task_best_train_skew
    tmp_dir = tempfile.mkdtemp()
    db_dir = os.path.join(tmp_dir, "db")
    snapshot_dir = os.path.join(tmp_dir, "snapshot")
    os.mkdir(db_dir)
    os.mkdir(snapshot_dir)
    caffe.set_mode_gpu()
    retarget_model = self.compact_model + '_retarget'
    target_layer =  self.train_config.get('train', 'target_layer')
    special_model = self.compact_model + '_special'
    train_skew = task_best_train_skew[self.task]
    self.train_config.set('train', 'target_labels', ','.join([str(e) for e in target_list]))
    self.train_config.set('train', 'incontext_percent', str(train_skew))
    self.train_config.set('train', 'target_test', '%s/%s.prototxt' % (self.model_dir, retarget_model))
    self.train_config.set('train', 'target_model', '%s/%s.caffemodel' % (self.model_dir, retarget_model))
    self.train_config.set('train', 'snapshot_prefix', '%s/%s' % (snapshot_dir, retarget_model))
    db_time, train_time = specialize(self.train_config, dbpath=db_dir)
    dbg.dbg('specialize for %s (%s)' % (target_list, len(target_list)))
    dbg.dbg('gen db %s s, specialize %s s' % (db_time, train_time))
    beg = time.time()
    utils.stitch_model(os.path.join(env.pretrained_model_dir, self.compact_model), 
                       os.path.join(self.model_dir, retarget_model), target_layer,
                       os.path.join(self.model_dir, special_model))
    end = time.time()
    dbg.dbg('stitch model together %s s' % (end-beg))
    self.special_net = utils.init_special_net(self.train_config,
                                              os.path.join(self.model_dir, special_model))
    self.train_finish_time = self.time + train_time
    dbg.dbg('training finish time: %s' % self.train_finish_time)
    self.test_special_model(target_list)
    if self.cpu: caffe.set_mode_cpu()
    shutil.rmtree(tmp_dir)
    
  def test_special_model(self, targetlist):
    val_dir = self.train_config.get('data', 'val_image_root')
    val_label = self.train_config.get('data', 'val_label')
    data, _ = utils.gen_skewed_test_data(val_dir, val_label, targetlist, 1.0)
    correct = 0
    for fn, lbl in data:
      img = caffe.io.load_image(fn)
      prepared = utils.prepare_input(self.special_net, [img], False)
      out = self.special_net.forward_all(**{self.special_net.inputs[0]: prepared})[self.special_net.outputs[0]]
      predict = out.argmax()
      if predict != len(targetlist) and targetlist[predict] == lbl:
        correct += 1
    if len(data) == 0:
      accuracy = 1
    else:
      accuracy = float(correct) / len(data)
    dbg.dbg('incontext accuracy: %.4f' % accuracy)

  def compute_special_model_param(self, ncls):
    if ncls > self.max_ncls: return None
    if ncls == 0: return None
    if ncls < 5: ncls = 5
    for n in sorted(self.special_model_info.keys()):
      if n <= ncls:
        lower_ncls = n
      else:
        upper_ncls = n
        break
    gap = float(upper_ncls - lower_ncls)
    acc_in = self.special_model_info[lower_ncls][0] * (upper_ncls - ncls) / gap + \
             self.special_model_info[upper_ncls][0] * (ncls - lower_ncls) / gap
    err_in_out = self.special_model_info[lower_ncls][1] * (upper_ncls - ncls) / gap + \
                 self.special_model_info[upper_ncls][1] * (ncls - lower_ncls) / gap
    acc_out = self.special_model_info[lower_ncls][2] * (upper_ncls - ncls) / gap + \
              self.special_model_info[upper_ncls][2] * (ncls - lower_ncls) / gap
    return acc_in, err_in_out, acc_out

  def estimate_accuracy(self, ncls, skew):
    if ncls == 0: return 0
    if ncls > self.max_ncls: return 0
    if skew < self.min_skew: return 0
    acc_in, err_in_out, acc_out = self.compute_special_model_param(ncls)
    acc = skew * acc_in + skew * err_in_out * self.primary_accuracy + \
          (1-skew) * acc_out * self.primary_accuracy
    return acc

  def dominant_set(self, samples):
    if len(samples) == 0: return []
    conf = self.conf_lookup[len(samples)]
    class_cnt = {}
    for e in samples:
      if e.predict not in class_cnt: class_cnt[e.predict] = 0
      class_cnt[e.predict] += 1
    class_ranks = sorted(class_cnt.items(), key=lambda x: x[1], reverse=True)
    domset = []
    for cls, cnt in class_ranks:
      if cnt < conf: break
      domset.append(cls)
    return domset

  def domset_distance(self, curr, prev):
    dist = 0
    for c in curr:
      if c not in prev: dist += 1
    return dist

  def compute_incontext_skew(self, history, incontext):
    """
      Three possible patterns in history:
      (a) Oldest ===========================================> Latest
          +--------------------------------------------------------+
          | Predicted by primary | Predicted by special classifier |
          +--------------------------------------------------------+
      (b) Predicted only by primary classifier
      (c) Predicted only by special classifier
    """

    length = len(history)
    incontext_cnt = 0
    acc_p = self.primary_accuracy
    n = len(incontext)
    N = self.total_class
    # first compute the incontext skew of the segment by primary classifier
    lb = 0
    i = 0
    while i < length and history[i].mode not in [tpb.SPECIAL, tpb.SPECIAL_EPSILON]:
      i += 1
    rb = i
    if lb < rb:
      cnt = 0
      for i in range(lb, rb):
        if history[i].predict in incontext:
          cnt += 1
      # emperical skew
      p_ = float(cnt) / (rb - lb)
      # corrected skew according to accuracy
      p = (p_ - (1-acc_p)*n/(N-1)) / (acc_p - (1-acc_p)/(N-1))
      dbg.dbg("normal history (%s-%s): %s -> %s" % (lb, rb, p_, p))
      if p > 1: p = 1.
      if p < 0: p = 0.
      incontext_cnt += (rb - lb) * p
    # second compute the incontext skew of the segment by special classifier
    lb = rb
    rb = length
    if lb < rb:
      # get params of special classifier
      acc_in, err_in_out, acc_out = self.compute_special_model_param(n)
      cascade_cnt = 0
      for i in range(lb, rb):
        if history[i].cascaded: cascade_cnt += 1
      cascade_rate = float(cascade_cnt) / (rb - lb)
      # corrected skew from cascade rate
      p = (acc_out - cascade_rate) / (acc_out - err_in_out)
      dbg.dbg("special history (%s-%s): %s -> %s" % (lb, rb, 1-cascade_rate, p))
      if p > 1: p = 1.
      if p < 0: p = 0.
      incontext_cnt += (rb - lb) * p
    skew = float(incontext_cnt) / length
    return skew
