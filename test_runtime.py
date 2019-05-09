import os
import sys
import caffe
import argparse
import cPickle as pickle

import lib
from lib import trace_pb2 as tpb
from lib import dbg, RuntimeClassifier

default_error_tolerance = {
    'obj': 0.05,
    'scene': 0.05,
    'face': -0.07
}

default_datadir = {
    'obj': lib.env.imagenet_val_dir,
    'scene': lib.env.mitplaces_val_dir,
    'face': lib.env.msrface_val_dir,
}
  

def test_synthesized_data(args, cl, outf):
  datadir = default_datadir[args.task]
  with open(args.input, 'rb') as f:
    trace = pickle.load(f)
  if args.model is not None:
    dbg.dbg("compact model is %s" % args.model)
    cl.set_compact_model(args.model)
  correct = 0
  total = 0
  for frame, fn, lbl in trace:
    total += 1
    lbl = int(lbl)
    img = caffe.io.load_image(os.path.join(datadir, fn))
    predict = cl.predict(frame, img)
    cl.history[-1].label = lbl
    if predict == lbl: correct += 1
    if args.frame is not None and total > args.frame: break
  accuracy = float(correct) / total
  avg_prepare_time, avg_forward_time = cl.get_avg_latency()
  # output the trace
  trace = tpb.Trace()
  trace.points.extend(cl.history)
  trace.accuracy = accuracy
  trace.avg_prepare_time = avg_prepare_time
  trace.avg_forward_time = avg_forward_time
  if outf is None:
    print(str(trace))
  else:
    with open(outf, 'w') as f:
      f.write(str(trace))

def test_video(args, cl, outf):
  total = 0
  correct = 0
  class_acc = {}
  if args.model is not None:
    dbg.dbg("compact model is %s" % args.model)
    cl.set_compact_model(args.model)
  with open(args.input) as f:
    for line in f:
      t = line.strip().split()
      frame = int(t[0])
      #if frame % 4 != 0: continue
      lbl = int(t[2])
      img = caffe.io.load_image(t[1])
      predict = cl.predict(frame, img)
      cl.history[-1].label = lbl
      c = (predict == lbl)
      total += 1
      if lbl not in class_acc: class_acc[lbl] = [0, 0]
      class_acc[lbl][1] += 1
      if c:
        correct += 1
        class_acc[lbl][0] += 1
      if args.frame is not None and total > args.frame: break
  accuracy = float(correct) / total
  avg_prepare_time, avg_forward_time = cl.get_avg_latency()
  # output the trace
  trace = tpb.Trace()
  trace.points.extend(cl.history)
  trace.accuracy = accuracy
  trace.avg_prepare_time = avg_prepare_time
  trace.avg_forward_time = avg_forward_time
  if outf is None:
    print(str(trace))
  else:
    with open(outf, 'w') as f:
      f.write(str(trace))
  # output the per-class accuracy
  dbg.dbg('per-class accuracy')
  for lbl in sorted(class_acc.keys()):
    correct, total = class_acc[lbl]
    acc = float(correct) / total
    dbg.dbg('%s: %s (%s/%s)' % (lbl, acc, correct, total))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--type', choices=['syn', 'video'], help='type of input data: synthesized / video data)')
  parser.add_argument('--task', choices=['obj', 'scene', 'face'], help='recognition task: obj/scene/face')
  parser.add_argument('-i', '--input', help='specify the input')
  parser.add_argument('-o', '--output', help='specify the output trace file')
  parser.add_argument('--model', default=None, help='set the special model (default: None)')
  parser.add_argument('--frame', type=int, help='set the max number of frames to process; '+
                      'if not set, process all frames in the input')
  parser.add_argument('--cpu', action='store_true', help='use cpu in testing')
  parser.add_argument('--window', choices=['on', 'off'], default='on', 
                      help='turn on/off window exploration (default: on)')
  parser.add_argument('--flexexit', choices=['on', 'off'], default='on', 
                      help='turn on/off flexible exit specialization (default: on)')
  parser.add_argument('--train_skew', choices=['fixed', 'dynamic'], default='fixed',
                      help='train skew is fixed or dynamic (default: fixed)')
  args = parser.parse_args()
  if args.type is None:
    print('Missing type')
    exit()
  if args.input is None:
    print('Missing input')
    exit()
  if args.task is None:
    print('Missing task')
    exit()
  config = {
    'cpu': args.cpu,
    'error_tolerance': default_error_tolerance[args.task],
  }
  cl = RuntimeClassifier(args.task, **config)
  if args.type == 'syn':
    test_synthesized_data(args, cl, args.output)
  else:
    test_video(args, cl, args.output)    
