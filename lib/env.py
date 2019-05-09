import os
import ConfigParser

_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

pretrained_model_dir = os.path.join(_root_dir, "pretrained")
model_info_dir = os.path.join(_root_dir, "model_info")
special_model_dir = os.path.join(_root_dir, "special_models")

template_dir = os.path.join(_root_dir, "template")
train_template = os.path.join(template_dir, "train_template.prototxt")
test_template = os.path.join(template_dir, "test_template.prototxt")

stitch_network_bin = os.path.join(_root_dir, "caffe", "build", "tools", "stitch_network")
if not os.path.exists(stitch_network_bin):
    raise RuntimeError("Please compile caffe first")

_cfg = ConfigParser.ConfigParser()
_cfg.read(os.path.join(_root_dir, "env.cfg"))
imagenet_train_dir = os.path.expanduser(_cfg.get("imagenet", "train_dir"))
imagenet_val_dir = os.path.expanduser(_cfg.get("imagenet", "val_dir"))
imagenet_train_label = os.path.expanduser(_cfg.get("imagenet", "train_label"))
imagenet_val_label = os.path.expanduser(_cfg.get("imagenet", "val_label"))

mitplaces_train_dir = os.path.expanduser(_cfg.get("mitplaces", "train_dir"))
mitplaces_val_dir = os.path.expanduser(_cfg.get("mitplaces", "val_dir"))
mitplaces_train_label = os.path.expanduser(_cfg.get("mitplaces", "train_label"))
mitplaces_val_label = os.path.expanduser(_cfg.get("mitplaces", "val_label"))

msrface_train_dir = os.path.expanduser(_cfg.get("msrface", "train_dir"))
msrface_val_dir = os.path.expanduser(_cfg.get("msrface", "val_dir"))
msrface_train_label = os.path.expanduser(_cfg.get("msrface", "train_label"))
msrface_val_label = os.path.expanduser(_cfg.get("msrface", "val_label"))
