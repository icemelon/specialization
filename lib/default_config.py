import os
import ConfigParser

import env

def obj_default_config(target_list=None):
    config = ConfigParser.ConfigParser()
    config.add_section("data")
    config.set("data", "train_image_root", env.imagenet_train_dir)
    config.set("data", "val_image_root", env.imagenet_val_dir)
    config.set("data", "train_label", env.imagenet_train_label)
    config.set("data", "val_label", env.imagenet_val_label)

    config.add_section("train")
    config.set("train", "target_layer", "relu7")
    config.set("train", "target_blob", "fc7")
    if target_list:
        config.set("train", "target_labels", ",".join([str(e) for e in target_list]))
    else:
        config.set("train", "target_labels", "")
        config.set("train", "item_per_class", "100")
        config.set("train", "top", env.train_template)
        config.set("train", "test", env.test_template)

    # init best_model and compact_model
    set_obj_googlenet(config, "best_model", "googlenet")
    set_obj_alexnet(config, "compact_model", "")
    return config

def set_obj_vgg(config, section, model):
    if config.has_section(section):
        config.remove_section(section)
    config.add_section(section)
    config.set(section, "name", model)
    config.set(section, "image_dim", "224")
    config.set(section, "raw_scale", "255")
    config.set(section, "mean", "103.939,116.779,123.68")

def set_obj_googlenet(config, section, model):
    if config.has_section(section):
        config.remove_section(section)
    config.add_section(section)
    config.set(section, "name", model)
    config.set(section, "image_dim", "224")
    config.set(section, "raw_scale", "255")
    config.set(section, "mean", "104,117,123")

def set_obj_alexnet(config, section, model):
    if config.has_section(section):
        config.remove_section(section)
    config.add_section(section)
    config.set(section, "name", model)
    config.set(section, "image_dim", "227")
    config.set(section, "raw_scale", "255")
    config.set(section, "mean", os.path.join(env.pretrained_model_dir, "alexnet_mean.binaryproto"))

def scene_default_config(target_list=None):
    config = ConfigParser.ConfigParser()
    config.add_section("data")
    config.set("data", "train_image_root", env.mitplaces_train_dir)
    config.set("data", "val_image_root", env.mitplaces_val_dir)
    config.set("data", "train_label", env.mitplaces_train_label)
    config.set("data", "val_label", env.mitplaces_val_label)

    config.add_section("train")
    config.set("train", "target_layer", "relu7")
    config.set("train", "target_blob", "fc7")
    if target_list:
        config.set("train", "target_labels", ",".join([str(e) for e in target_list]))
    else:
        config.set("train", "target_labels", "")
    config.set("train", "item_per_class", "100")
    config.set("train", "top", env.train_template)
    config.set("train", "test", env.test_template)

    # init best_model and compact_model
    set_scene_vgg(config, "best_model", "scene_vgg16")
    set_scene_alexnet(config, "compact_model", "")
    return config

def set_scene_vgg(config, section, model):
    if config.has_section(section):
        config.remove_section(section)
    config.add_section(section)
    config.set(section, "name", model)
    config.set(section, "image_dim", "224")
    config.set(section, "raw_scale", "255")
    config.set(section, "mean", "105.4878,113.7411,116.0604")

def set_scene_alexnet(config, section, model):
    if config.has_section(section):
        config.remove_section(section)
    config.add_section(section)
    config.set(section, "name", model)
    config.set(section, "image_dim", "227")
    config.set(section, "raw_scale", "255")
    config.set(section, "mean", os.path.join(env.pretrained_model_dir, "places205_mean.binaryproto"))

def face_default_config(target_list=None):
    config = ConfigParser.ConfigParser()
    config.add_section("data")
    config.set("data", "train_image_root", env.msrface_train_dir)
    config.set("data", "val_image_root", env.msrface_val_dir)
    config.set("data", "train_label", env.msrface_train_label)
    config.set("data", "val_label", env.msrface_val_label)

    config.add_section("train")
    config.set("train", "target_layer", "relu6")
    config.set("train", "target_blob", "fc6")
    if target_list:
        config.set("train", "target_labels", ",".join([str(e) for e in target_list]))
    else:
        config.set("train", "target_labels", "")
    config.set("train", "item_per_class", "200")
    config.set("train", "top", env.train_template)
    config.set("train", "test", env.test_template)
    config.set("train", "step_epoch", "20")
    config.set("train", "max_epoch", "60")
    config.set("train", "base_lr", "0.01")

    # init best_model and compact_model
    set_face_vgg(config, "best_model", "vgg_face_msr201")
    set_face_compact(config, "compact_model", "")
    return config

def set_face_vgg(config, section, model):
    if config.has_section(section):
        config.remove_section(section)
    config.add_section(section)
    config.set(section, "name", model)
    config.set(section, "image_dim", "224")
    config.set(section, "raw_scale", "255")
    config.set(section, "mean", "93.5940,104.7624,129.1863")

def set_face_compact(config, section, model):
    if config.has_section(section):
        config.remove_section(section)
    config.add_section(section)
    config.set(section, "name", model)
    config.set(section, "image_dim", "152")
    config.set(section, "raw_scale", "255")
    config.set(section, "mean", "99.5503,115.7630,151.2761")
