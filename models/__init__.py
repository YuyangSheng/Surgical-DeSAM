# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_seg import build


def build_model(args):
    return build(args)
