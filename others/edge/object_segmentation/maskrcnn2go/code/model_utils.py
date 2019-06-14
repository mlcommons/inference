from __future__ import absolute_import, division, print_function, unicode_literals

import os
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace


def create_blobs_if_not_existed(blob_names):
    existd_names = set(workspace.Blobs())
    for xx in blob_names:
        if xx not in existd_names:
            workspace.CreateBlob(str(xx))


def load_model_pb(net_file, init_file=None, is_run_init=True, is_create_net=True):
    net = core.Net("net")
    if net_file is not None:
        net.Proto().ParseFromString(open(net_file, "rb").read())

    if init_file is None:
        fn, ext = os.path.splitext(net_file)
        init_file = fn + "_init" + ext

    init_net = caffe2_pb2.NetDef()
    init_net.ParseFromString(open(init_file, "rb").read())

    if is_run_init:
        workspace.RunNetOnce(init_net)
        create_blobs_if_not_existed(net.external_inputs)
        if net.Proto().name == "":
            net.Proto().name = "net"
    if is_create_net:
        workspace.CreateNet(net)

    return (net, init_net)
