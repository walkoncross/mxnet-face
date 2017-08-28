import sys
import mxnet as mx
from symbol.resnet_new import *
from symbol.config import config
from symbol.processing import bbox_pred, clip_boxes, nms


def main(n_layers):
    if n_layers == 18:
        sym = resnet_18(num_class=2, is_train=False)
    elif n_layers == 34:
        sym = resnet_34(num_class=2, is_train=False)
    else:
        sym = resnet_50(num_class=2, is_train=False)

    # save_fn = './faster-resnet-%d-symbol-deploy.json' % n_layers
    save_fn = './faster-resnet-%d-symbol.json' % n_layers

    sym.save(save_fn)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_layers = int(sys.argv[1])
    else:
        n_layers = 50

    main(n_layers)
