"""Reader for prebuilt graph directories (feat.npy, pos.npy, cell_class.npy, cell_ID.npy, offsets.npy, infos.pkl, meta.json)."""
import json
import os

import numpy as np
import torch

from orbis.data.holder import to_batch, to_device


def read_infos(d):
    """The dataset summary stored in d/infos.pkl (num_genes, num_cell_class, cell_class_decoder, ...); a trusted local file."""
    return torch.load(os.path.join(d, "infos.pkl"), map_location="cpu", weights_only=False)


class PrebuiltGraphs:
    """Memory-mapped graphs of a prebuilt directory; graph j holds rows offsets[j]:offsets[j + 1]."""

    def __init__(self, d):
        self.dir = d
        self.feat = np.load(os.path.join(d, "feat.npy"), mmap_mode="r")
        self.pos = np.load(os.path.join(d, "pos.npy"), mmap_mode="r")
        self.cc = np.load(os.path.join(d, "cell_class.npy"), mmap_mode="r")
        self.cid = np.load(os.path.join(d, "cell_ID.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(d, "offsets.npy"))
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.pos_dim = int(self.meta.get("pos_dim", self.pos.shape[1]))

    def __len__(self):
        return len(self.offsets) - 1

    def get_batch(self, j, dev):
        """Graph j as a dense, masked and centred one-graph DataHolder on dev."""
        s, e = int(self.offsets[j]), int(self.offsets[j + 1])
        return to_device(to_batch(torch.from_numpy(np.ascontiguousarray(self.feat[s:e])).float(),
                                  torch.from_numpy(np.ascontiguousarray(self.pos[s:e])).float(),
                                  torch.from_numpy(np.ascontiguousarray(self.cc[s:e])),
                                  torch.from_numpy(np.ascontiguousarray(self.cid[s:e]))), dev)
