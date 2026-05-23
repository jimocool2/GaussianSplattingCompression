from dataclasses import dataclass
import struct

import numpy as np

from plyfile import PlyData
from util_gau import GaussianData

@dataclass
class RawGaussianData(GaussianData):
    pass

class GaussEncoder:
    MAGIC = b'GCZ1'

    def __init__(self, bits=8):
        self.bits = bits
        self.dtype = np.uint8 if bits == 8 else np.uint16
        self.levels = (1 << bits) - 1

    def _quantize(self, arr):
        mn = arr.min().astype(np.float32)
        mx = arr.max().astype(np.float32)
        scale = mx - mn
        if scale == 0:
            scale = 1.0
        q = np.round((arr - mn) / scale * self.levels).clip(0, self.levels).astype(self.dtype)
        return q, mn, mx

    def encode(self, gau: GaussianData, out_path: str):
        attrs = [gau.xyz, gau.rot, gau.scale, gau.opacity, gau.sh]
        quants = [self._quantize(a) for a in attrs]
        n = len(gau.xyz)
        sh_dim = gau.sh.shape[-1]
        with open(out_path, 'wb') as f:
            f.write(self.MAGIC)
            f.write(struct.pack('<IHBx', n, sh_dim, self.bits))
            for q, mn, mx in quants:
                f.write(struct.pack('<ff', mn, mx))
            for q, mn, mx in quants:
                f.write(q.tobytes())


class GaussDecoder:
    @staticmethod
    def decode(in_path: str) -> GaussianData:
        MAGIC = b'GCZ1'
        with open(in_path, 'rb') as f:
            magic = f.read(4)
            assert magic == MAGIC, f"Bad magic: {magic}"
            n, sh_dim, bits = struct.unpack('<IHBx', f.read(8))
            dtype = np.uint8 if bits == 8 else np.uint16
            levels = (1 << bits) - 1
            mins_maxs = [struct.unpack('<ff', f.read(8)) for _ in range(5)]
            shapes = [(n, 3), (n, 4), (n, 3), (n, 1), (n, sh_dim)]
            arrays = []
            for (mn, mx), shape in zip(mins_maxs, shapes):
                count = shape[0] * shape[1]
                raw = np.frombuffer(f.read(count * np.dtype(dtype).itemsize), dtype=dtype)
                dequant = (raw.astype(np.float32) / levels) * (mx - mn) + mn
                arrays.append(dequant.reshape(shape))
            xyz, rots, scales, opacities, shs = arrays

            rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
            scales = np.exp(scales)
            opacities = 1.0 / (1.0 + np.exp(-opacities))  # sigmoid

            return GaussianData(xyz, rots, scales, opacities, shs)

class GaussLoader:
    @staticmethod
    def load_ply_raw(file_path):
        max_sh_degree = 3
        plydata = PlyData.read(file_path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
        features_extra = np.transpose(features_extra, [0, 2, 1])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        shs = np.concatenate([features_dc.reshape(-1, 3),
                               features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)

        # raw: no exp(scale), no sigmoid(opacity), no rot normalization
        return RawGaussianData(xyz, rots, scales, opacities, shs)
    
    

def set_parser(parser):
    parser.add_argument(
        "-q",
        nargs="?",
        const=8,
        default=8,
        type=int,
        choices=[8, 16],
        help="Number of bits for quantization"
    )

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    parser = argparse.ArgumentParser()
    set_parser(parser)
    args = parser.parse_args()

    root = Tk()
    root.withdraw()

    # Open file picker
    selected_file  = askopenfilename(
        filetypes=[("Ply files", "*.ply")]
    )
    print(selected_file)

    if selected_file and selected_file != '':
        file_path = Path(selected_file)

        new_path = file_path.with_suffix(".ply.comp")

        gauss = GaussLoader.load_ply_raw(str(file_path))

        encoder = GaussEncoder()
        encoder.encode(gauss, str(new_path))

    root.destroy()