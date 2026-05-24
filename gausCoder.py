from dataclasses import dataclass
import struct
from typing import Any
import zlib
import io

import numpy as np

from coderUtils import *
from util_gau import GaussianData

FLAGS_HILBERT = 0x01
FLAGS_QUANTIZED = 0x02

class GaussEncoder:
    MAGIC = b'GCZ1'

    def __init__(self, bits=None):
        self.bits = bits
        if bits is not None:
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
    
    @staticmethod
    def _prune_by_opacity(gau: GaussianData, threshold: float) -> GaussianData:
        mask = gau.opacity[:, 0] > threshold
        return GaussianData(
            gau.xyz[mask], gau.rot[mask], gau.scale[mask],
            gau.opacity[mask], gau.sh[mask]
        )

    def encode(self, gau: GaussianData, args: dict[str, Any], out_path: str):
        flags = 0

        if args.get("po") is not None:
            gau = self._prune_by_opacity(gau, args["po"])

        if args.get("ps") is not None:
            ...

        if args.get("hilbert"):
            order = hilbert_sort(gau.xyz)
            gau = apply_order(gau, order)
            flags |= FLAGS_HILBERT

        raw_attrs = [gau.xyz, gau.rot, gau.scale, gau.opacity, gau.sh]
        if flags & FLAGS_HILBERT:
            attrs_delta = [delta_encode(a) for a in raw_attrs]
        else:
            attrs_delta = raw_attrs

        n = len(gau.xyz)
        sh_dim = gau.sh.shape[-1]
        bits = self.bits if self.bits is not None else 0

        if self.bits is not None:
            flags |= FLAGS_QUANTIZED

        buf = bytearray()
        buf += self.MAGIC
        buf += struct.pack('<IHBB', n, sh_dim, bits, flags)
        if self.bits is not None:
            quants = [self._quantize(a) for a in attrs_delta]
            for q, mn, mx in quants:
                buf += struct.pack('<ff', mn, mx)
            for q, mn, mx in quants:
                buf += q.tobytes()
        else:
            for a in attrs_delta:
                buf += a.astype(np.float32).tobytes()

        use_zlib = args.get("hilbert") or args.get("compress")
        out = zlib.compress(bytes(buf), level=6) if use_zlib else bytes(buf)
        with open(out_path, 'wb') as f:
            f.write(out)


class GaussDecoder:
    @staticmethod
    def decode(in_path: str) -> GaussianData:
        MAGIC = b'GCZ1'
        with open(in_path, 'rb') as f:
            raw = f.read()
            data = raw if raw[:4] == MAGIC else zlib.decompress(raw)
            f = io.BytesIO(data)
            magic = f.read(4)

            assert magic == MAGIC, f"Bad magic: {magic}"
            n, sh_dim, bits, flags = struct.unpack('<IHBB', f.read(8))

            shapes = [(n, 3), (n, 4), (n, 3), (n, 1), (n, sh_dim)]
            arrays = []

            if flags & FLAGS_QUANTIZED:
                dtype = np.uint8 if bits == 8 else np.uint16
                levels = (1 << bits) - 1
                mins_maxs = [struct.unpack('<ff', f.read(8)) for _ in range(5)]
                for (mn, mx), shape in zip(mins_maxs, shapes):
                    count = shape[0] * shape[1]
                    raw = np.frombuffer(f.read(count * np.dtype(dtype).itemsize), dtype=dtype)
                    dequant = (raw.astype(np.float32) / levels) * (mx - mn) + mn
                    arr = dequant.reshape(shape)
                    arrays.append(delta_decode(arr) if flags & FLAGS_HILBERT else arr)
            else:
                for shape in shapes:
                    count = shape[0] * shape[1]
                    raw = np.frombuffer(f.read(count * 4), dtype=np.float32)
                    arr = raw.reshape(shape)
                    arrays.append(delta_decode(arr) if flags & FLAGS_HILBERT else arr)

        xyz, rots, scales, opacities, shs = arrays

        # Apply activations
        rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
        scales = np.exp(scales)
        opacities = 1.0 / (1.0 + np.exp(-opacities))

        return GaussianData(xyz, rots, scales, opacities, shs)
    
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    parser = argparse.ArgumentParser()
    set_parser(parser)
    args = parser.parse_args()
    check_args(args)

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

        encoder = GaussEncoder(bits=args.q)
        encoder.encode(gauss, vars(args), str(new_path))

    root.destroy()