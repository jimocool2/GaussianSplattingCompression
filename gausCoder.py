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
FLAGS_VQ = 0x04

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
        # Opacity is stored as logits, so apply sigmoid before thresholding.
        mask = 1.0 / (1.0 + np.exp(-gau.opacity))[:, 0] > threshold
        return GaussianData(
            gau.xyz[mask], gau.rot[mask], gau.scale[mask],
            gau.opacity[mask], gau.sh[mask]
        )

    def encode(self, gau: GaussianData, args: dict[str, Any], out_path: str):
        flags = 0

        if args.get("po") is not None:
            gau = self._prune_by_opacity(gau, args["po"])

        if args.get("hilbert"):
            order = hilbert_sort(gau.xyz)
            gau = apply_order(gau, order)
            flags |= FLAGS_HILBERT

        # Vector quantization of color (f_dc) + SH (f_rest). Runs after pruning
        # and sorting, so labels follow the final Gaussian order.
        use_vq = args.get("vq") is not None
        if use_vq:
            flags |= FLAGS_VQ
            k = int(args["vq"])
            cb_dc, lbl_dc, cb_rest, lbl_rest = vq_encode_sh(gau.sh, k, k)
            raw_attrs = [gau.xyz, gau.rot, gau.scale, gau.opacity]
        else:
            raw_attrs = [gau.xyz, gau.rot, gau.scale, gau.opacity, gau.sh]

        # Delta only the scalar attrs; VQ indices are not delta coded.
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

        # VQ block: codebooks stay float32 (small) and are independent of -q.
        if use_vq:
            _, nb_dc = vq_idx_dtype(len(cb_dc))
            k_rest = len(cb_rest)
            nb_rest = vq_idx_dtype(k_rest)[1] if k_rest > 0 else 1
            buf += struct.pack('<IIBB', len(cb_dc), k_rest, nb_dc, nb_rest)
            buf += np.ascontiguousarray(cb_dc, np.float32).tobytes()
            buf += lbl_dc.astype(np.uint8 if nb_dc == 1 else np.uint16).tobytes()
            if k_rest > 0:
                buf += np.ascontiguousarray(cb_rest, np.float32).tobytes()
                buf += lbl_rest.astype(np.uint8 if nb_rest == 1 else np.uint16).tobytes()

        # zlib also benefits VQ: spatially sorted neighbours reuse indices.
        use_zlib = args.get("hilbert") or args.get("compress") or use_vq
        out = zlib.compress(bytes(buf), level=6) if use_zlib else bytes(buf)
        with open(out_path, 'wb') as f:
            f.write(out)


class GaussDecoder:
    @staticmethod
    def decode(in_path: str) -> GaussianData:
        MAGIC = b'GCZ1'
        with open(in_path, 'rb') as fh:
            raw = fh.read()
        data = raw if raw[:4] == MAGIC else zlib.decompress(raw)
        f = io.BytesIO(data)
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        n, sh_dim, bits, flags = struct.unpack('<IHBB', f.read(8))

        if flags & FLAGS_VQ:
            shapes = [(n, 3), (n, 4), (n, 3), (n, 1)]
        else:
            shapes = [(n, 3), (n, 4), (n, 3), (n, 1), (n, sh_dim)]

        arrays = []
        if flags & FLAGS_QUANTIZED:
            dtype = np.uint8 if bits == 8 else np.uint16
            levels = (1 << bits) - 1
            mins_maxs = [struct.unpack('<ff', f.read(8)) for _ in range(len(shapes))]
            for (mn, mx), shape in zip(mins_maxs, shapes):
                count = shape[0] * shape[1]
                buf = np.frombuffer(f.read(count * np.dtype(dtype).itemsize), dtype=dtype)
                dequant = (buf.astype(np.float32) / levels) * (mx - mn) + mn
                arr = dequant.reshape(shape)
                arrays.append(delta_decode(arr) if flags & FLAGS_HILBERT else arr)
        else:
            for shape in shapes:
                count = shape[0] * shape[1]
                buf = np.frombuffer(f.read(count * 4), dtype=np.float32)
                arr = buf.reshape(shape)
                arrays.append(delta_decode(arr) if flags & FLAGS_HILBERT else arr)

        if flags & FLAGS_VQ:
            k_dc, k_rest, nb_dc, nb_rest = struct.unpack('<IIBB', f.read(10))
            dt_dc = np.uint8 if nb_dc == 1 else np.uint16
            dt_rest = np.uint8 if nb_rest == 1 else np.uint16
            cb_dc = np.frombuffer(f.read(k_dc * 3 * 4), dtype=np.float32).reshape(k_dc, 3)
            lbl_dc = np.frombuffer(f.read(n * nb_dc), dtype=dt_dc)
            if k_rest > 0:
                rest_dim = sh_dim - 3
                cb_rest = np.frombuffer(f.read(k_rest * rest_dim * 4), dtype=np.float32).reshape(k_rest, rest_dim)
                lbl_rest = np.frombuffer(f.read(n * nb_rest), dtype=dt_rest)
            else:
                cb_rest = np.zeros((0, 0), dtype=np.float32)
                lbl_rest = np.zeros(n, dtype=np.int64)
            sh = vq_decode_sh(cb_dc, lbl_dc, cb_rest, lbl_rest)
            xyz, rots, scales, opacities = arrays
        else:
            xyz, rots, scales, opacities, sh = arrays

        # Apply activations
        rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
        scales = np.exp(scales)
        opacities = 1.0 / (1.0 + np.exp(-opacities))

        return GaussianData(xyz, rots, scales, opacities, sh)

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