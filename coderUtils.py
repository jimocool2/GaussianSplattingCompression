import numpy as np
from dataclasses import dataclass
from util_gau import GaussianData
from plyfile import PlyData

# ---------- Utility Functions ---------

def hilbert_sort(xyz: np.ndarray) -> np.ndarray:
    from hilbertcurve.hilbertcurve import HilbertCurve
    p = 16  # 16 bits per dimension, 2^16 grid levels
    hc = HilbertCurve(p, n=3)

    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    scale = mx - mn
    scale[scale == 0] = 1.0

    grid = ((xyz - mn) / scale * ((1 << p) - 1)).clip(0, (1 << p) - 1).astype(np.int64)
    distances = hc.distances_from_points(grid.tolist())
    return np.argsort(np.fromiter(distances, dtype=int))


def apply_order(gau: GaussianData, order: np.ndarray) -> GaussianData:
    return GaussianData(
        gau.xyz[order], gau.rot[order], gau.scale[order],
        gau.opacity[order], gau.sh[order]
    )

def delta_encode(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr)
    out[0] = arr[0]
    out[1:] = np.diff(arr, axis=0)
    return out


def delta_decode(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(arr, axis=0)

# ---------- Vector Quantization ---------

def vq_idx_dtype(k):
    # 1 byte per index up to 256 codewords, else 2 bytes
    return (np.uint8, 1) if k <= 256 else (np.uint16, 2)


def _vq_assign(data, cb, mem_cap=8_000_000):
    # nearest-codeword index per row, in memory-bounded batches
    k = len(cb)
    batch = max(1024, mem_cap // max(k, 1))
    cb_sq = np.einsum('ij,ij->i', cb, cb)
    labels = np.empty(len(data), dtype=np.int64)
    for s in range(0, len(data), batch):
        x = data[s:s + batch]
        d = cb_sq[None, :] - 2.0 * (x @ cb.T)  # drop ||x||^2 (constant per row)
        labels[s:s + batch] = d.argmin(axis=1)
    return labels


def vq_fit(data, k, iters=12, sample=120_000, seed=0):
    # Lloyd k-means trained on a subsample, then assign all rows.
    # Returns (codebook (K,d) float32, labels (n,) int64). K = min(k, n).
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data.shape
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(seed)
    train = data if n <= sample else data[rng.choice(n, sample, replace=False)]
    cb = np.ascontiguousarray(train[rng.choice(len(train), k, replace=False)])
    for _ in range(iters):
        lbl = _vq_assign(train, cb)
        counts = np.bincount(lbl, minlength=k).astype(np.float32)
        sums = np.empty((k, d), dtype=np.float32)
        for c in range(d):
            sums[:, c] = np.bincount(lbl, weights=train[:, c], minlength=k)
        nz = counts > 0
        new_cb = cb.copy()
        new_cb[nz] = sums[nz] / counts[nz, None]
        empty = np.where(~nz)[0]
        if empty.size:
            new_cb[empty] = train[rng.integers(0, len(train), empty.size)]
        moved = np.abs(new_cb - cb).max()
        cb = new_cb
        if moved < 1e-6:
            break
    return cb, _vq_assign(data, cb)


def vq_encode_sh(sh, k_color, k_rest, seed=0):
    # Split sh into DC color (first 3 cols) and SH rest; one codebook each.
    sh = np.ascontiguousarray(sh, dtype=np.float32)
    cb_dc, lbl_dc = vq_fit(sh[:, :3], k_color, seed=seed)
    if sh.shape[1] > 3:
        cb_rest, lbl_rest = vq_fit(sh[:, 3:], k_rest, seed=seed + 1)
    else:
        cb_rest = np.zeros((0, 0), dtype=np.float32)
        lbl_rest = np.zeros(len(sh), dtype=np.int64)
    return cb_dc, lbl_dc, cb_rest, lbl_rest


def vq_decode_sh(cb_dc, lbl_dc, cb_rest, lbl_rest):
    dc = cb_dc[lbl_dc]
    if cb_rest.shape[0] > 0:
        return np.concatenate([dc, cb_rest[lbl_rest]], axis=1).astype(np.float32)
    return dc.astype(np.float32)

# ---------- Parser Functions ---------

def set_parser(parser):
    parser.add_argument(
        "-q",
        nargs="?",
        const=8,
        default=None,
        type=int,
        choices=[8, 16],
        help="Quantization bits (8 or 16). Omit to skip quantization."
    )

    parser.add_argument(
        "-hb",
        action="store_true",
        default=False,
        dest="hilbert",
        help="Sort Gaussians along Hilbert curve before encoding"
    )

    parser.add_argument(
        "-po",
        type=float,
        nargs="?",
        const=0.08,
        default=None,
        help="Opacity pruning threshold (0.0001 - 1.0)"
    )

    parser.add_argument(
        "-vq",
        type=int,
        nargs="?",
        const=4096,
        default=None,
        help="Vector quantize color (f_dc) and SH (f_rest); value = codebook size (default 4096)."
    )

    parser.add_argument(
        "-compress",
        action="store_true",
        default=False,
        dest="compress",
        help="Compress output with zlib (without delta coding)"
    )
    
def check_args(args):
    if args.po is not None and not (0.0001 <= args.po <= 1.0):
        raise ValueError("Opacity pruning threshold must be between 0.0001 and 1.0")
    if args.vq is not None and args.vq < 1:
        raise ValueError("VQ codebook size must be >= 1")

# ---------- Helper Classes ---------

@dataclass
class RawGaussianData(GaussianData):
    pass

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