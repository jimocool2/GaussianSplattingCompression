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
        "-ps",
        type=float,
        nargs="?",
        const=0.0025,
        default=None,
        help="Size pruning threshold"
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
    if args.ps is not None and args.ps < 0:
        raise ValueError("Size pruning threshold must be non-negative")

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