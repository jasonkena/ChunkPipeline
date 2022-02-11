import cc3d
import edt
import h5py
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
from imu.io import get_bb_all3d
import os

import numpy as np

# NOTE: here naive separation between segments is used: each segment id is processed separately
# can potentially come up with a way to do it in a chunk-based manner instead of by segments
# will need to deal with chunk boundaries then


def pad_vol(vol, kernel_shape):
    assert torch.all(torch.tensor(kernel_shape) % 2 == 1)
    padded_vol = F.pad(
        vol,
        [
            *[kernel_shape[0] // 2] * 2,
            *[kernel_shape[1] // 2] * 2,
            *[kernel_shape[2] // 2] * 2,
        ],
    )
    return padded_vol


def get_boundary(vol):
    # vol [z,y,x] binary image (also works with ints, where 0s are background)
    # True means part of contour
    # kernel [z,y,x] binary image
    # if True, not eroded
    # z erode: whether to erode in the z-axis
    # NOTE: this does not care about anisotropy on purpose

    vol = torch.from_numpy(vol.astype(bool))
    padded_vol = pad_vol(vol, [3, 3, 3])
    boundary = torch.logical_and(
        F.max_pool3d((~padded_vol).float().unsqueeze(0), kernel_size=3, stride=1), vol
    ).squeeze(0)

    assert vol.shape == boundary.shape
    return boundary.numpy()


# adapted from https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def get_bbox(vol):
    # assuming z,y,x ordering
    zs = np.any(vol, axis=(1, 2))
    ys = np.any(vol, axis=(0, 2))
    xs = np.any(vol, axis=(0, 1))
    zmin, zmax = np.where(zs)[0][[0, -1]]
    ymin, ymax = np.where(ys)[0][[0, -1]]
    xmin, xmax = np.where(xs)[0][[0, -1]]

    return [zmin, zmax], [ymin, ymax], [xmin, xmax]


def get_dt(vol, anisotropy, black_border):
    assert (vol.flags["C_CONTIGUOUS"] + vol.flags["F_CONTIGUOUS"]) == 1

    dt = edt.edt(
        vol,
        anisotropy=anisotropy,
        black_border=black_border,
        order="C"
        if vol.flags["C_CONTIGUOUS"]
        else "F",  # depends if Fortran contiguous or not
        parallel=0,  # max CPU
    )
    return dt


def get_sphere_bounds(boundary_idx, vals, boundary_inverse, erode_delta, anisotropy):
    anisotropy = anisotropy[::-1]
    result = []
    for i in range(len(vals)):
        # z,y,x
        coords = [boundary_idx[j][boundary_inverse == i] for j in range(3)]
        # NOTE: may need to add +1 for good measure
        offsets = [math.ceil((vals[i] + erode_delta) / anisotropy[j]) for j in range(3)]

        result.append(
            [
                [max(0, np.min(coords[j]) - offsets[j]), np.max(coords[j]) + offsets[j]]
                for j in range(3)
            ]
        )
    return result


def sphere_expansion(vals, dt_boundary, sphere_bounds, erode_delta, anisotropy):
    result = np.zeros_like(dt_boundary, dtype=bool)
    print("Expanding spheres")
    for i in tqdm(range(len(vals))):
        zs, ys, xs = sphere_bounds[i]
        subvol = (
            dt_boundary[zs[0] : zs[1] + 1, ys[0] : ys[1] + 1, xs[0] : xs[1] + 1]
        ) != vals[i]
        # distance from center of spheres
        dt = get_dt(subvol, anisotropy, black_border=False)
        result[zs[0] : zs[1] + 1, ys[0] : ys[1] + 1, xs[0] : xs[1] + 1] = np.logical_or(
            result[zs[0] : zs[1] + 1, ys[0] : ys[1] + 1, xs[0] : xs[1] + 1],
            dt <= (vals[i] + erode_delta),
        )
    return result


def sphere_iteration(expanded, dt, vol, erode_delta, anisotropy):
    # expanded is the current sphere expansion to be dilated
    # dt is dt from air
    # vol is original vol
    boundary = get_boundary(expanded)
    boundary_idx = np.nonzero(boundary)
    vals, boundary_inverse = np.unique(dt[boundary], return_inverse=True)

    sphere_bounds = get_sphere_bounds(
        boundary_idx, vals, boundary_inverse, erode_delta, anisotropy
    )
    dt_boundary = dt * boundary
    expanded_sphere = sphere_expansion(
        vals, dt_boundary, sphere_bounds, erode_delta, anisotropy
    )
    expanded_sphere = np.logical_and(np.logical_or(expanded, expanded_sphere), vol)

    return expanded_sphere


def extract(
    vol, num_iter, max_erode, erode_delta, anisotropy=(6, 6, 30), connectivity=26
):
    # assert that volume is connected
    assert (
        cc3d.largest_k(
            vol,
            k=1,
            connectivity=connectivity,
            return_N=True,
        )[1]
        == 1
    )
    # anisotropy in nm
    # max_erode is how far from the boundary function extraction should occur
    dt = get_dt(vol, anisotropy, black_border=False)
    remaining = dt >= max_erode
    largest, N_remaining = cc3d.largest_k(
        remaining,
        k=1,
        connectivity=connectivity,
        return_N=True,
    )
    # TODO: assert that final segmentation is only composed of single CC
    expanded = largest
    for i in tqdm(range(num_iter)):
        expanded = sphere_iteration(expanded, dt, vol, erode_delta, anisotropy)

    others = np.logical_xor(vol, expanded)
    # segment the non trunks
    others, N_others = cc3d.connected_components(
        others, connectivity=connectivity, return_N=True
    )
    # relabel so that trunk is idx 1
    others[others > 0] += 1

    print(f"number of components in segmentation: {N_others+1}")
    seg = expanded + others

    return seg.astype(np.uint16)


def process_task(file, id, z1, z2, y1, y2, x1, x2):
    save_file = os.path.join("results", f"{id}.npy")
    if os.path.isfile(save_file):
        return
    vol = h5py.File(file).get("main")
    assert vol.dtype == np.uint16
    
    # offset in order to fix dt on boundaries
    # new start
    nz, ny, nx = [max(0, i-1) for i in [z1, y1, x1]]
    vol = vol[nz : z2 + 2, ny : y2 + 2, nx : x2 + 2]
    vol = np.ascontiguousarray(vol) == id

    seg = extract(vol, num_iter=1, max_erode=50, erode_delta=5, connectivity=26)
    # remove offset
    seg = seg[nz:nz+z2-z1+1, ny:ny+y2-y1+1, nx:nx+x2-x1+1]
    np.save(save_file, seg)


if __name__ == "__main__":
    # file = h5py.File("./train-labels.h5").get("main")
    # file = h5py.File("./den_s24_16nm.h5").get("main")
    file = h5py.File("./den_ruilin_v2_16nm.h5").get("main")

    __import__("pdb").set_trace()
    vol = (np.array(file[:])).astype(np.uint16)
    alpha = np.unique(vol)
    # vol = (np.array(file[:]) == 1).astype(np.uint16)
    import time

    time0 = time.time()
    bbox = get_bbox(vol)
    time1 = time.time()
    print(time1 - time0)
    imu_bbox = get_bb_all3d(vol)
    time2 = time.time()
    print(time2 - time1)
    __import__("pdb").set_trace()
    small_vol = vol[
        bbox[0][0] : bbox[0][1] + 1,
        bbox[1][0] : bbox[1][1] + 1,
        bbox[2][0] : bbox[2][1] + 1,
    ]
    small_vol = np.ascontiguousarray(small_vol)

    seg = extract(small_vol, num_iter=2, max_erode=50, erode_delta=5, connectivity=26)
    np.save("seg.npy", seg)
