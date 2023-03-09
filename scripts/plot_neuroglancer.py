import argparse
import numpy as np

import neuroglancer
import h5py
import imageio
import zarr

ip = "localhost"  # or public IP of the machine for sharable display
port = 8900
neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
viewer = neuroglancer.Viewer()

# SNEMI (# 3d vol dim: z,y,x)
res = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"],
    units=["nm", "nm", "nm"],
    # scales=[30, 6, 6]
    scales=[30, 8, 8]
    # names=["z", "y", "x"], units=["nm", "nm", "nm"], scales=[30, 6, 6]
    # names=["z", "y", "x"], units=["nm", "nm", "nm"], scales=[30, 64, 64]
)

print("load im and gt segmentation")


def ngLayer(data, res, oo=[0, 0, 0], tt="segmentation"):
    return neuroglancer.LocalVolume(
        data, dimensions=res, volume_type=tt, voxel_offset=oo
    )

def get_binary(raw, spine):
    raw = raw.copy().astype(np.uint16)
    spine = spine.copy()
    max_idx = np.max(raw)

    raw += ((spine>0) * (max_idx + 1)).astype(raw.dtype)

    return raw


# seg = h5py.File("r0.h5").get("seg")[:]
# seg = h5py.File("/home/jason/Downloads/snemisubmissions/human/test-input.h5").get("main")[:]
# im = h5py.File("/mmfs1/data/adhinart/foundation/zebrafinch_im_80nm.h5").get("main")[:]
# im = h5py.File("/mmfs1/data/adhinart/foundation/mouse_microns-phase1_64nm.h5").get("main")[:]
# im = h5py.File("/mmfs1/data/adhinart/foundation/mouse_moritz2019_88nm.h5").get("main")[:]
# seg = zarr.open_group("/mmfs1/data/adhinart/dendrite/data/foundation_mouse_microns")["seg"]["seg"][:]
# seg = zarr.open_group("/mmfs1/data/adhinart/dendrite/data/foundation_mouse_moritz")["seg"]["seg"][:]

#im = h5py.File("/mmfs1/data/adhinart/dendrite/raw/human_raw.h5").get("main")[:]


#
# raw = h5py.File("/mmfs1/data/adhinart/dendrite/raw/seg_den_raw.h5").get("main")[:,:4000,:4000]
# raw = raw[::4,::4,::4]
# print("done loading raw")
# spine = h5py.File("/mmfs1/data/adhinart/dendrite/raw/seg_den_spine.h5").get("main")[:, :4000, :4000]
# spine = spine[::4,::4,::4]
# print("done loading spine")
# inst = h5py.File("/mmfs1/data/adhinart/dendrite/raw/seg_den_seg.h5").get("main")[:, :4000, :4000]
# inst = inst[::4,::4,::4]
# binary = get_binary(raw, spine)
#
raw = h5py.File("/mmfs1/data/adhinart/dendrite/raw/mouse_raw.h5").get("main")[:]
raw = raw[::4,::4,::4]
print("done loading raw")
spine = h5py.File("/mmfs1/data/adhinart/dendrite/raw/mouse_spine.h5").get("main")[:]
spine = spine[::4,::4,::4]
print("done loading spine")
# inst = h5py.File("/mmfs1/data/adhinart/dendrite/raw/mouse_seg.h5").get("main")[:]
# inst = inst[::4,::4,::4]
binary = get_binary(raw, spine)
#seg = h5py.File("/mmfs1/data/adhinart/dendrite/raw/human_seg.h5").get("main")[:]

#im = h5py.File("/mmfs1/data/adhinart/vesicle/new_im_vesicle/data.h5").get("im")[:]
#seg = h5py.File("/mmfs1/data/adhinart/vesicle/new_im_vesicle/data.h5").get("seg")[:]

# im = h5py.File("r0.h5").get("original")[:]
# seg = h5py.File("inferred_1.h5").get("main")[:].astype(np.uint8)
# seg = h5py.File("/home/jason/Downloads/mouse.h5").get("seg")[:].astype(np.uint8)
# seg = (
#     h5py.File("/home/jason/Downloads/mouse.h5")
#     .get("main")[:300, :2000, :2000]
#     .astype(np.uint8)
# )
# seg = h5py.File("/home/jason/Downloads/mouse.h5").get("main")[:].astype(np.uint8)
# seg = h5py.File("/mnt/andromeda/dendrite/human/baseline/seg_1.h5").get("seg")[:].astype(np.uint8)
# seg = h5py.File("/mnt/andromeda/dendrite/den_seg/baseline/seg_20.h5").get("seg")[:].astype(np.uint8)
# seg = h5py.File("seg_2.h5").get("seg")[:].astype(np.uint8)
# full = h5py.File("14.h5").get("main")[:].astype(np.uint8)
# gt = (
#     h5py.File("seg_den_6nm.h5")
#     .get("main")[1070:1332, 7712:8624, 4640:5088]
#     .astype(np.uint8)
# )

# seg = np.load("seg.npy").astype(np.uint8)
# im = imageio.volread("snemi/image/"+'train-input.tif')
# with h5py.File("train-labels.h5", 'r') as fl:
#     gt = np.array(fl['main']).astype(np.uint8)

with viewer.txn() as s:
    #s.layers.append(name="im", layer=ngLayer(im, res, tt="image"))
    # s.layers.append(name='im',layer=ngLayer(im,res,tt='image'))
    # s.layers.append(name='gt',layer=ngLayer(gt,res,tt='segmentation'))
    # s.layers.append(name="seg", layer=ngLayer(gt, res, tt="segmentation"))
    s.layers.append(name="bin", layer=ngLayer(binary, res, tt="segmentation"))
    s.layers.append(name="seg", layer=ngLayer(raw, res, tt="segmentation"))
    # s.layers.append(name="inst", layer=ngLayer(inst, res, tt="segmentation"))
    # Dd = "precomputed://https://rhoana.rc.fas.harvard.edu/ng/R0/im_64nm/"
    # s.layers["image"] = neuroglancer.ImageLayer(source=Dd)

print(viewer)
