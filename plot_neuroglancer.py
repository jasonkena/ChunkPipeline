import argparse
import numpy as np

import neuroglancer
import h5py
import imageio

ip = "localhost"  # or public IP of the machine for sharable display
port = 8889
neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
viewer = neuroglancer.Viewer()

# SNEMI (# 3d vol dim: z,y,x)
res = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"], units=["nm", "nm", "nm"], scales=[30, 64, 64]
)

print("load im and gt segmentation")


def ngLayer(data, res, oo=[0, 0, 0], tt="segmentation"):
    return neuroglancer.LocalVolume(
        data, dimensions=res, volume_type=tt, voxel_offset=oo
    )


seg = h5py.File("r0.h5").get("seg")[:]
im = h5py.File("r0.h5").get("original")[:]
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
    s.layers.append(name="im", layer=ngLayer(im, res, tt="image"))
    # s.layers.append(name='im',layer=ngLayer(im,res,tt='image'))
    # s.layers.append(name='gt',layer=ngLayer(gt,res,tt='segmentation'))
    # s.layers.append(name="seg", layer=ngLayer(gt, res, tt="segmentation"))
    s.layers.append(name="seg", layer=ngLayer(seg, res, tt="segmentation"))

print(viewer)
