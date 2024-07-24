import argparse
import numpy as np
import fastremap

import neuroglancer
import h5py
import imageio
import zarr
import cc3d

ip = "localhost"  # or public IP of the machine for sharable display
port = 8901
neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
viewer = neuroglancer.Viewer()

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

def get_inst(raw, seg, id):
    import cc3d
    raw = (raw == id).astype(np.uint16)
    spines = cc3d.connected_components(seg == id)
    raw = raw + spines
    print(list(range(1, np.max(raw) + 1)))

    return raw

def get_vol(dataset, id, downsample=[1, 4, 4]):
    assert dataset in ["human", "mouse", "seg_den"]
    anisotropy = [30, 6, 6] if dataset == "seg_den" else [30, 8, 8]
    anisotropy = [a*d for a, d in zip(anisotropy, downsample)]

    id = str(id)
    group = zarr.open_group(f"/mmfs1/data/adhinart/dendrite/data/{dataset}/point_cloud_{id}")

    idx = group["idx"][:]
    dtype = idx.dtype
    idx = idx.astype(float) / np.array(downsample)
    idx = np.floor(idx)
    idx = idx.astype(dtype)

    min_idx = np.min(idx, axis=0)
    idx -= min_idx

    # +1 to give non-zero labels
    labels = group["seg"][:] 
    remap = np.zeros(np.max(labels) + 1, dtype=labels.dtype)
    unique = np.unique(labels)
    remap[unique] = np.arange(1, len(unique) + 1)
    labels = remap[labels]
    shape = np.max(idx, axis=0) + 1
    print("loading shape: ", shape)
    vol = np.zeros(np.max(idx, axis=0) + 1, dtype=labels.dtype)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = labels

    print(list(range(1, np.max(labels) + 1)))

    return vol, anisotropy


# seg = h5py.File("r0.h5").get("seg")[:]
# seg = h5py.File("/home/jason/Downloads/snemisubmissions/human/test-input.h5").get("main")[:]
# im = h5py.File("/mmfs1/data/adhinart/foundation/zebrafinch_im_80nm.h5").get("main")[:]
# im = h5py.File("/mmfs1/data/adhinart/foundation/mouse_microns-phase1_64nm.h5").get("main")[:]
# im = h5py.File("/mmfs1/data/adhinart/foundation/mouse_moritz2019_88nm.h5").get("main")[:]
# seg = zarr.open_group("/mmfs1/data/adhinart/dendrite/data/foundation_mouse_microns")["seg"]["seg"][:]
# seg = zarr.open_group("/mmfs1/data/adhinart/dendrite/data/foundation_mouse_moritz")["seg"]["seg"][:]

#im = h5py.File("/mmfs1/data/adhinart/dendrite/raw/human_raw.h5").get("main")[:]

# im = h5py.File("/data/bccv/dataset/foundation/low_res/macaque_mpi_176-240nm.h5").get("main")[:]
# seg = zarr.open_group("/data/adhinart/dendrite/data/foundation_macaque/foundation.zip")["seg"][:]

# im = h5py.File("/data/bccv/dataset/foundation/low_res/human_h01_256-264nm.h5").get("main")[:,5000:5000+512,5000:5000+512]
# seg = zarr.open_group("/data/adhinart/dendrite/data/foundation_human/foundation.zip")["seg"][:,5000:5000+512,5000:5000+512]
# # im = h5py.File("/data/bccv/dataset/foundation/low_res/macaque_mpi_176-240nm.h5").get("main")[:]
# # seg = zarr.open_group("/data/adhinart/dendrite/data/foundation_macaque/foundation.zip")["seg"][:]
# # im = h5py.File("/data/bccv/dataset/foundation/low_res/macaque_mpi_176-240nm.h5").get("main")[:]
# # seg = zarr.open_group("/data/adhinart/dendrite/data/foundation_macaque/foundation.zip")["seg"][:]
# # im = h5py.File("/data/bccv/dataset/foundation/low_res/macaque_mpi_176-240nm.h5").get("main")[:]
# # seg = zarr.open_group("/data/adhinart/dendrite/data/foundation_macaque/foundation.zip")["seg"][:]

# id = "200_1600_1080"
# im = h5py.File(f"/scratch/wanjr/mouse/mouse_errors/{id}_img.h5").get("main")[:]
# orig = h5py.File(f"/scratch/wanjr/mouse/mouse_errors/{id}_pred.h5").get("main")[:]
# seg = zarr.open_group("/data/adhinart/dendrite/data/vessel/cleaned_sam.zip")["seg"][:]
# seg = seg[370*4:410*4, 250*4:390*4, 375*4:530*4]
# # seg = seg[260*4:320*4,910*4:1010*4,380*4:460*4]
# # seg = seg[:500, :500, :500]
# # seg = seg[::4,::4,::4]
# orig = h5py.File("/scratch/wanjr/mouse/mouse_final_seg_mobile_all_clean.h5").get("main")[:]
# orig = orig[370*4:410*4, 250*4:390*4, 375*4:530*4]
# orig = orig.astype(np.uint8) # need this for the h5
#

# id = ["SHL18_soma_image", "SHL18_junction_image", "SHL18_end_image"][0]
# im = h5py.File(f"/mmfs1/data/bccv/dataset/hydra/mito/train_3vol/{id}.h5").get("main")[:]
# seg = h5py.File(f"/data/adhinart/vesicle/3vol/{id}_seg.h5").get("main")[:]

# im = h5py.File("/data/adhinart/trisam/unet/data/human_train.h5").get("raw")[:]
# seg = h5py.File("/data/adhinart/trisam/unet/data/human_train.h5").get("label")[:]

# im = h5py.File("/data/projects/weilab/dataset/foundation/low_res/bvem/macaque_mpi_176-240nm.h5").get("main")[:]
# # gt = h5py.File("/data/projects/weilab/dataset/foundation/low_res/bvem/macaque_mpi_176-240nm_bv_gt_cc.h5").get("main")[:]
# # seg = h5py.File("/data/adhinart/trisam/unet/output/macaque/macaque_mpi_176-240nm_predictions.h5")["predictions"][:]
# # seg = (seg[0] > 0.5).astype(np.uint8)
# # seg[0:30] = 0
# # seg[-30:] = 0
# # seg[:, 0:30] = 0
# # seg[:, -30:] = 0
# # seg[:, :, 0:30] = 0
# # seg[:, :, -30:] = 0
# # seg = cc3d.largest_k(seg, k=126)
# # print("welp")
# # # seg = cc3d.connected_components(seg, connectivity=6)
# #
# # print(np.unique(seg))
# # print(np.unique(gt))
#
# seg = zarr.open_group("/data/adhinart/dendrite/data/foundation_macaque/foundation.zip")["seg"][:]
# seg = cc3d.largest_k(seg, k=126)
# remap = {i: i for i in range(128)}
# remap[1] = 126
# remap[126] = 1
# seg = fastremap.remap(seg, remap)
# print(np.unique(seg))

id = ["vol0", "vol1", "vol2"][0]
im = h5py.File(f"/data/adhinart/hydra/{id}_sample.h5").get("image")[:]
seg = h5py.File(f"/data/adhinart/hydra/{id}_sample.h5").get("label")[:]

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


# raw = h5py.File("/mmfs1/data/adhinart/dendrite/raw/human_raw.h5").get("main")[:]
# raw = raw[::4,::4,::4]
# print("done loading raw")
# spine = h5py.File("/mmfs1/data/adhinart/dendrite/raw/human_spine.h5").get("main")[:]
# spine = spine[::4,::4,::4]
# print("done loading spine")
# # inst = h5py.File("/mmfs1/data/adhinart/dendrite/raw/mouse_seg.h5").get("main")[:]
# # inst = inst[::4,::4,::4]
# binary = get_binary(raw, spine)
#seg = h5py.File("/mmfs1/data/adhinart/dendrite/raw/human_seg.h5").get("main")[:]

# raw = h5py.File("/mmfs1/data/adhinart/dendrite/raw/seg_den_raw_96nm.h5").get("main")[:]
# inst = h5py.File("/mmfs1/data/adhinart/dendrite/raw/seg_den_spine_96nm.h5").get("main")[:]
# inst = get_inst(raw, inst, 5)
# inst = get_inst(raw, inst, 6)


# inst, anisotropy = get_vol("human", 5)
# inst, anisotropy = get_vol("seg_den", 5)

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

# SNEMI (# 3d vol dim: z,y,x)
res = neuroglancer.CoordinateSpace(
    names=["z", "y", "x"],
    units=["nm", "nm", "nm"],
    # scales=anisotropy,
    scales=[1, 1, 1],
    # scales=[30, 6, 6]
    # scales=[30, 8, 8]
    # scales=[60, 96, 96]
    # names=["z", "y", "x"], units=["nm", "nm", "nm"], scales=[30, 6, 6]
    # names=["z", "y", "x"], units=["nm", "nm", "nm"], scales=[30, 64, 64]
)


with viewer.txn() as s:
    #s.layers.append(name="im", layer=ngLayer(im, res, tt="image"))
    s.layers.append(name='im',layer=ngLayer(im,res,tt='image'))
    s.layers.append(name="seg", layer=ngLayer(seg, res, tt="segmentation"))
    # s.layers.append(name="orig", layer=ngLayer(orig, res, tt="segmentation"))
    # s.layers.append(name='gt',layer=ngLayer(gt,res,tt='segmentation'))
    # s.layers.append(name="seg", layer=ngLayer(gt, res, tt="segmentation"))
    # s.layers.append(name="bin", layer=ngLayer(binary, res, tt="segmentation"))
    # s.layers.append(name="seg", layer=ngLayer(raw, res, tt="segmentation"))
    # s.layers.append(name="inst", layer=ngLayer(inst, res, tt="segmentation"))
    # Dd = "precomputed://https://rhoana.rc.fas.harvard.edu/ng/R0/im_64nm/"
    # s.layers["image"] = neuroglancer.ImageLayer(source=Dd)

print(viewer)
