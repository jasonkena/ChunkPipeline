import chunk_pipeline.tasks.chunk as chunk

# from chunk_pipeline.tasks.coarse import chunk_grey_erode_or_dilate, fill_and_remove_dust
from chunk_pipeline.tasks.sphere import get_dt, sphere_iteration


def task_clean_sam(cfg, h5):
    general = cfg["GENERAL"]
    anisotropy = general["ANISOTROPY"]
    # uint_dtype = general["UINT_DTYPE"]

    vessel = cfg["VESSEL"]
    max_erode = vessel["MAX_ERODE"]
    erode_delta = vessel["ERODE_DELTA"]

    vol = h5["main"]

    dt = get_dt(vol, anisotropy, black_border=False, threshold=max_erode + erode_delta)
    remaining = dt >= max_erode
    expanded = sphere_iteration(remaining, dt, vol, erode_delta, anisotropy)

    cc, voxel_counts = chunk.chunk_cc3d(
        expanded, vessel["CONNECTIVITY"], False, uint_dtype=general["UINT_DTYPE"]
    )

    return {"seg": cc, "voxel_counts": voxel_counts}


def task_vessel_cc(cfg, h5):
    general = cfg["GENERAL"]
    vessel = cfg["VESSEL"]

    __import__("pdb").set_trace()
