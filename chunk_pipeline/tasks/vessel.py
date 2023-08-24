import chunk_pipeline.tasks.chunk as chunk
from chunk_pipeline.tasks.coarse import chunk_grey_erode_or_dilate, fill_and_remove_dust


def task_clean_sam(cfg, h5):
    general = cfg["GENERAL"]
    vessel = cfg["VESSEL"]
    uint_dtype = general["UINT_DTYPE"]
    vol = h5["main"]

    # erode then dilate
    eroded = chunk_grey_erode_or_dilate(
        vol, vessel["EROSION_STRUCTURE"], uint_dtype, operation="erode"
    )
    dilated = chunk_grey_erode_or_dilate(
        eroded, vessel["EROSION_STRUCTURE"], uint_dtype, operation="dilate"
    )

    filtered = fill_and_remove_dust(
        dilated, vessel["DUST_THRESHOLD"], vessel["DUST_CONNECTIVITY"]
    )

    cc, voxel_counts = chunk.chunk_cc3d(
        filtered, vessel["CONNECTIVITY"], False, uint_dtype=general["UINT_DTYPE"]
    )

    return {"seg": cc, "voxel_counts": voxel_counts}
