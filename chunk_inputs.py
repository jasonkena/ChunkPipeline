import numpy as np
import open3d as o3d
import sys
import os


def extract_pc(sid, root_dir):
    fn = os.path.join(root_dir, str(sid) + ".npy")
    if not os.path.exists(fn):
        print("point cloud file does not exist")
        sys.exit(0)

    with open(fn, "rb") as fp:
        data = np.load(fp)
        points = data[:, 0:3].astype("float32") * [5, 1, 1]
        label = data[:, 3].astype("bool")
    colors = [[1, 0, 0] if l else [0, 1, 0] for l in label]
    return points, colors


def crop_point_cloud(sid, root_dir, result_dir, voxel_size=200, point_threshold=30000):

    print("PROCESSING POINT CLOUD ", sid)
    print()
    points, colors = extract_pc(sid, root_dir)
    max_points = np.amax(points, axis=0).astype(int)
    min_points = np.amin(points, axis=0).astype(int)

    # create point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    max_points = np.amax(points, axis=0).astype(int)
    min_points = np.amin(points, axis=0).astype(int)
    min_point_count, max_point_count = points.shape[0], 0
    small_point_cloud, big_point_cloud = 0, 0
    selected_points = 0
    for i in range(min_points[0], max_points[0], voxel_size):
        for j in range(min_points[1], max_points[1], voxel_size):
            for k in range(min_points[2], max_points[2], voxel_size):
                min_bound = [i, j, k]
                max_bound = [i + voxel_size, j + voxel_size, k + voxel_size]
                bb = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=min_bound, max_bound=max_bound
                )
                cropped = pc.crop(bounding_box=bb)
                cropped_points = np.asarray(cropped.points)
                point_count = cropped_points.shape[0]
                min_point_count = min(min_point_count, point_count)
                max_point_count = max(max_point_count, point_count)
                if point_count > point_threshold:
                    big_point_cloud += 1
                    print(
                        "cropping ",
                        big_point_cloud,
                        ", number of points: ",
                        point_count,
                    )
                    selected_points += point_count
                    cropped_color = np.asarray(cropped.colors)
                    cropped_label = np.asarray(
                        [
                            True if (l == [1, 0, 0]).all() else False
                            for l in cropped_color
                        ]
                    )
                    data = {}
                    data["points"] = cropped_points
                    data["label"] = cropped_label
                    data["min_bound"] = min_bound
                    data["max_bound"] = max_bound
                    outfile = (
                        result_dir
                        + "/"
                        + str(sid)
                        + "_"
                        + str(big_point_cloud)
                        + ".npz"
                    )
                    np.savez(outfile, **data)
                else:
                    small_point_cloud += 1
    print()
    print("min_point_count: ", min_point_count, " max_point_count: ", max_point_count)
    print(
        "small_point_cloud count: ",
        small_point_cloud,
        " big_point_cloud count: ",
        big_point_cloud,
    )
    print(
        "selected points: ",
        selected_points,
        " lost points: ",
        points.shape[0] - selected_points,
    )
    print()


if __name__ == "__main__":
    root_dir = "results"
    result_dir = "cropped_res200_th30000"
    crop_point_cloud(int(sys.argv[1]) + 1, root_dir, result_dir)
