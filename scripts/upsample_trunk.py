import h5py
import numpy as np

ANISOTROPY = [30, 8, 8]
# determine the spacing between vertices of the shaft
SAMPLE_RATE_NM = 200


file = h5py.File("human/skels/skel_1.h5")
vertices = file.get("vertices")[:]
longest_path = file.get("longest_path")[:]
trunk = vertices[longest_path] * np.array(ANISOTROPY)

distances = np.sqrt(np.sum((trunk[1:] - trunk[:-1]) ** 2, axis=1))

cum_distances = np.concatenate(([0], np.cumsum(distances)))
total_distance = np.sum(distances)
locations = np.arange(0, total_distance, SAMPLE_RATE_NM)
idx = np.searchsorted(cum_distances, locations, side="right") - 1
assert np.all(idx >= 0)

# linear interpolation
final = trunk[idx] + (trunk[idx + 1] - trunk[idx]) * (
    (locations - cum_distances[idx]) / distances[idx]
).reshape(-1, 1)
