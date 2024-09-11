import glob
from dataloader import visualize_batch
from cache_dataloader import CachedDataset
from magicpickle import MagicPickle

# def visualize_batch(pc, trunk_skel, label):
if __name__ == "__main__":
    with MagicPickle("think-jason") as mp:
        if mp.is_remote:
            dataset = CachedDataset(
                "/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/dataset_0_10000.0_4096"
            )
            data = [dataset[i] for i in range(len(dataset))]
            mp.save(data)
        else:
            data = mp.load()
            for i in range(len(data)):
                print(f"Visualizing {i}/{len(data)}")
                trunk_id, pc, trunk_pc, label = data[i]
                visualize_batch(pc, trunk_pc, label)
