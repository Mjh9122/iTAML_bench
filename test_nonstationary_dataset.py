from non_stationary_datasets import get_dataset_registry

reg, names = get_dataset_registry('../Datasets/nonstationary_root')
Sx, Sy, Qx, Qy, Sg, Qg = reg['vggflowers'].sample_n_way_k_shot(5, 5, 15)
for t in [Sx, Sy, Qx, Qy, Sg, Qg ]:
    print(t.shape)