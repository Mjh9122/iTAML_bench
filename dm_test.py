from RotatedMNISTDataModule import RotatedMNISTDataModule
import matplotlib.pyplot as plt

# task_order = [18, 1, 19, 8, 10, 17, 6, 13, 4, 2, 5, 14, 9, 7, 16, 11, 3, 0, 15, 12]
task_order = list(range(20))
# task_order = [8, 28, 9, 29, 12, 32, 13, 33, 16, 36, 17, 37, 0, 20, 1, 21, 4, 24, 5, 25]

dm = RotatedMNISTDataModule(
    data_dir="./Datasets/data",
    task_order= task_order,
    recurring = False,
    batch_size=20
)
dm.setup()

fig, ax = plt.subplots(20, 20, figsize=(12, 12))
plt.subplots_adjust(wspace=0, hspace=0)

for i, task in enumerate(task_order):
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    batch = next(iter(train_loader))
    img, target, task, idx = batch
    for j, im in enumerate(img):
        im_np = im.numpy()
        ax[i, j].imshow(im_np.reshape(28, 28))
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

    batch = next(iter(test_loader))
    img, target, task, idx = batch
    
    if i < 19:
        dm.next_task()

plt.savefig("ro_mnist_training_in_order.png", dpi=150, bbox_inches='tight', pad_inches=0)
plt.show()