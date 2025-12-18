from RotatedMNISTDataModule import RotatedMNISTDataModule

def main(task_order):
    dm = RotatedMNISTDataModule(
        data_dir = "./Datasets/data",
        task_order = task_order,
        recurring = False,
        batch_size = 20
    )
    dm.setup()
    
    for i, task in enumerate(task_order):
        train_loader = dm.train_dataloader()
        test_loader = dm.test_dataloader()
        
        for img, target, task, idx in train_loader:
            # model.train(img, target)

        for img, target, task, idx in test_loader:
            # model.test(img, target)
        
        if i < 19:
            dm.next_task()

if __name__ == "__main__": 
    task_order = list(range(20))
    # task_order = [18, 1, 19, 8, 10, 17, 6, 13, 4, 2, 5, 14, 9, 7, 16, 11, 3, 0, 15, 12]
    # task_order = [8, 28, 9, 29, 12, 32, 13, 33, 16, 36, 17, 37, 0, 20, 1, 21, 4, 24, 5, 25]
    
    