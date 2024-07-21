### TODO: add your import
import matplotlib.pyplot as plt
import os
def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    ### TODO: visualize loss of training & validation and save to [out_dir]/loss.png
    ###
    train_x = list(range(train_interval, train_interval * len(train_loss_list) + 1, train_interval))
    val_x = list(range(val_interval, val_interval * len(val_loss_list) + 1, val_interval))
    plt.figure(figsize=(10, 6))
    plt.plot(train_x, train_loss_list, label='Training Loss')
    plt.plot(val_x, val_loss_list, label='Validation Loss')

    plt.legend()
    plt.title(f'Training and Validation Loss for {dataset}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()
    pass
