import argparse

def input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../dataset/shapenet/train_imgs', help='input images path')
    parser.add_argument('--vox_dir', type=str, default='../dataset/shapenet/train_voxels', help='input voxels path')
    parser.add_argument('--lr', type=float, default='0.0002', help='learning rate')
    parser.add_argument('--batch_size', type=int, default='32', help='batch_size in training')
    parser.add_argument("--epoch", type=int, default=500, help="epoch in training")

    args = parser.parse_args()
    return args