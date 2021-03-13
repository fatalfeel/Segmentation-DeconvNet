import os
import torch.utils.data as data
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}

# refer to:
# https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
# https://blog.csdn.net/tcltyan/article/details/112206590
class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``tainval`` orr ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 transform=None,
                 target_transform=None):

        self.root               = os.path.expanduser(root)
        self.year               = year
        self.url                = DATASET_YEAR_DICT[year]['url']
        self.filename           = DATASET_YEAR_DICT[year]['filename']
        self.md5                = DATASET_YEAR_DICT[year]['md5']
        self.transform          = transform
        self.target_transform   = target_transform
        self.image_set          = image_set
        base_dir    = DATASET_YEAR_DICT[year]['base_dir']
        voc_root    = os.path.join(self.root, base_dir)
        image_dir   = os.path.join(voc_root, 'JPEGImages')
        mask_dir    = os.path.join(voc_root, 'SegmentationClass')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        splits_dir  = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f     = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered! Please use image_set="train" '
                             'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks  = [os.path.join(mask_dir, x + ".png") for x in file_names]

        self.class_colors = self.color_map(21)

    def show_MNIST(self, img):
        plt.imshow(img)
        plt.title('Batch from dataloader')
        plt.axis('off')
        plt.show()

    def bitget(self, byteval, idx):
        return (byteval & 1 << idx) != 0  # 判断输入字节的idx比特位上是否为1

    def color_map(self, classes, normalized=False):
        dtype   = 'float32' if normalized else 'uint8'
        cmap    = np.zeros((classes, 3), dtype=dtype)
        for i in range(classes):
            c = i
            r = g = b = 0                               # 将类别索引和rgb分量都视为8位2进制数，即一个字节
            for j in range(8):                          # 从高到低填入rgb分量的每个比特位
                r = r | self.bitget(c, 0) << (7 - j)    # 每次将类别索引的第0位放置到r分量
                g = g | self.bitget(c, 1) << (7 - j)    # 每次将类别索引的第1位放置到g分量
                b = b | self.bitget(c, 2) << (7 - j)    # 每次将类别索引的第2位放置到b分量
                c = c >> 3  # 将类别索引移位
            cmap[i] = np.array([r, g, b])
        cmap = cmap / 255 if normalized else cmap
        return cmap

    def show_color_map(self):
        labels = ['Background', 'Aero plane', 'Bicycle', 'Bird', 'Boat',
                  'Bottle',   'Bus', 'Car', 'Cat', 'Chair',
                  'Cow', 'Dining-Table', 'Dog', 'Horse', 'Motorbike',
                  'Person', 'Potted-Plant', 'Sheep', 'Sofa', 'Train',
                  'TV/Monitor', 'Void/Unlabelled']

        row_size    = 80
        col_size    = 250
        cmap        = self.color_map(256)
        r           = 3
        c           = 7
        delta = 10
        array = np.empty((row_size * (r + 1), col_size * c, cmap.shape[1]), dtype=cmap.dtype)
        for r_idx in range(0, r):
            for c_idx in range(0, c):
                i = r_idx * c + c_idx
                array[r_idx * row_size:(r_idx + 1) * row_size, c_idx * col_size: (c_idx + 1) * col_size, :] = cmap[i]
                x = c_idx * col_size + delta
                y = r_idx * row_size + row_size / 2
                s = labels[i]
                plt.text(x, y, s, fontsize=9, color='white')
                print("write {} at pixel (r={},c={})".format(labels[i], y, x))

        array[r * row_size:(r + 1) * row_size, :] = cmap[-1]
        x = 3 * col_size + delta
        y = r * row_size + row_size / 2
        s = labels[-1]
        plt.text(x, y, s, fontsize=9, color='black')
        print("write {} at pixel (r={},c={})".format(labels[i], y, x))
        plt.title("PASCAL VOC Label Color Map")
        plt.imshow(array)
        plt.axis('off')
        plt.show()

    def encode_segmap(self, ground):
        temp            = ground.astype(int)
        label_mask      = np.zeros((ground.shape[:2]), dtype=np.int16)

        for ii, label in enumerate(self.class_colors):
            #label_mask[np.where(np.all(temp == label, axis=-1))[:2]] = ii
            compare = np.all(temp == label, axis=-1)
            label_mask[np.where(compare)[:2]] = ii

        label_mask = label_mask.astype(int)

        return label_mask
    
    def decode_segmap(self, label_mask):
        r   = label_mask.copy().astype(int)
        g   = label_mask.copy().astype(int)
        b   = label_mask.copy().astype(int)
        
        for ll in range(0, 21):
            r[label_mask == ll] = self.class_colors[ll, 0]
            g[label_mask == ll] = self.class_colors[ll, 1]
            b[label_mask == ll] = self.class_colors[ll, 2]
            
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r #/ 255.0
        rgb[:, :, 1] = g #/ 255.0
        rgb[:, :, 2] = b #/ 255.0

        return rgb
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image           = Image.open(self.images[index]).convert('RGB')
        ground_color    = Image.open(self.masks[index]).convert('RGB')
        ground_color    = transforms.Resize((224, 224))(ground_color)
        ground_color    = np.array(ground_color)
        target_label    = self.encode_segmap(ground_color) #transfer to 21 classes ccolor
        target_label    = torch.from_numpy(target_label)

        #self.show_MNIST(image)
        #self.show_MNIST(target_label) #0~21

        #data transfer to 0.0 ~ 1.0
        if self.transform is not None:
            image = self.transform(image)

        return image, target_label
    



