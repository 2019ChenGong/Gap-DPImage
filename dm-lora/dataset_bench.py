import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
from typing import Any
import copy

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict()
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        return image, self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        # if label.dtype == np.int64:
        #     onehot = np.zeros(self.label_shape, dtype=np.float32)
        #     onehot[label] = 1
        #     label = onehot
        return label.copy()

    def get_details(self, idx):
        d = EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        c = None,
        transform = None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.c = c
        self.transform = transform

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).size)
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                if self.c == 1:
                    image = PIL.Image.open(f).convert("L")
                else:
                    image = PIL.Image.open(f).convert("RGB")
            return copy.deepcopy(image)

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

def get_prompt(data_name: str):

    if data_name.startswith("mnist"):
        return ["A grayscale image of a handwritten digit 0", "A grayscale image of a handwritten digit 1", "A grayscale image of hand-written 2", "A grayscale image of hand-written 3", "A grayscale image of hand-written 4", "A grayscale image of hand-written 5", "A grayscale image of hand-written 6", "A grayscale image of hand-written 7", "A grayscale image of hand-written 8", "A grayscale image of hand-written 9"]
    
    elif data_name.startswith("fmnist"):
        return ["A grayscale image of a T-shirt", "A grayscale image of a Trouser", "A grayscale image  of Pullover", "A grayscale image of Dress", "A grayscale image of Coat", "A grayscale image of  Sandal", "A grayscale image of Shirt", "A grayscale image of Sneaker", "A grayscale image of  Bag", "A grayscale image of Ankle boot"]
    
    elif data_name.startswith("cifar100"):
        cifar100_y = '''Superclass	Classes
        aquatic mammals	beaver, dolphin, otter, seal, whale
        fish	aquarium fish, flatfish, ray, shark, trout
        flowers	orchids, poppies, roses, sunflowers, tulips
        food containers	bottles, bowls, cans, cups, plates
        fruit and vegetables	apples, mushrooms, oranges, pears, sweet peppers
        household electrical devices	clock, computer keyboard, lamp, telephone, television
        household furniture	bed, chair, couch, table, wardrobe
        insects	bee, beetle, butterfly, caterpillar, cockroach
        large carnivores	bear, leopard, lion, tiger, wolf
        large man-made outdoor things	bridge, castle, house, road, skyscraper
        large natural outdoor scenes	cloud, forest, mountain, plain, sea
        large omnivores and herbivores	camel, cattle, chimpanzee, elephant, kangaroo
        medium-sized mammals	fox, porcupine, possum, raccoon, skunk
        non-insect invertebrates	crab, lobster, snail, spider, worm
        people	baby, boy, girl, man, woman
        reptiles	crocodile, dinosaur, lizard, snake, turtle
        small mammals	hamster, mouse, rabbit, shrew, squirrel
        trees	maple, oak, palm, pine, willow
        vehicles 1	bicycle, bus, motorcycle, pickup truck, train
        vehicles 2	lawn-mower, rocket, streetcar, tank, tractor'''
        class_lins = cifar100_y.split("\n")

        prompt = []

        for line in class_lins[1:]:
            super_class, child_classes = line.split("\t")
            for child_class in child_classes.split(", "):
                prompt.append("An image of {}".format(child_class))
        return prompt
    
    elif data_name.startswith("cifar10"):
        return ["An image of an airplane", "An image of an automobile", "An image of a bird", "An image of a cat", "An image of a deer", "An image of a dog", "An image of a frog", "An image of a horse", "An image of a ship", "An image of a truck"]

    elif data_name.startswith("eurosat"):
        return ["A remote sensing image of an industrial area", "A remote sensing image of a residential area", "A remote sensing image of an annual crop area", "A remote sensing image of a permanent crop area", "A remote sensing image of a river area", "A remote sensing image of a sea or lake area", "A remote sensing image of a herbaceous veg. area", "A remote sensing image of a highway area", "A remote sensing image of a pasture area", "A remote sensing image of a forest area"]
 
    elif data_name.startswith("celeba_male"):
        return ["An image of a female face", "An image of a male face"]
    
    elif data_name.startswith("camelyon"):
        return ["A normal lymph node image", "A lymph node histopathology image"]
    
    elif data_name.startswith("covid"):
        return ["A chest radiograph", "A chest radiograph"]

    elif data_name.startswith("octmnist"):
        return ["An optical coherence tomography (OCT) images for retinal disease 1", "An optical coherence tomography (OCT) images for retinal disease 2", "An optical coherence tomography (OCT) images for retinal disease 3", "An optical coherence tomography (OCT) images for retinal disease 4"]
    
    else:
        raise NotImplementedError