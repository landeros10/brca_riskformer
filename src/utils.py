'''
Created Feb 2022
author: landeros10

Lee Laboratory
Center for Systems Biology
Massachusetts General Hospital
'''
from __future__ import (print_function, division,
                        absolute_import, unicode_literals)
from os.path import splitext, basename
import os
import json
import torch
import numpy as np
import pandas as pd
import random
import logging

logger = logging.getLogger(__name__)

# from shapely.ops import cascaded_union
# from rtree import index

# FLAGS = flags.FLAGS
# LARGE = 1e9

# PCA_MATRIX = np.array([[0.25277847, 0.13902594, -0.15348206],
#                        [0.13902594, 0.14048096, -0.19129017],
#                        [-0.15348206, -0.19129017, 0.53886775]])
# PCA_MIN = np.array([-24.61131465, -30.84998116, -77.20466764])
# PCA_MAX = np.array([116.73598581,  94.25730744, 182.47919644])


# LABEL2CLASS = {"Benign": 1,
#                "Carcinoma in situ": 2,
#                "In situ carcinoma": 2,
#                "Invasive carcinoma": 3,
#                "Carcinoma invasive": 3}

# def stain_augmentor_wrapper(image, **kwargs):
#     if np.std(image) < 10:
#         return image
#     else:
#         return stain_augmentor(image)

# def augmentor_batch(batch_images, tile_size, bs, num_gpus, crop_frac, rotate_limit, blur_radius, blur_p):
#     avg_pixel = batch_images.mean(axis=(0, 1, 2)).astype(int).tolist()

#     transform = A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=1.0),
#         A.RandomResizedCrop(tile_size, tile_size, scale=(crop_frac, 1.0), ratio=(1.0, 1.0), p=1.0),
#         A.Lambda(image=stain_augmentor_wrapper),
#         A.ElasticTransform(alpha=50, sigma=50, alpha_affine=15, p=0.80),
#         A.Defocus(radius=(1, blur_radius), alias_blur=0.0, p=blur_p),
#         A.ShiftScaleRotate(scale_limit=0.25, rotate_limit=(-rotate_limit, rotate_limit), border_mode=0, value=avg_pixel, p=1.0),
#     ])

#     # num_gpus = max(num_gpus, 1)
#     small_bs = bs // num_gpus
#     augmented_images = np.zeros((bs * 2, tile_size, tile_size, 3), dtype=batch_images.dtype)
#     for i in range(bs):
#         start_idx = (i // small_bs) * small_bs * 2
#         aug1_idx = start_idx + (i % small_bs)
#         aug2_idx = start_idx + (i % small_bs) + small_bs
#         # print(f"bs: {bs}\tidx1:{aug1_idx}\tidx2:{aug2_idx}")

#         image = batch_images[i].astype(np.uint8)
#         augmented = transform(image=image)['image']
#         assert augmented.shape[1] == tile_size
#         augmented_images[aug1_idx] = augmented

#         augmented = transform(image=image)['image']
#         assert augmented.shape[1] == tile_size
#         augmented_images[aug2_idx] = augmented
#     return augmented_images


# def _augment(x, tile_size, bs, num_gpus,
#              crop_frac=0.9, rotate_limit=10, blur_radius=5, blur_p=0.25,
#              train=True):
#     """ Performs stochastic augmentaiton steps.

#     Performs augmentation steps, using the following probability params:
#     """
#     out_size = (bs * 2, tile_size, tile_size, 3)
#     x = tf.numpy_function(
#         func=augmentor_batch,
#         inp=[x, tile_size, bs, num_gpus, crop_frac, rotate_limit, blur_radius, blur_p],
#         Tout=tf.uint8)
#     x.set_shape(out_size)
#     x = tf.cast(x, tf.float32) / 255.0
#     return x


# def combine_preds(dataset_batch, pred, n_class,
#                   auto_pred=None,
#                   bs=25, resize_out=1.0):
#     """
#     Combines the data, grouth thruth and the prediction into one rgb image

#     :param dataset_batch: x - (N, H, W, n_channel), y - (N, H, W),
#     :param preds: predicted tensor size (bs, H, W)

#     :returns img: the concatenated rgb image
#     """
#     ps = pred.shape
#     width = ps[2]

#     x, y = dataset_batch
#     x, y = cropto(x, ps), cropto(y, ps)
#     n_ch = x.shape[-1]

#     # convert to [bs x width] image
#     x = x[:min(x.shape[0], bs), ...].numpy()
#     # x = (((x + 1) / 2) * PCA_MAX)
#     # x = ((x + PCA_MIN) @ np.linalg.inv(PCA_MATRIX))
#     x = ((x + 1) / 2) * 255
#     x = np.around(x).astype(int)
#     x = x.reshape(-1, width, n_ch)

#     if auto_pred is not None:
#         auto_pred = cropto(auto_pred, ps)
#         auto_pred = auto_pred[:min(x.shape[0], bs), ...].numpy()
#         # auto_pred = (((auto_pred + 1) / 2) * PCA_MAX)
#         # auto_pred = (auto_pred + PCA_MIN) @ np.linalg.inv(PCA_MATRIX)
#         auto_pred = ((auto_pred + 1) / 2) * 255
#         auto_pred = np.around(auto_pred).astype(int)
#         auto_pred = auto_pred.reshape(-1, width, n_ch)
#         x = np.concatenate((x, auto_pred), axis=1)

#     # convert to [bs x width] image
#     y = y[:min(y.shape[0], bs), ...].numpy()
#     pred = pred[:min(y.shape[0], bs), ...].numpy().astype(np.float32)

#     gt = to_rgb(y.reshape(-1, width), amin=0, amax=n_class-1)
#     p = to_rgb(pred.reshape(-1, width), amin=0, amax=n_class-1)

#     # Combine and resize
#     img = np.concatenate((x, gt, p), axis=1)
#     img = rescale(img.astype(float), resize_out, multichannel=True).astype(int)
#     return img


# def save_image(img, path, resize_out=3.0):
#     """
#     Writes the image to disk

#     :param img: the rgb image to save
#     :param path: the target path
#     """
#     h, w = img.shape
#     img = img[:, :w//2].astype(np.float32)
#     h, w = img.shape
#     h_, w_ = int(resize_out * h), int(resize_out * w)
#     img = img + img.min()
#     img = img / img.max()

#     red = img.copy()
#     red[range(red.shape[0]), red.argmax(axis=-1)] *= 1.1
#     img[range(img.shape[0]), red.argmax(axis=-1)] *= .5
#     red = np.clip(red, None, 1.0)

#     img = cv2.resize(img, dsize=(w_, h_), interpolation=cv2.INTER_NEAREST)
#     red = cv2.resize(red, dsize=(w_, h_), interpolation=cv2.INTER_NEAREST)

#     img = np.stack([red, img, img], axis=-1)
#     img = (img * 255.0).round().astype(np.uint8)
#     img = Image.fromarray(img)
#     img.save(path, 'JPEG', dpi=[300, 300], quality=90)


# def norm(arr, axis=None):
#     arr_min, arr_max = arr.min(axis=axis), arr.max(axis=axis)
#     return (arr - arr_min) / (arr_max - arr_min)


# def affine_norm(input, mu, sigma):
#     input_std = input.std(axis=(1, 2), keepdims=True)
#     input_std[input_std == 0] = 1
#     input_mean = input.mean(axis=(1, 2), keepdims=True)
#     input -= input_mean
#     input *= (sigma/input_std)
#     return input + mu



# def process_image(args):
#     file, x, y, in_s, out_s = args
#     slideObj = OpenSlide(file.decode("utf-8"))
#     x, y = int(x), int(y)
#     image = slideObj.read_region((x, y), 0, (in_s, in_s)).convert('RGB')
#     image = np.array(image.resize((out_s, out_s), Image.BILINEAR))
#     return image


# def read_slide_batch_parallel(files, xs, ys, in_ss, out_s, pool_num=4):
#     bs = xs.shape[0]
#     batch_images = np.zeros((2 * bs, out_s, out_s, 3), dtype=np.uint8)

#     with Pool(processes=pool_num) as pool:  # Adjust number of processes based on your system's capabilities
#         args = [(files[i], xs[i], ys[i], in_ss[i], out_s) for i in range(bs)]
#         results = pool.map(process_image, args)

#     for i, image in enumerate(results):
#         batch_images[i] = image
#         batch_images[i + bs] = image

#     return batch_images


# def read_slide_batch(files, xs, ys, in_ss, out_s, normalize=False):
#     bs = xs.shape[0]
#     batch_images = np.zeros((2 * bs, out_s, out_s, 3), dtype=np.uint8)

#     for i, (file, x, y, in_s) in enumerate(zip(files, xs, ys, in_ss)):
#         slideObj = OpenSlide(file.numpy().decode("utf-8"))
#         x, y = int(x), int(y)
#         image = slideObj.read_region((x, y), 0, (in_s, in_s)).convert('RGB')
#         image = np.array(image.resize((out_s, out_s), Image.BILINEAR))
#         batch_images[i] = image
#         batch_images[i + bs] = image
#     return batch_images


# def read_tif(file, in_s, x, y):
#     tf_arr = Image.open(file.numpy().decode("utf-8")).convert('RGB')
#     return np.array(tf_arr)[x:(x + in_s), y:(y + in_s)]


# def read_gt(file, in_s, out_s, x, y, down_factor):
#     # TODO add STRtree
#     file = file.numpy().decode("utf-8")
#     x, y = x.numpy(), y.numpy()
#     regions = pickle.load(open(file.replace("svs", "pickle"), "rb"))

#     sampleCoords = np.array([x, y, x + in_s, y + in_s])
#     sampleCoords = np.around(sampleCoords / down_factor).astype(int)
#     samplePoly = box(*sampleCoords)

#     map_s = sampleCoords[2] - sampleCoords[0] + 1
#     gtmap = np.zeros((map_s, map_s), dtype="uint8")
#     for (rPoly, label) in regions:
#         rPoly = affinity.scale(rPoly,
#                                xfact=(1/down_factor),
#                                yfact=(1/down_factor),
#                                origin=(0, 0))
#         if rPoly.intersects(samplePoly):
#             if samplePoly.within(rPoly):
#                 gtmap = np.ones((out_s, out_s)) * label
#                 break
#             else:
#                 overlap = rPoly.intersection(samplePoly)
#                 if overlap.geom_type == "GeometryCollection":
#                     idx = [g.geom_type == "Polygon" for g in overlap.geoms]
#                     idx = idx.index(True)
#                     overlap = overlap.geoms[idx]
#                 if overlap.geom_type != "Polygon":
#                     continue
#                 rx, ry = overlap.exterior.xy
#                 rr, cc = np.array(polygon(rx, ry)) - sampleCoords[0][..., None]
#                 rr, cc = np.clip(rr, 0, map_s - 1), np.clip(cc, 0, map_s - 1)
#                 newregion = np.zeros_like(gtmap)
#                 newregion[rr, cc] = 1
#                 ndimage.binary_fill_holes(newregion, output=newregion)
#                 gtmap[newregion == 1] = label
#     gtmap = resize(gtmap.T, [out_s, out_s], order=0, anti_aliasing=False)
#     gtmap = np.eye(4)[gtmap.astype(np.uint8)]
#     return gtmap


# def read_gt_1d(file, in_s, x, y):
#     if isinstance(file, str):
#         file_str = file
#     else:
#         file_str = file.numpy().decode("utf-8")
#     if "WSI/A" not in file_str:
#         return np.array([-1])
#     treefile = file_str.replace("svs", "pickle")
#     regionstree, labels = pickle.load(open(treefile, "rb"))

#     if isinstance(x, np.uint32):
#         x, y = x.item(), y.item()
#     else:
#         x, y = x.numpy(), y.numpy()
#     samplePoly = box(x, y, x + in_s, y + in_s)
#     areas = np.array([samplePoly.area, 0, 0, 0])
#     for i in regionstree.query(samplePoly, predicate="intersects"):
#         label = labels[i]
#         rPoly = regionstree.geometries.take(i)
#         if samplePoly.within(rPoly):
#             return np.array([label])
#         overlap = rPoly.intersection(samplePoly)
#         if overlap.geom_type == "GeometryCollection":
#             overlap = overlap.geoms[0]
#         if overlap.geom_type != "Polygon":
#             continue
#         area = overlap.area
#         areas[0] -= area
#         areas[label] += area
#     # TODO
#     # Change to include 25% minimum
#     return np.array([areas.argmax()])


# def _sample_from_tfrecord(pf, out_s, batched=False, return_dict=False, normalize=False):
#     """ SVS coords are saved such that x: col, y: row."""
#     files = pf["file"]
#     xs = pf["x"]
#     ys = pf["y"]
#     in_ss = pf["input_s"]
#     extractor = read_slide_batch if batched else read_slide_
#     out_shape = [None, out_s, out_s, 3] if batched else [out_s, out_s, 3]

#     tf_arr = tf.py_function(extractor,
#                             [files, xs, ys, in_ss, out_s, normalize],
#                             Tout=tf.uint8)
#     tf_arr.set_shape(out_shape)
#     if return_dict:
#         pf["images"] = tf_arr
#         return pf
#     return tf_arr


# def filter_func_TCGA(pf):
#     def filter_fn(file):
#         file_str = file.numpy().decode("utf-8")
#         return (~("WSI/A" in file_str) and ("svs" in file_str))

#     return tf.py_function(filter_fn,
#                           [pf["file"]],
#                           [tf.bool])[0]


# def setup_logstring(loss_names, epoch, evalTime, metrics):
#     losses_temp = ""

#     trn_losses, trn_accs, tst_losses, tst_accs = metrics
#     if len(loss_names) > 1:
#         losses_temp = ''.join('%s: {:.3f} ' % n for n in loss_names)

#     logstring = "".join(("Epoch {}, Time:{:.1f}m,",
#                          "Train Loss: {:.3f}, ",
#                          losses_temp,
#                          "Train Accuracy: {:.3f}, ",
#                          "Train Recall: {:.2f}, ",
#                          "Train Precision: {:.2f}",
#                          "Test Loss: {:.3f},  ",
#                          losses_temp,
#                          "Test Accuracy: {:.3f}, ",
#                          "Test Recall: {:.2f}, ",
#                          "Test Precision: {:.2f}"))
#     values = [epoch+1, evalTime,
#               tf.math.reduce_sum(trn_losses).numpy(),
#               *trn_losses.numpy(),
#               *[t.numpy() for t in trn_accs],
#               tf.math.reduce_sum(tst_losses).numpy(),
#               *tst_losses.numpy(),
#               *[t.numpy() for t in tst_accs]]

#     logstring = logstring.format(*values)
#     return logstring, values


# # For use in deeplabv3+ architecture
# def mods_list(length=18, down=(), up=()):
#     default = np.repeat("in", length)
#     default[np.array(down, int)] = "dn"
#     default[np.array(up, int)] = "up"
#     return list(default)


# def make_3c_label(y, border_size=2):
#     """ Create 3-class label from segmentation map.

#     Segmentation map is an integer array with 0 as background. Each segmented
#     object is represented by a unique non-zero int. """
#     numCells = int(y.max())
#     new_y = np.zeros_like(y).astype(np.int)

#     for i in range(1, 1 + numCells):
#         interior = (y == i)
#         new_y[interior] = 1

#         boundary = interior & ~binary_erosion(interior, disk(border_size))
#         new_y[boundary] = 2
#     return new_y


# def print_logits(logits, labels):
#     l = logits.numpy()
#     print(np.argmax(l, axis=1)[:5])
#     print(np.argmax(l, axis=1).max())
#     l = labels.numpy()
#     print(np.argmax(l, axis=1)[:5])
#     print(np.argmax(l, axis=1).max())
#     return 1


# def build_optimizer(name, lr, clip_norm=1.0, momentum=0.9, decay_steps=None):
#     if decay_steps is not None:
#         lr = tf.keras.optimizers.schedules.CosineDecay(
#             initial_learning_rate=lr,
#             decay_steps=decay_steps,
#             alpha=1e-3,
#             name='cosine_decay_scheduler'
#         )

#     if name == "momentum":
#         print("Using SGD\n")
#         optimizer = optimizers.SGD(lr, momentum, nesterov=True, clipnorm=clip_norm)
#     elif name == "adam":
#         print("Using Adam \n")
#         optimizer = optimizers.Adam(lr, clipnorm=clip_norm)
#     elif name == "adamw":
#         print("Using Adam \n")
#         optimizer = optimizers.AdamW(lr, clipnorm=clip_norm)
#     elif name == "nadam":
#         print("Using Nadam \n")
#         optimizer = optimizers.Nadam(lr, clipnorm=clip_norm)
#     elif name == "lion":
#         print("Using Lion \n")
#         optimizer = optimizers.Lion(lr, clipnorm=clip_norm)
#     elif name == "lars":
#         print("Using LARS \n")
#         optimizer = LARSOptimizer(
#                 lr,
#                 momentum=momentum,
#                 weight_decay=FLAGS.l2_coeff,
#                 exclude_from_weight_decay=[
#                     'batch_normalization', 'bias', 'head_supervised'
#                 ],
#                 clipnorm=clip_norm)
#     return optimizer


# class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
#     """Applies a warmup schedule on a given learning rate decay schedule."""

#     def __init__(self, base_learning_rate, num_examples, warmup_epochs, total_steps, name=None):
#         super().__init__()
#         self.warmup_epochs = warmup_epochs
#         self.base_learning_rate = base_learning_rate
#         self.num_examples = num_examples
#         self.total_steps = total_steps
#         self._name = name

#     def __call__(self, step):
#         with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
#             warmup_steps = int(
#               round(self.warmup_epochs * self.num_examples //
#                     FLAGS.bs))
#         if FLAGS.learning_rate_scaling == 'linear':
#             scaled_lr = self.base_learning_rate * float(FLAGS.bs) / 256.
#         elif FLAGS.learning_rate_scaling == 'sqrt':
#             scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.bs)
#         else:
#             raise ValueError('Unknown learning rate scaling {}'.format(
#                 FLAGS.learning_rate_scaling))

#         learning_rate = tf.cast(
#             tf.cast(step, tf.float32) / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr,
#             tf.float32
#         )

#         # Cosine decay learning rate schedule

#         cosine_decay = tf.keras.experimental.CosineDecay(
#             scaled_lr, self.total_steps - warmup_steps)
#         cosine_decay = tf.cast(cosine_decay(step - warmup_steps), tf.float32)
#         learning_rate = tf.where(
#             step < warmup_steps, learning_rate,
#             cosine_decay)
#         return learning_rate

#     def get_config(self):
#         return {
#             "base_learning_rate": self.base_learning_rate,
#             "num_examples": self.num_examples,
#             "warmup_epochs": self.warmup_epochs,
#             "total_steps": self.total_steps,
#             "hold_base_rate_steps": int(round(self.warmup_epochs * self.num_examples // FLAGS.bs)),
#             "name": self._name
#         }



# def count_ds_size(ds, bs=1e3):
#     count = 0
#     for b in ds.batch(int(bs)):
#         count += len(b)
#     return count


# def get_ds_length(ds_files, max_ex):
#     N = len(ds_files)

#     if max_ex is None:
#         ds = TFRecordDataset(ds_files[0])
#         max_ex = sum((1 for _ in ds))
#     if N == 1:
#         return max_ex

#     ds = TFRecordDataset(ds_files[-1], num_parallel_reads=cpu_count())
#     last_ex = count_ds_size(ds)
#     del ds
#     return ((N-1) * max_ex) + last_ex


# # def read_tfrecord(serialized_example, patch_size):
# #     example = _parse_function(serialized_example, batched=False)
# #     return _sample_from_tfrecord(example, patch_size, batched=False)
# #
# # def build_dataset_shard(filename, patch_size, n_parallel):
# #     ds = tf.data.TFRecordDataset(filename)
# #     ds = ds.map(lambda x: read_tfrecord(x, patch_size), num_parallel_calls=n_parallel)
# #     return ds.repeat()


# def build_dataset(dataset_dir, patch_size, n_parallel=-1, shuffle=False, max_ex=1e5, return_len=False):
#     """ Converts dataset into a distributed dataset according to
#     distribute_datasets_from_function """
#     logging.info("Using Dataset located at %s", dataset_dir)
#     shards = glob(join(dataset_dir, f"dataset_{patch_size}_*.tfrecords"))

#     if n_parallel == 0:
#         n_parallel = None
#     elif n_parallel == -1:
#         n_parallel = tf.data.AUTOTUNE
#     elif n_parallel == -2:
#         n_parallel = cpu_count()

#     # ds = TFRecordDataset(shards, num_parallel_reads=n_parallel)
#     files_ds = tf.data.Dataset.from_tensor_slices(shards)
#     if shuffle:
#         files_ds = files_ds.shuffle(len(shards))
#     ds = files_ds.interleave(
#         lambda x: tf.data.TFRecordDataset(x).repeat(),
#         # lambda x: build_dataset_shard(x, patch_size, n_parallel),
#         num_parallel_calls=n_parallel,
#         deterministic=(not shuffle))
#     logging.info("Data File Path : {}".format(dataset_dir))
#     logging.info("Patch Size: {}".format(patch_size))
#     logging.info("Total Shards: {}".format(len(shards)))

#     total_examples = get_ds_length(shards, max_ex)
#     logging.info("Total Samples: {:.2f}M\n\n".format(total_examples / 1e6))
#     if return_len:
#         return ds, int(total_examples)
#     return ds


# def filter_func_BACH(pf):
#     def filter_fn(file):
#         file_str = file.numpy().decode("utf-8")
#         return (("WSI/A" in file_str) and ("svs" in file_str))

#     return tf.py_function(filter_fn,
#                           [pf["file"]],
#                           [tf.bool])[0]


# def build_dataset_bach(dataset_name, patch_size, parallel=False, num_parallel_calls=None):
#     ds = build_dataset(dataset_name, patch_size, parallel=parallel)
#     ds = ds.map(_parse_function, num_parallel_calls=num_parallel_calls)
#     ds = ds.filter(filter_func_BACH)
#     logging.info("Using supervised BACH dataset")
#     return ds


# def build_dataset_TCGA(dataset_name, patch_size, parallel=False, num_parallel_calls=None):
#     ds = build_dataset(dataset_name, patch_size, parallel=parallel)
#     ds = ds.map(_parse_function, num_parallel_calls=num_parallel_calls)
#     ds = ds.filter(filter_func_TCGA)
#     logging.info("Using unsupervised TCGA dataset")
#     return ds


# def build_distributed_dataset(strategy, dataset):
#     """ distributed training for CPC """
#     dataset_dist = strategy.experimental_distribute_dataset(dataset)
#     return dataset_dist


# def create_tf_example(file, mag, input_s, x, y, label=None, include_labels=False):
#     feature_dict = {
#         'file': Feature(bytes_list=BytesList(value=[file.encode('utf-8')])),
#         'mag': Feature(float_list=FloatList(value=[mag])),
#         'input_s': Feature(int64_list=Int64List(value=[input_s])),
#         'x': Feature(int64_list=Int64List(value=[x])),
#         'y': Feature(int64_list=Int64List(value=[y])),
#     }

#     if include_labels:
#         feature_dict['labels'] = Feature(int64_list=Int64List(value=[label]))

#     tf_example = Example(
#         features=Features(
#             feature=feature_dict
#         )
#     )
#     return tf_example


# def create_tf_example_sup(file, mag, input_s, x, y, gt):
#     tf_example = Example(
#         features=Features(
#             feature={
#                 'file': Feature(bytes_list=BytesList(value=[file.encode('utf-8')])),
#                 'mag': Feature(float_list=FloatList(value=[mag])),
#                 'input_s': Feature(int64_list=Int64List(value=[input_s])),
#                 'x': Feature(int64_list=Int64List(value=[x])),
#                 'y': Feature(int64_list=Int64List(value=[y])),
#                 'labels': Feature(int64_list=Int64List(value=[gt])),
#                 }
#         )
#     )
#     return tf_example


# def _parse_function(proto, batched=False):
#     keys_to_features = {
#                         'file': tf.io.FixedLenFeature([], tf.string),
#                         'mag': tf.io.FixedLenFeature([], tf.float32),
#                         'input_s': tf.io.FixedLenFeature([], tf.int64),
#                         'x': tf.io.FixedLenFeature([], tf.int64),
#                         'y': tf.io.FixedLenFeature([], tf.int64),
#                         }
#     parser = tf.io.parse_example if batched else tf.io.parse_single_example
#     parsed_features = parser(proto, keys_to_features)
#     return parsed_features


# def create_slide_example(file, input_s, x, y):
#     tf_example = Example(
#         features=Features(
#             feature={
#                 'file': Feature(bytes_list=BytesList(value=[file.encode('utf-8')])),
#                 'input_s': Feature(int64_list=Int64List(value=[input_s])),
#                 'x': Feature(int64_list=Int64List(value=x)),
#                 'y': Feature(int64_list=Int64List(value=y)),
#                 }
#         )
#     )
#     return tf_example


# def _parse_function_slide(proto, max_len=512):
#     keys_to_features = {
#         'file': tf.io.FixedLenFeature([], tf.string),
#         'input_s': tf.io.FixedLenFeature([], tf.int64),
#         'x': tf.io.FixedLenFeature([max_len], tf.int64),
#         'y': tf.io.FixedLenFeature([max_len], tf.int64),
#     }

#     parsed_features = tf.io.parse_single_example(proto, keys_to_features)
#     return parsed_features


# def create_feature_example(file, feats):
#     flattened_feats = feats.flatten().astype(np.float64)
#     # Convert file to bytes
#     file_bytes = file.encode('utf-8')

#     # Create a dictionary of features
#     features = {
#         'file': (file_bytes, "byte"),
#         '4kfeatures': (flattened_feats, "float")
#     }

#     return features


# def hash_model_weights(model):
#     # Combine all model weights into a single list
#     weights = []
#     for layer in model.layers:
#         weights += [w.numpy() for w in layer.weights]
#     weights = np.concatenate([w.flatten() for w in weights])

#     # Compute hash of the weights
#     weight_hash = hashlib.sha256(weights.tobytes()).hexdigest()

#     return weight_hash

## SVS FILES ##


def collect_patients_svs_files(patient_list_csv, svs_files):
    """
    Collects the svs files that are in the patient list csv
    
    Args:
        patient_list_csv (str): path to the patient list csv file
        svs_files (iterable): list of svs files

    Returns:
        filtered (dict): dictionary of svs files that are in the patient list csv
    """
    svs_files_df = pd.read_csv(patient_list_csv)
    svs_files_df = svs_files_df.iloc[:, 1:]

    ids_odx = set(svs_files_df.slide)

    filtered = {}
    for file in svs_files:
        file = file.replace("./resources", "/data/resources")
        if file.endswith(".txt"):
            continue
        id_name = splitext(basename(file))[0]
        if id_name in ids_odx:
            patient_data = svs_files_df[svs_files_df.slide == id_name].iloc[0].to_dict()
            cleaned_data = {k: (None if pd.isna(v) else v) for k, v in patient_data.items()}
            filtered[file] = cleaned_data
    return filtered


def log_training_params(logger, params):
    formatted_params = json.dumps(params, indent=4)
    logger.info(f"Training Parameters:")
    logger.info(formatted_params)


def set_seed(seed):
    """
    Set all relevant seeds for reproducibility in Python, NumPy, and PyTorch.
        
    Args:
        seed (int): seed to set
    """
    logger.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_coords_dict(coords_dict: dict, filename: str):
    """
    Save a dictionary where keys are filenames (str) and values are np.ndarrays.
    
    Args:
        coords_dict (dict): Dictionary mapping filenames to (N, 2) NumPy arrays.
        filename (str): Path to the output file (.npz format).
    """
    np.savez_compressed(filename, **coords_dict)


def load_coords_dict(filename: str) -> dict:
    """
    Load a dictionary from a compressed .npz file.

    Args:
        filename (str): Path to the .npz file.

    Returns:
        dict: Dictionary where keys are filenames (str) and values are np.ndarrays.
    """
    data = np.load(filename, allow_pickle=True)
    return {key: data[key] for key in data}


def load_slide_paths(slides_list_file):
    """
    Load slide paths from a list of slides.

    Args:
        slides_list_file (str): json file containing the list of slides.

    Returns:
        slides_dict (dict): dictionary where 
        the keys are slide paths (str) and the values are metadata dictionaries
    """
    if slides_list_file or os.path.isfile(slides_list_file):
        with open(slides_list_file, "r") as f:
            slides_dict = json.load(f)

        if not slides_dict:
            logger.error(f"No slides found in {slides_list_file}")
            raise ValueError(f"No slides found in {slides_list_file}")
        return slides_dict
    else:
        logger.error(f"Slide list file not found: {slides_list_file}")
        raise FileNotFoundError(f"Slide list file not found: {slides_list_file}")
