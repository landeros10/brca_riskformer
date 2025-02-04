from abc import abstractmethod
from absl import flags
import logging
from datetime import datetime, timedelta
import time
from os.path import join, abspath, exists
from os import makedirs

import tensorflow as tf
from tensorflow.keras import mixed_precision as mp
from tensorflow.compat.v2.train import CheckpointManager

import numpy as np
import gc
import pandas as pd
import pickle
from collections.abc import Iterable

from .layers import VisionTransformerWSI, VisionTransformerWSI_256
from ..util import (build_optimizer, WarmUpAndCosineDecay,
                    build_distributed_dataset, hash_model_weights)

FLAGS = flags.FLAGS

def convert_to_soft_label(score, beta=1.50):
    cutoff = 0.7169
    min_score = -2.009
    max_score = 2.744
    if score <= cutoff:
        soft_label = (score - min_score) / (cutoff - min_score)
        return 0.50 * soft_label ** beta
    else:
        soft_label = (score - cutoff) / (max_score - cutoff)
        return 1 - 0.50 * (1 - soft_label) ** beta
    return soft_label


class SlideLevelModel:
    def __init__(self,
                 strategy=None,
                 policy_type=""):
        # self.strategy = strategy if strategy is not None else tf.distribute.MirroredStrategy()
        self.strategy = strategy if strategy is not None else tf.distribute.get_strategy()
        self.policy_type = policy_type
        logging.info("Building model...")
        self.build_model()

    # noinspection PyAttributeOutsideInit
    def build_model(self):
        if self.policy_type == "mixed_float16":
            self.policy = mp.Policy("mixed_float16")
            mp.set_global_policy(self.policy)
        self.define_model()

    def count_params(self):
        params = self.model.trainable_weights
        params_flat = [np.prod(p.get_shape().as_list()) for p in params]
        return np.sum(params_flat)

    def summary(self):
        self.model.summary()

    @abstractmethod
    def define_model(self):
        pass

    def gen_checkpoint(self, optimizer, fold=None, max_to_keep=5):
        directory = self.ckpts if fold is None else f"{self.ckpts}/fold_{fold}"
        if not exists(directory):
            makedirs(directory)

        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              model=self.model)

        self.ckpt_manager = CheckpointManager(self.checkpoint,
                                              directory=directory,
                                              max_to_keep=max_to_keep)

    def restore(self, model_path,
                gen_checkpoint=False,
                optimizer_type=None,
                lr=None,
                fold=None):
        """
        Restores a session from a checkpoint
        :param lr:
        :param optimizer_type:
        :param gen_checkpoint:
        :param model_path: path to file system checkpoint location
        """
        lr = lr if lr is not None else 1e-3
        if gen_checkpoint:
            logging.info("Creating checkpoint manager...")
            self.ckpts = join(model_path, "saved_model")
            if optimizer_type is not None:
                self.optimizer = optimizer_type(learning_rate=lr)
            else:
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.gen_checkpoint(self.optimizer, fold=fold)

        model_directory = self.ckpts if fold is None else f"{self.ckpts}/fold_{fold}"
        latest_ckpt = tf.train.latest_checkpoint(model_directory)
        print("Hash of model weights after gen checkpoint: ", hash_model_weights(self.model))
        self.checkpoint.restore(latest_ckpt)
        print("Hash of model weights after restoration: ", hash_model_weights(self.model))
        logging.info("Model restored from file: %s" % model_directory)

    def log_to_file(self, message, mode="a"):
        with open(self.logFile, mode) as file:
            file.write(message + "\n")

    def parse_train_params(self, hyper_params={}):
        # Can be used in initial dataset preparation
        self.trn_len = FLAGS.train_len
        self.train_round = hyper_params.get("train_round", FLAGS.train_round)
        self.train_steps = hyper_params.get("train_steps", FLAGS.train_steps)

        self.bs = hyper_params.get("bs", FLAGS.bs)
        self.eval_bs = hyper_params.get("eval_bs", FLAGS.eval_bs)
        self.training_rounds = self.train_steps // self.train_round
        self.eval_round = hyper_params.get("eval_round", FLAGS.eval_round)

        self.early_stop_cutoff = hyper_params.get("early_stop_cutoff", FLAGS.early_stop_cutoff)

        self.preFetch = hyper_params.get("prefetch", tf.data.AUTOTUNE)
        self.preFetch_val = hyper_params.get("prefetch", tf.data.AUTOTUNE)
        self.eval_every = hyper_params.get("eval_every", FLAGS.eval_every)

        # Gradient Descent Params\
        self.base_lr = hyper_params.get("learning_rate", FLAGS.learning_rate)
        self.lr = self.base_lr
        self.use_warmup = hyper_params.get("use_warmup", FLAGS.use_warmup)
        self.optimizer_type = hyper_params.get("optimizer", FLAGS.optimizer)
        self.clip_norm = hyper_params.get("clip_norm", FLAGS.clip_norm)
        self.clip_norm = float(self.clip_norm) if self.clip_norm > 0 else None
        self.warmup_epochs = hyper_params.get("warmup_epochs", FLAGS.warmup_epochs)
        if self.use_warmup:
            lr_epoch = self.bs * self.train_round * self.eval_every
            self.lr = WarmUpAndCosineDecay(self.lr,
                                           lr_epoch,
                                           self.warmup_epochs,
                                           FLAGS.train_steps)
        with self.strategy.scope():
            self.optimizer = build_optimizer(self.optimizer_type, self.lr,
                                             clip_norm=self.clip_norm,
                                             decay_steps=self.train_steps)
            if FLAGS.policy == "mixed_float16":
                self.optimizer = mp.LossScaleOptimizer(self.optimizer,
                                                       dynamic=True,
                                                       initial_scale=2 ** 10)
            else:
                print("Not using mixed precision\n")
        self.global_step = self.optimizer.iterations

        #  Regularization parameters
        self.l2_coeff = hyper_params.get("l2_coeff", FLAGS.l2_coeff)
        self.regional_coeff = hyper_params.get("regional_coeff", FLAGS.regional_coeff)
        self.cce_weight = hyper_params.get("cce_weight", [1.0, 1.0])
        self.l1_coeff = hyper_params.get("l1_coeff", 1e-5)
        if self.use_phi:
            self.model.phi.layers[0].kernel_regularizer = tf.keras.regularizers.l1(self.l1_coeff)
        self.drop_path_rate = hyper_params.get("drop_path_rate", 0.1)
        self.drop_rate = hyper_params.get("drop_rate", 0.1)
        self.noise_aug = hyper_params.get("noise", 0.05)
        self.dampen_noise = hyper_params.get("dampen_noise", 0.2)
        self.dampen_fraction = hyper_params.get("dampen_fraction", 0.5)

        self.model.noise_aug = self.noise_aug
        self.model.dampen_noise = self.dampen_noise
        self.model.dampen_fraction = self.dampen_fraction

        #   For saving models
        self.bestLoss = hyper_params.get("bestLoss", FLAGS.bestLoss)
        self.bestAcc = hyper_params.get("bestAcc", FLAGS.bestAcc)

        # Dataset Parameters
        self.num_parallel_calls = FLAGS.num_parallel
        if self.num_parallel_calls < 0:
            self.num_parallel_calls = tf.data.AUTOTUNE

    def setup_training(self, metrics, hyper_params={}):
        # Start log file
        template = (
            "Model Training Details:\n"
            "\n== Training Parameters ==\n"
            "Optimizer: {}, LR {:.2e}, Batch Size: {}, Eval Batch Size: {}\n"
            "Train Steps: {:d} x {}, Eval Steps: {:d}\n"
            "Warmup Epochs: {:d} ({})\n"
            "Best Loss: {:.2f}, Best Acc: {:.2f}\n"
            "Training Dataset Length: {}\n"

            "\n== Regularization and Augmentation Parameters ==\n"
            "l2_coeff: {:.2e}\n"
            "l1_coeff: {:.2e}\n"
            "noise_aug: {:.2f}\n"
            "dampen_noise: {:.2f}\n"
            "dampen_fraction: {:.2f}\n"
            "Output Dim: {}\n"

            "\n== Paths ==\n"
            "Output Path: {}\n"
            "Log Path: {}\n"
            "\n"
        )

        template = template.format(
            self.optimizer_type,
            self.base_lr,
            self.bs,
            self.eval_bs,
            self.train_round,
            self.eval_every,
            self.eval_round,
            self.warmup_epochs,
            self.use_warmup,
            self.bestLoss,
            self.bestAcc,
            self.trn_len,

            self.l2_coeff,
            self.l1_coeff,
            self.noise_aug,
            self.dampen_noise,
            self.dampen_fraction,

            self.num_classes,

            self.out_path,
            self.log_path,
        )
        template +=  "\n== All Other Parameters ==\n"
        template += '\n'.join(f'{key}: {value}' for key, value in hyper_params.items())

        self.log_to_file(template)
        self.log_to_file(FLAGS.log_note + "\n")

        columns = ["step", "t",
                   *[m.name.replace("/", "_") for m in metrics["train"].values()],
                   *[m.name.replace("/", "_") for m in metrics["test"].values()]
                   ]
        log_df = pd.DataFrame(columns=columns)
        with open(self.logFile_df, "wb") as handle:
            pickle.dump(log_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def define_loss(self):
        """Store loss and accuracy functions. """
        # Any funcs needed by self.eval_loss
        self.loss = lambda input, y_true, y_pred: None

        # Must be defined
        self.trn_loss = None
        self.tst_loss = None
        self.metrics = {"train": [self.trn_loss], "test": [self.tst_loss]}

    @abstractmethod
    def eval_loss(self,
                  input: tf.Tensor,
                  y_true: tf.Tensor,
                  y_pred: tf.Tensor):
        """Uses loss funcs defined in self.define_loss to return loss"""
        return self.loss(input, y_true, y_pred)

    def get_grads(self, gt, loss):
        if self.policy_type == "mixed_float16":
            scaled_loss = self.optimizer.get_scaled_loss(loss)
            scaled_grads = gt.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = gt.gradient(loss, self.model.trainable_variables)
        gradient_pairs = zip(grads, self.model.trainable_variables)
        return grads, gradient_pairs

    @abstractmethod
    def reset_metrics(self):
        pass

    @abstractmethod
    def output_stats(self, step):
        pass

    def log_and_write_metrics(self, global_step, evalTime):
        vals = [global_step.numpy(), evalTime]
        vals.extend(
            [m.result().numpy().astype(float) for m in self.metrics["train"].values()])
        vals.extend(
            [m.result().numpy().astype(float) for m in self.metrics["test"].values()])

        for i in range(len(vals)):
            if isinstance(vals[i], Iterable):
                vals[i] = vals[i][-1]

        log_df = pickle.load(open(self.logFile_df, "rb"))
        temp_df = pd.DataFrame(dict(zip(log_df.columns, [[v] for v in vals])))
        log_df = pd.concat([log_df, temp_df], ignore_index=True)
        with open(self.logFile_df, "wb") as handle:
            pickle.dump(log_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_checkpoints(self, save_metric='mean_loss'):

        metric = self.metrics["test"][save_metric].result().numpy().astype(float)
        if isinstance(metric, Iterable):
            metric = metric[-1]

        cond = (metric < self.bestLoss) if ('loss' in save_metric) else (metric > self.bestLoss)
        if cond:
            self.miss_count = 0
            with self.strategy.scope():
                self.ckpt_manager.save()
            logging.info("Saved Model!")

            f = open(self.logFile, "a")
            f.write("\t\tSaved Model!")
            f.close()

            self.bestLoss = metric
        else:
            self.miss_count += 1

    def check_early_stop(self):
        return self.miss_count > self.early_stop_cutoff

    def check_nans(self):
        """Check if the loss has become NaN and if so, raise an error."""
        if tf.math.is_nan(self.metrics["train"]['mean_loss'].result().numpy()) or \
                tf.math.is_nan(self.metrics["test"]['mean_loss'].result().numpy()):
            raise ValueError("Loss became NaN")


class RS_Predictor_ViT(SlideLevelModel):
    def __init__(self, in_dim, out_dim, drop_path_rate, drop_rate,
                 num_embed, num_classes, max_dim,
                 depth, num_heads,
                 log_path, out_path, data_dir):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        self.num_embed = num_embed
        self.num_classes = num_classes
        self.max_dim = max_dim

        self.depth = depth
        self.num_heads = num_heads

        # Paths to save files for documenting traing progress
        self.out_path = out_path
        self.log_path = log_path
        self.ckpts = join(self.out_path, "saved_model")
        self.data_dir = data_dir

        log = "log-" + datetime.today().strftime("%y-%b-%d-%H")
        self.logFile = join(self.log_path, log + ".txt")
        self.logFile_df = join(self.log_path, log + ".pickle")

        if not exists(abspath(self.log_path)):
            logging.info("Allocating '{:}'".format(self.log_path))
            makedirs(abspath(self.log_path))

        if not exists(abspath(self.out_path)):
            logging.info("Allocating '{:}'".format(self.out_path))
            makedirs(abspath(self.out_path))

        super().__init__()

    def define_model(self):
        with self.strategy.scope():
            name = "wsi"
            self.model = VisionTransformerWSI(
                input_embed_dim=self.in_dim,
                output_embed_dim=self.out_dim,
                num_patches=self.num_embed,
                max_dim=self.max_dim,
                drop_path_rate=self.drop_path_rate,
                drop_rate=self.drop_rate,
                num_classes = self.num_classes,
                depth=self.depth,
                num_heads=self.num_heads,
                name=f'{name}_vit'
            )

            # Log VisionTransformer4K configuration
            self.log_to_file(f"VisionTransformer config for {name}:", mode="w")
            model_config = self.model.get_config()
            for k, v in model_config.items():
                self.log_to_file(f"\t{k}: {v}")

            # Create dummy data and initialize model
            dummy_input = tf.random.normal([2, self.num_embed, self.in_dim])
            _ = self.model(dummy_input)

    def define_loss(self):
        with self.strategy.scope():
            # Define metric names and types
            metric_names = [
                'mean_loss', 'loss_local', 'loss_global', 'loss_l2', 'loss_l1',
                'auc', 'f1', 'recall', 'acc'
                ]
            metric_types = {
                'mean_loss': tf.keras.metrics.Mean,
                'loss_local': tf.keras.metrics.Mean,
                'loss_global': tf.keras.metrics.Mean,
                'loss_l2': tf.keras.metrics.Mean,
                'loss_l1': tf.keras.metrics.Mean,
                'auc': tf.keras.metrics.AUC,
                'f1': lambda name: tf.keras.metrics.F1Score(name=name, threshold=0.5),
                'recall': lambda name: tf.keras.metrics.Recall(name=name, thresholds=0.5),
                'acc': tf.keras.metrics.BinaryAccuracy
            }

            # Initialize metrics for train and test
            self.metrics = {'train': {}, 'test': {}}
            for metric_name in metric_names:
                for mode in ['train', 'test']:
                    metric_full_name = f'{metric_name}_{mode}'
                    metric = metric_types[metric_name](name=metric_full_name)
                    setattr(self, metric_full_name, metric)
                    self.metrics[mode][metric_name] = getattr(self, metric_full_name)

            # Binary Crossentropy Loss
            self.cce = tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE)

    def eval_loss(self, predictions: tf.Tensor, labels: tf.Tensor,
                  update_metrics=True, mode="train") -> tf.Tensor:
        cce_loss = self.cce(labels, predictions)

        l2s = [tf.nn.l2_loss(v) for v in self.model.trainable_weights
               if 'batch_norm' not in v.name]
        l2_loss = self.l2_coeff * tf.cast(tf.add_n(l2s), cce_loss.dtype)
        loss = cce_loss + l2_loss

        if update_metrics:
            self.update_metrics(loss, labels, predictions, mode=mode)
            return loss

    def update_metrics(self, results):
        losses, labels, predictions, mode = results
        # mode = mode.numpy().decode("utf-8")
        predictions = tf.stack([1-predictions, predictions], axis=0)[tf.newaxis, ...]

        labels = tf.cast(labels, tf.float32)
        predictions = tf.cast(predictions, tf.float32)
        binary_labels = tf.cast(labels >= 0.5, labels.dtype)  # Convert to binary labels

        # Update the loss metrics
        loss_index = 0
        for metric_name in self.metrics[mode]:
            if 'loss' in metric_name:
                self.metrics[mode][metric_name].update_state(losses[loss_index])
                loss_index += 1

        # Update other metrics
        for metric_name in self.metrics[mode]:
            if 'loss' not in metric_name:
                metric = self.metrics[mode][metric_name]
                if isinstance(metric, tf.keras.metrics.Recall):
                    metric.update_state(binary_labels[:, 1], predictions[:, 1])
                else:
                    metric.update_state(binary_labels, predictions)

    def reset_metrics(self):
        for mode in ["train", "test"]:
            for metric_name in self.metrics[mode]:
                self.metrics[mode][metric_name].reset_states()

    def output_stats(self, global_step):
        template = "Iter {}, Loss (Trn/ Tst): ({:.4f}, {:.4f}) "
        template += "Acc: ({:.3f} , {:.3f}) "
        template += "F1: ({:.3f} , {:.3f})"
        trn_loss = self.mean_loss_train.result().numpy().astype(np.float32)
        tst_loss = self.mean_loss_test.result().numpy().astype(np.float32)

        trn_acc = self.acc_train.result().numpy().astype(np.float32)
        tst_acc = self.acc_test.result().numpy().astype(np.float32)

        trn_f1 = self.f1_train.result().numpy().astype(np.float32)[-1]
        tst_f1 = self.f1_test.result().numpy().astype(np.float32)[-1]

        template = template.format(
            global_step.numpy(),
            trn_loss, tst_loss,
            trn_acc, tst_acc,
            trn_f1, tst_f1,
        )
        logging.info(template)

    def train(self, ds, train_steps, val_ds=None, num_versions=1, hyper_params={},
              restore=False, cache_dataset=False, shuffle_buffer=0,
              save_checkpoints=True, fold=None,
              save_metric='mean_loss'):
        hyper_params["train_steps"] = train_steps
        self.parse_train_params(hyper_params=hyper_params)
        self.define_loss()
        self.setup_training(self.metrics, hyper_params=hyper_params)

        logging.info("\nSet up Training! Preparing Datasets...")

        if val_ds is None:
            n_eval = self.eval_round * self.eval_bs
            val_ds = ds.take(n_eval)
            ds = ds.skip(n_eval)
            if self.trn_len > 0:
                ds = ds.take(self.trn_len)

        if cache_dataset: # cache if cache_dataset is True
            ds = ds.cache()
            val_ds = val_ds.cache()

        # Set up batches, example parsing, augmentation, etc.
        if shuffle_buffer > 0:
            ds = ds.shuffle(shuffle_buffer)
        ds = ds.repeat()
        if self.bs > 1:
            ds = ds.batch(self.bs)
        ds = ds.prefetch(buffer_size=self.preFetch)

        val_ds = val_ds.repeat()
        if self.eval_bs > 1:
            val_ds = val_ds.batch(self.eval_bs)
        val_ds = val_ds.prefetch(buffer_size=self.preFetch)

        with self.strategy.scope():
            self.gen_checkpoint(self.optimizer, fold=fold)
            if restore:
                self.restore(self.out_path, gen_checkpoint=False, fold=fold)

        self.train_on_dist_dataset(ds, val_ds, train_steps,
                                   save_checkpoints=save_checkpoints,
                                   save_metric=save_metric)
        del ds, val_ds
        gc.collect()

    def train_on_dist_dataset(self, trn, val, train_steps,
                              save_checkpoints=True,
                              save_metric='mean_loss'):
        logging.info("Updates per training cycle: {}".format(self.train_round))
        logging.info("Updates per evaluation cycle: {}".format(self.eval_round))
        self.epoch = 0
        if train_steps == 0:
            self.early_stop = True
            return

        with self.strategy.scope():
            def trn_step(data_batch):
                # inputs, label = data_batch
                with tf.GradientTape() as gt:
                    inputs = data_batch["inputs"]
                    labels = data_batch["label"]
                    preds = self.model(inputs)
                    results, loss = self.eval_loss(preds, labels,
                                          update_metrics=True,
                                          mode="train")
                    self.update_metrics(results)

                    # update the model's gradients
                    grads, gradient_pairs = self.get_grads(gt, loss)
                    self.optimizer.apply_gradients(gradient_pairs)
                grad_norms = [tf.norm(g) for g in grads if g is not None]
                if any(tf.math.is_nan(norm) for norm in grad_norms):
                    logging.warning("NAN detected in gradients!")
                if max(grad_norms) < 1e-8: # any(norm < 1e-8 for norm in grad_norms):
                    logging.warning("Potential vanishing gradient detected!")
                if any(norm > 1e5 for norm in grad_norms):
                    logging.warning("Potential exploding gradient detected!")

                return loss

            def val_step(data_batch):
                # inputs, labels = data_batch
                inputs = data_batch["inputs"]
                labels = data_batch["label"]

                preds = self.model(inputs)
                results, loss = self.eval_loss(preds, labels,
                                      update_metrics=True,
                                      mode="test")
                self.update_metrics(results)
                return loss

            # @tf.function
            def dist_trn_step(iterator):
                for _ in tf.range(self.train_round):
                    data_batch = next(iterator)
                    self.strategy.run(trn_step, args=(data_batch,))

            # @tf.function
            def dist_val_step(iterator):
                for _ in tf.range(self.eval_round):
                    data_batch = next(iterator)
                    self.strategy.run(val_step, args=(data_batch,))

            trn_dist = iter(build_distributed_dataset(self.strategy, trn))
            val_dist = iter(build_distributed_dataset(self.strategy, val))

        logging.info("Starting training!")
        startTime = time.time()
        self.global_step = self.optimizer.iterations
        self.miss_count = 0
        self.early_stop = False
        for epoch in range(self.training_rounds):
            self.epoch = epoch // self.eval_every
            with self.strategy.scope():
                dist_trn_step(trn_dist)
                if epoch % self.eval_every == 0:
                    logging.info("Eval..")
                    dist_val_step(val_dist)
                    self.output_stats(self.global_step)
                try:
                    self.check_nans()
                except ValueError:
                    logging.info("Stopping early cuz NaNs")
                    self.early_stop = True
                    return

            if epoch % self.eval_every == 0:
                evalTime = (time.time() - startTime) / 60
                self.log_and_write_metrics(self.global_step, evalTime)
                if epoch >= 1:
                    if save_checkpoints:
                        self.save_checkpoints(save_metric)
                    if self.check_early_stop():
                        logging.info("Optimization Finished Early!")
                        self.early_stop = True
                        return
            with self.strategy.scope():
                self.reset_metrics()

        total_time = time.time() - startTime
        total_time_str = str(timedelta(seconds=int(total_time)))
        logging.info('Training time {}'.format(total_time_str))

    def train_kfold(self, ds, train_steps,
                    k=5, n=None, total_size=None,
                    hyper_params={},
                    shuffle_ds=True,
                    save_metric='mean_loss'):
        """
        Train using k-fold cross-validation.

        :param ds: A tf.data.Dataset object.
        :param k: Number of folds. Default is 5.
        :param kwargs: Additional arguments to pass to the train() method.
        :return: A tuple containing lists of best losses and accuracies for each fold.
        """

        # Get the total size of the dataset
        total_size = sum(1 for _ in iter(ds)) if total_size is None else total_size
        fold_size = total_size // k

        fold_losses = []
        fold_accuracies = []

        for fold in range(k):
            logging.info(f"Training for Fold {fold + 1}/{k} ...")

            # Determine start and end indices for validation segment
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size

            val_ds = ds.skip(val_start).take(fold_size)
            trn_ds = ds.take(val_start).concatenate(ds.skip(val_end))

            # Set train_round and eval_round for each fold:
            val_size = fold_size
            trn_size = total_size - fold_size
            hyper_params["train_round"] = trn_size // hyper_params["bs"]
            hyper_params["eval_round"] = val_size // hyper_params["eval_bs"]

            # Train using the derived train and validation datasets
            self.train(trn_ds, train_steps, val_ds=val_ds,
                       restore=False,
                       shuffle_buffer=trn_size if shuffle_ds else 0,
                       cache_dataset=False,
                       save_checkpoints=True,
                       hyper_params=hyper_params,
                       fold=fold+1,
                       save_metric=save_metric)

            # Store the best loss and accuracy for this fold
            fold_losses.append(self.bestLoss)
            fold_accuracies.append(self.bestAcc)

            # Reset the model for the next fold
            self.model = None  # If you have a model attribute in your class
            tf.keras.backend.clear_session()  # Clearing TensorFlow session to free up resources
            gc.collect()  # Explicitly calling garbage collector
            self.build_model()  # Building your model for the new set of hyperparameters

            if n is not None:
                if k + 1 >= n:
                    break
        return fold_losses, fold_accuracies

    def parameter_search(self, ds, size, ranges, train_steps, best_loss,
                         save_dir,
                         n_iter=100,
                         k=5,
                         n=None,
                         standard_bs=128,
                         save_file_name="best_hypers.feather",
                         read_previous=False,
                         save_metric='mean_loss'):
        """Uses a smaller dataset to evaluate the hyperparameters
        in the training paradigm.

        Args:
            ds tf.data.Dataset: full dataset
            size int: number of examples to keep for hyperparameter tuning
            ranges dict: dictionary that pairs hyperparameter to ranges
        Returns:
            optimums dict: dictionary with the best hyperparameter values

        """
        self.train_steps = train_steps
        if size > 0:
            ds = ds.take(size)
        param_names = list(ranges.keys())
        save_file = join(save_dir, save_file_name)
        if not read_previous:
            # Define column names
            logging.info("Creating our log file for parameter search")
            columns = param_names + ['loss', 'acc', 'iter', 'steps', 'early_stop']
            df = pd.DataFrame(columns=columns).reset_index(drop=True)
            df.to_feather(save_file)

        best_params = None
        for i in range(n_iter):
            hyper_params = {
                param: np.random.choice(ranges[param]) for param in param_names
            }
            hyper_params["eval_every"] = 1
            f = hyper_params["pos_frac"]
            hyper_params["cce_weight"] = [(1 / (2 * (1-f))), (1 / (2 * f))]
            hyper_params["bestLoss"] = 0.0
            logging.info(f"Iter: {i}\nParams:\n{hyper_params}")
            df = pd.read_feather(save_file)
            new_row = {**hyper_params,
                       "loss": np.nan,
                       "acc": np.nan,
                       "iter": i,
                       "steps": np.nan,
                       "early_stop": np.nan}
            df = df.append(new_row, ignore_index=True).reset_index(drop=True)
            df.to_feather(save_file)

            self.model = None  # If you have a model attribute in your class
            tf.keras.backend.clear_session()  # Clearing TensorFlow session to free up resources
            gc.collect()  # Explicitly calling garbage collector
            self.build_model()  # Building your model for the new set of hyperparameters

            # mask_num
            self.model.set_mask_num(hyper_params["mask_num"])

            # Adjust Train Steps for a standard of 128 examples per step
            new_train_steps = int(train_steps * float(standard_bs) / hyper_params["bs"])
            logging.info(f"Training for {new_train_steps} steps..")

            # Start Training with k-fold cross validation
            fold_losses, fold_accuracies = self.train_kfold(
                ds,
                new_train_steps,
                k=k,
                n=n,
                hyper_params=hyper_params,
                save_metric=save_metric)

            # Compute mean loss and accuracy across the k folds
            loss = np.mean(fold_losses)
            acc = np.mean(fold_accuracies)

            df = pd.read_feather(save_file)
            new_row = {**hyper_params,
                       "loss": loss,
                       "acc": acc,
                       "iter": i,
                       "steps": int(self.optimizer.iterations.numpy()),
                       "early_stop": self.early_stop}
            df = df.append(new_row, ignore_index=True).reset_index(drop=True)
            df.to_feather(save_file)

            cond = (loss < best_loss) if ('loss' in save_metric) else (loss > best_loss)
            if cond:
                best_loss = loss
                best_params = hyper_params
                print(f"\n\nNew best found: loss = {best_loss}, params = {best_params}\n\n")
        return best_params


class RS_Predictor_ViT_RPE(RS_Predictor_ViT):
    def __init__(self, in_dim, phi_dim, use_phi, out_dim,
                 drop_path_rate, drop_rate,
                 num_classes, max_dim, mask_num, mask_preglobal,
                 depth, global_depth, num_heads, use_attn_mask, mlp_ratio,
                 encoding_method, use_class_token,
                 use_nystrom, num_landmarks, global_k,
                 log_path, out_path, data_dir):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.phi_dim = phi_dim
        self.use_phi = use_phi
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        self.num_classes = num_classes
        self.max_dim = max_dim
        self.mask_num = mask_num
        self.mask_preglobal = mask_preglobal
        self.depth = depth
        self.global_depth = global_depth
        self.num_heads = num_heads
        self.use_attn_mask = use_attn_mask
        self.mlp_ratio = mlp_ratio
        self.encoding_method = encoding_method
        self.use_class_token = use_class_token
        self.use_nystrom = use_nystrom
        self.num_landmarks = num_landmarks
        self.global_k = global_k

        # Paths to save files for documenting traing progress
        self.out_path = out_path
        self.log_path = log_path
        self.ckpts = join(self.out_path, "saved_model")
        self.data_dir = data_dir

        log = "log-" + datetime.today().strftime("%y-%b-%d-%H")
        self.logFile = join(self.log_path, log + ".txt")
        self.logFile_df = join(self.log_path, log + ".pickle")

        if not exists(abspath(self.log_path)):
            logging.info("Allocating '{:}'".format(self.log_path))
            makedirs(abspath(self.log_path))

        if not exists(abspath(self.out_path)):
            logging.info("Allocating '{:}'".format(self.out_path))
            makedirs(abspath(self.out_path))

        SlideLevelModel.__init__(self)

    def define_model(self):
        with self.strategy.scope():
            name = "wsi"
            self.model = VisionTransformerWSI(
                input_embed_dim=self.in_dim,
                phi_dim=self.phi_dim,
                use_phi=self.use_phi,
                max_dim=self.max_dim,
                output_embed_dim=self.out_dim,
                drop_path_rate=self.drop_path_rate,
                drop_rate=self.drop_rate,
                num_classes = self.num_classes,
                depth=self.depth,
                global_depth=self.global_depth,
                encoding_method=self.encoding_method,
                mask_num=self.mask_num,
                mask_preglobal=self.mask_preglobal,
                num_heads=self.num_heads,
                use_attn_mask=self.use_attn_mask,
                mlp_ratio=self.mlp_ratio,
                use_class_token=self.use_class_token,
                use_nystrom=self.use_nystrom,
                num_landmarks=self.num_landmarks,
                global_k=self.global_k,
                name=f'{name}_vit'
            )

            # Log VisionTransformer4K configuration
            self.log_to_file(f"VisionTransformer config for {name}:", mode="w")
            model_config = self.model.get_config()
            for k, v in model_config.items():
                self.log_to_file(f"\t{k}: {v}")

            # Create dummy data and initialize model
            dummy_input = tf.random.normal([1, 288, 288, self.in_dim])
            _ = self.model(dummy_input, training=True)

    def get_class_weight(self, cce_weights, label):
        label = tf.cast(label >= 0.5, tf.int32)
        return cce_weights[0] * (1 - tf.cast(label, tf.float32)) + cce_weights[1] * tf.cast(label, tf.float32)

    # @tf.function
    def eval_loss(self, predictions, label,
                  update_metrics=True, mode="train") -> tf.Tensor:
        """
        We assume a batchsize of 1 for all operations.  predictions is a
        vector of class predictions where predictions[0] is the prediction
        after the global head, and each predictions[1:] are pseudo-bag-level
        predictions. Label is in {0, 1} and is our risk assessment ground truth
        """
        # class_weight = self.cce_weight[1] if label == 1 else self.cce_weight[0]
        class_weight = self.get_class_weight(self.cce_weight, label)
        class_weight = tf.cast(class_weight, predictions.dtype)


        label = tf.stack([1-label, label], axis=0)[tf.newaxis, ...]
        # regional_coeff = 0
        # if self.epoch % 5 == 0 and self.epoch >= 5:
        regional_coeff = self.regional_coeff

        # Global Loss
        global_pred = predictions[0, :]
        global_loss = self.cce(label, global_pred)
        global_loss *= (1 - regional_coeff) * class_weight

        # Pseudo-bag Loss
        # local_loss = 0
        # eps = 1e-7
        # bag_negative_probs = tf.reduce_prod(predictions[1:, 0])
        # bag_positive_probs = 1 - bag_negative_probs
        #
        # y = tf.cast(label[0, 1], bag_negative_probs.dtype)
        # local_loss += -y * tf.math.log(bag_positive_probs + eps)
        # local_loss -= (1 - y) * tf.math.log(bag_negative_probs + eps)

        # Max pooling loss
        # max_positive_idx = tf.argmax(predictions[1:, 1], axis=0)
        # bag_positive_probs = predictions[1:, :][max_positive_idx]
        # local_loss = self.cce(label, bag_positive_probs)

        # top k loss
        total_instances = tf.shape(predictions)[0] - 1
        k = tf.cast(tf.maximum(1, total_instances // 10), tf.int32)
        top_k_values, top_k_indices = tf.nn.top_k(predictions[1:, 1], k=k)
        top_k_preds = tf.gather(predictions[1:, :], top_k_indices)

        local_loss = self.cce(tf.tile(label, [k, 1]), top_k_preds)
        local_loss = tf.reduce_mean(local_loss)
        local_loss *= regional_coeff * class_weight

        l2s = [tf.nn.l2_loss(v) for v in self.model.trainable_weights
               if 'batch_norm' not in v.name]
        l2_loss = self.l2_coeff * tf.cast(tf.add_n(l2s), local_loss.dtype)

        l1_phi = tf.add_n(self.model.phi.losses) if (self.use_phi and self.model.phi.losses) else 0
        loss = global_loss + local_loss + l2_loss + l1_phi

        # if mode == "test":
        #     phi_weights = self.model.phi.layers[0].get_weights()[0]  # Get the weights (ignoring biases)
        #     non_zero_counts = np.sum(np.abs(phi_weights) > 1e-2, axis=0)
        #     print("Non-zero weights count for each set:\n", non_zero_counts[:16])
        #
        #     global_weights = self.model.global_attn.layers[0].get_weights()[0]  # Get the weights (ignoring biases)
        #     non_zero_counts = np.sum(np.abs(global_weights) > 1e-2, axis=0)
        #     print("Non-zero weights count for each set:\n", non_zero_counts[:16])

        if update_metrics:
            global_acc_pred = predictions[0, 1]
            local_acc_pred = tf.reduce_mean(top_k_values)
            acc_pred = ((1 - regional_coeff) * global_acc_pred) + (regional_coeff * local_acc_pred)
            losses = [loss, local_loss, global_loss, l2_loss, l1_phi]
            return (losses, label, acc_pred, mode), loss
        return loss


class RS_Predictor_ViT_RPE_256(RS_Predictor_ViT_RPE):

    def __init__(self, *args,
                 downscale_depth=1,
                 downscale_multiplier=1.25,
                 noise_aug = 0.1,
                 **kwargs):

        self.downscale_depth = downscale_depth
        self.downscale_multiplier = downscale_multiplier
        self.noise_aug = noise_aug
        super().__init__(*args, **kwargs)

    def define_model(self):
        with self.strategy.scope():
            name = "wsi"
            self.model = VisionTransformerWSI_256(
                input_embed_dim=self.in_dim,
                phi_dim=self.phi_dim,
                use_phi=self.use_phi,
                output_embed_dim=self.out_dim,
                drop_path_rate=self.drop_path_rate,
                drop_rate=self.drop_rate,
                num_classes = self.num_classes,
                max_dim=self.max_dim,
                depth=self.depth,
                global_depth=self.global_depth,
                encoding_method=self.encoding_method,
                mask_num=self.mask_num,
                mask_preglobal=self.mask_preglobal,
                num_heads=self.num_heads,
                use_attn_mask=self.use_attn_mask,
                mlp_ratio=self.mlp_ratio,
                use_class_token=self.use_class_token,
                use_nystrom=self.use_nystrom,
                num_landmarks=self.num_landmarks,
                global_k=self.global_k,
                downscale_depth=self.downscale_depth,
                downscale_multiplier=self.downscale_multiplier,
                noise_aug=self.noise_aug,
                data_dir=self.data_dir,
                attnpool_mode="conv",
                name=f'{name}_vit_256'
            )

            # Log VisionTransformer4K configuration
            self.log_to_file(f"VisionTransformer config for {name}:", mode="w")
            model_config = self.model.get_config()
            for k, v in model_config.items():
                self.log_to_file(f"\t{k}: {v}")

            # Create dummy data and initialize model
            dummy_input = tf.random.normal([45, self.max_dim, self.max_dim, self.in_dim])
            _ = self.model(dummy_input, training=True)
            _ = self.model(dummy_input, training=False)
