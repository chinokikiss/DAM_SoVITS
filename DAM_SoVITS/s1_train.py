import os
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
import argparse
import logging
import platform
from pathlib import Path
import sys
import time
import shutil
import torch
import matplotlib.pyplot as plt
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from collections import OrderedDict

from AR.utils import get_newest_ckpt

def my_save(fea, path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (time.time())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


def get_dist_strategy_and_devices():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s) available.")
        if gpu_count > 1:
            print("Environment supports DDP multi-GPU training, configuring DDP strategy...")
            backend = "gloo" if platform.system() == "Windows" else "nccl"
            strategy = DDPStrategy(
                process_group_backend=backend,
                find_unused_parameters=False
            )
            devices = -1
            return devices, strategy, "gpu"
        
        else:
            print("Only one GPU detected, using single GPU mode with auto strategy.")
            return 1, "auto", "gpu"
    else:
        print("No GPU detected, using CPU mode.")
        return 1, "auto", "cpu"

class TrainingPlotCallback(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.train_timer = time.time()
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'epochs': []
        }
    
    def on_train_epoch_end(self, trainer, pl_module):
        if (time.time()-self.train_timer)/3600 > 29:
            sys.exit()

        self.metrics['epochs'].append(trainer.current_epoch)
        for key in trainer.logged_metrics:
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(trainer.logged_metrics[key].item())
    
    def on_train_end(self, trainer, pl_module):
        self.plot_metrics()
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for key, values in self.metrics.items():
            if key == 'epochs' or len(values) == 0:
                continue
            
            if plot_idx < len(axes):
                axes[plot_idx].plot(self.metrics['epochs'], values, marker='o')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel(key)
                axes[plot_idx].set_title(f'{key} over epochs')
                axes[plot_idx].grid(True)
                plot_idx += 1
        
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300)
        plt.close()

class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if (
                    self.if_save_latest == True
                ):
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)

                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    plot_callback = TrainingPlotCallback(output_dir)
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"
    devices, strategy, accelerator = get_dist_strategy_and_devices()
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator=accelerator,
        limit_val_batches=0,
        devices=devices, 
        benchmark=False,
        fast_dev_run=False,
        strategy=strategy, 
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback, plot_callback],
        use_distributed_sampler=False,
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir, is_train=True)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        data_path=config["data_path"],
    )

    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


# python DAM_SoVITS\s1_train.py -c DAM_SoVITS\configs\s1.yaml
# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1.yaml",
        help="path of config file",
    )

    parser.add_argument("--pretrained_s1", type=str, default="")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--save_every_n_epoch", type=int, default=5)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--min_mask_ratio", type=float, default=0.1)
    parser.add_argument("--max_mask_ratio", type=float, default=1.0)
    parser.add_argument("--duration_loss_weight", type=float, default=0.2)
    parser.add_argument("--if_save_latest", type=bool, default=True)
    parser.add_argument("--if_save_every_weights", type=bool, default=True)
    parser.add_argument("--half_weights_save_dir", type=str, default="weights")
    parser.add_argument("--exp_name", type=str, default="dam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_init", type=float, default=0.00001)
    parser.add_argument("--lr_end", type=float, default=0.0001)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--decay_steps", type=int, default=40000)

    args = parser.parse_args()
    logging.info(str(args))
    config = load_yaml_config(args.config_file)

    if args.pretrained_s1 is not None:
        config['pretrained_s1'] = args.pretrained_s1
    if args.data_path is not None:
        config['data_path'] = args.data_path
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir

    train_args = [
        'seed', 'epochs', 'batch_size', 'gradient_accumulation',
        'save_every_n_epoch', 'precision', 'gradient_clip',
        'min_mask_ratio', 'max_mask_ratio', 'duration_loss_weight',
        'if_save_latest', 'if_save_every_weights', 'half_weights_save_dir', 'exp_name'
    ]
    if 'train' not in config: config['train'] = {}
    for arg_name in train_args:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            config['train'][arg_name] = arg_value

    optimizer_args = ['lr', 'lr_init', 'lr_end', 'warmup_steps', 'decay_steps']
    if 'optimizer' not in config: config['optimizer'] = {}
    for arg_name in optimizer_args:
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            config['optimizer'][arg_name] = arg_value

    main(config)
