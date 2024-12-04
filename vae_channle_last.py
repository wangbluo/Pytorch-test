import gc
import math
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()

import tensordict as td
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from colossalai.booster import Booster
from colossalai.utils import set_seed
from peft import LoraConfig
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, build_module

# from opensora.utils.ckpt import CheckpointIO, model_sharding, record_model_param_shape, rm_checkpoints
from opensora.utils.config import config_to_name, create_experiment_workspace, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import (
    NsysProfiler,
    Timers,
    all_reduce_mean,
    create_tensorboard_writer,
    is_log_process,
    is_pipeline_enabled,
    log_cuda_max_memory,
    log_cuda_memory,
    log_model_params,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.sampling import get_res_lin_function, pack, prepare, time_shift
from opensora.utils.train import create_colossalai_plugin, setup_device
from time import time


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    # checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))

    # == init ColossalAI booster ==
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin = create_colossalai_plugin(
        plugin=plugin_type,
        dtype=cfg.get("dtype", "bf16"),
        grad_clip=cfg.get("grad_clip", 0),
        **plugin_config,
    )
    booster = Booster(plugin=plugin)
    # set_torch_compile_flags()

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),  # useful for automatic restart to specify the exp_name
    )

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "Open-Sora"),
                name=exp_name,
                config=cfg.to_dict(),
                dir=exp_dir,
            )

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=cfg.get("pin_memory", True),
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )

    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    # num_steps_per_epoch = len(dataloader)
    dataloader_iter = iter(dataloader)
    print(
        "==debug== dataloader_iter",
    )

    # == buildn autoencoder ==
    model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    del model_ae.decoder
    log_cuda_memory("autoencoder")
    log_model_params(model_ae)
    for name, param in model_ae.named_parameters():
        if param.ndim == 4:  
            param.data = param.to(memory_format=torch.channels_last)
        elif param.ndim == 5: 
            param.data = param.to(memory_format=torch.channels_last_3d)

    model_ae = torch.compile(model_ae, mode="max-autotune", fullgraph=True, dynamic=True)

    # == boosting ==
    torch.set_default_dtype(dtype)
    torch.set_default_dtype(torch.float)
    log_cuda_memory("boost")

    timers = Timers(
        record_time=cfg.get("record_time", False),
        record_barrier=cfg.get("record_barrier", False),
    )
    nsys = NsysProfiler(warmup_steps=2, num_steps=30, enabled=False)

    dummy_data = cfg.get("dummy_data", False)
    if dummy_data:
        print("==debug== using dummy_data")

    # debug with real data loader
    @torch.no_grad()
    def prepare_inputs(batch, dummy_data=False):
        inp = dict()
        if not dummy_data:
            x = batch.pop("video")
            bs = x.shape[0]
        else:
            x = batch["video"]
            bs = x.shape[0]
        

        x = x.to(memory_format=torch.channels_last_3d)
        # == encode video ==
        with nsys.range("encode_video"), timers["encode_video"]:
            x_0 = model_ae.encode(x)

        return (inp, x_0)

    # =======================================================
    # 6. training loop
    # =======================================================
    dist.barrier()

    epoch = 0
    start_step = 0
    step = 0
    batch = {}
    size = [8, 3, 32, 192, 336]
    x1 = torch.rand(size, dtype=dtype, device=device)

    # prefetch one for non-blocking data loading
    def fetch_data():
        batch_ = next(dataloader_iter)
        batch_["video"] = batch_["video"].to(device, dtype, non_blocking=True)
        batch_["video"] = td.TensorDict({"video": batch_["video"]}).to(
            device=device, dtype=dtype, non_blocking_pin=True, num_threads=4
        )
        return batch_

    num_steps_per_epoch = 100000
    total_time = [] 
    for _ in range(start_step, num_steps_per_epoch):
        nsys.step()
        start_time = time()
        # == load data ===
        with nsys.range("load_video"), timers["load_data"]:
            # batch_ = fetch_data()
            if not dummy_data:
                pass
            else:
                step += 1
                batch["video"] = x1

        # == run iter ==
        with nsys.range("iter"), timers["iter"]:
            prepare_inputs(batch, dummy_data=dummy_data)
        dist.barrier()
        step_time = time() - start_time  
        total_time.append(step_time)  
        
        if step % 10 == 0:  
            avg_time = sum(total_time[-10:]) / len(total_time[-10:])  
            print(f"Step {step}/{num_steps_per_epoch}: Avg time per iter (last 10 steps): {avg_time:.4f}s")


        print(timers.to_str(0, step))


if __name__ == "__main__":
    main()
