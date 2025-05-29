# src/solver/det_engine.py
import math
import sys
from typing import Dict, Iterable, List # Đảm bảo đã import

import numpy as np
import torch
# from torch.cuda.amp.grad_scaler import GradScaler # Đã có trong file gốc
# from torch.utils.tensorboard import SummaryWriter # Đã có trong file gốc

# from ..data import CocoEvaluator # Đã có trong file gốc
# from ..data.dataset import mscoco_category2label # Đã có trong file gốc
from ..misc import MetricLogger, SmoothedValue, dist_utils # Giả sử đường dẫn đúng
# from ..optim import ModelEMA, Warmup # Đã có trong file gốc
# from .validator import Validator, scale_boxes # Đã có trong file gốc


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    # Các tham số KD được truyền qua kd_loss_args
    teacher_model: torch.nn.Module = None,
    kd_loss_obj: torch.nn.Module = None,
    kd_loss_args: dict = None,
    **kwargs, # Các tham số khác như epochs, print_freq, ema, scaler, writer, v.v.
):
    model.train()
    criterion.train() # Đảm bảo criterion cũng ở train mode nếu nó có params

    # Lấy các tham số từ kwargs
    epochs = kwargs.get("epochs", 1) # Tổng số epoch
    print_freq = kwargs.get("print_freq", 10)
    writer = kwargs.get("writer", None)
    ema = kwargs.get("ema", None)
    scaler = kwargs.get("scaler", None) # Sẽ là None nếu không dùng AMP
    lr_warmup_scheduler = kwargs.get("lr_warmup_scheduler", None)
    max_norm = kwargs.get("max_norm", 0.1) # Max grad norm
    use_wandb = kwargs.get('use_wandb', False)
    output_dir = kwargs.get('output_dir', None) # Cho save_samples nếu có

    if use_wandb:
        try:
            import wandb
        except ImportError:
            wandb = None # Đặt là None nếu không import được
            if dist_utils.is_main_process():
                print("WandB not installed, disabling WandB logging for this epoch.")
            use_wandb = False
    else:
        wandb = None


    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # Thêm meter cho các thành phần loss
    metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}")) # Total loss
    metric_logger.add_meter("main_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
    if teacher_model and kd_loss_obj: # Chỉ thêm meter KD nếu KD được kích hoạt
        metric_logger.add_meter("kd_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

    header = "Epoch: [{}]".format(epoch) if epochs == 1 else "Epoch: [{}/{}]".format(epoch, epochs - 1)

    # Xử lý các tham số runtime của KD
    apply_kd_this_iteration = False
    current_kd_distill_decay = 1.0
    current_kd_final_ratio = 0.0

    if teacher_model and kd_loss_obj and kd_loss_args:
        kd_base_ratio = kd_loss_args.get('ratio', 1.0)
        kd_decay_type = kd_loss_args.get('decay_type', 'constant')
        kd_stop_epoch_ratio = kd_loss_args.get('stop_epoch_ratio', 1.0)
        
        stop_kd_at_epoch_num = int(epochs * kd_stop_epoch_ratio)

        if epoch < stop_kd_at_epoch_num:
            apply_kd_this_iteration = True # Sẽ tính decay sau cho từng iter
            
            total_iters_for_kd_phase = stop_kd_at_epoch_num * len(data_loader)
            # current_iter_in_kd_phase được tính trong vòng lặp
            
            if kd_decay_type == 'constant':
                current_kd_distill_decay = 1.0
            # Các kiểu decay khác sẽ được tính trong vòng lặp iteration
            # current_kd_final_ratio sẽ được tính bằng kd_base_ratio * current_kd_distill_decay

    epoch_losses = [] # Để log mean loss cuối epoch cho wandb

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas_for_criterion = dict(epoch=epoch, iter_in_epoch=i, global_step=global_step, epoch_total_iters=len(data_loader))

        # Optional: save samples for debugging
        # if i < kwargs.get("num_visualization_sample_batch", 1) and output_dir and dist_utils.is_main_process():
        #     from ..misc import save_samples # Giả sử hàm này tồn tại
        #     save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")


        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Tính distill_decay cho iteration hiện tại nếu KD được áp dụng
        iter_distill_decay_factor = 1.0
        if apply_kd_this_iteration: # Đã check epoch < stop_kd_at_epoch_num
            total_iters_for_kd_phase = int(epochs * kd_stop_epoch_ratio * len(data_loader))
            current_iter_in_kd_phase = global_step # Giả sử KD bắt đầu từ iter 0 của epoch 0

            kd_decay_type_runtime = kd_loss_args.get('decay_type', 'constant')
            if kd_decay_type_runtime == 'linear_epoch':
                if total_iters_for_kd_phase > 0:
                    iter_distill_decay_factor = 1.0 - (current_iter_in_kd_phase / float(total_iters_for_kd_phase))
                else: iter_distill_decay_factor = 0.0
            elif kd_decay_type_runtime == 'cosine_epoch':
                if total_iters_for_kd_phase > 0:
                    iter_distill_decay_factor = 0.5 * (1 + math.cos(math.pi * current_iter_in_kd_phase / float(total_iters_for_kd_phase)))
                else: iter_distill_decay_factor = 0.0
            elif kd_decay_type_runtime == 'constant':
                 iter_distill_decay_factor = 1.0
            # Thêm các kiểu decay khác nếu cần
            iter_distill_decay_factor = max(0.0, iter_distill_decay_factor)
            current_kd_final_ratio = kd_loss_args.get('ratio', 1.0) * iter_distill_decay_factor
        else: # Không áp dụng KD cho epoch này
            current_kd_final_ratio = 0.0


        optimizer.zero_grad(set_to_none=True) # set_to_none=True có thể nhanh hơn
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            student_outputs_dict = model(samples, targets=targets) # Student forward, targets cho DN

            kd_loss_iter_value = torch.tensor(0.0, device=device)
            if apply_kd_this_iteration and current_kd_final_ratio > 0 and kd_loss_obj is not None:
                with torch.no_grad():
                    teacher_outputs_dict = teacher_model(samples) # Teacher forward, chỉ cần samples

                # student_outputs_dict['pred_logits'] và ['pred_boxes'] là output lớp cuối của student
                # teacher_outputs_dict['pred_logits'] và ['pred_boxes'] là output lớp eval_idx của teacher
                kd_loss_iter_value = kd_loss_obj(
                    student_outputs_dict, # Truyền toàn bộ dict cho DFINELogicLoss
                    teacher_outputs_dict,
                    targets
                )
                kd_loss_iter_value = kd_loss_iter_value * current_kd_final_ratio # Áp dụng ratio đã có decay
            
            main_loss_dict = criterion(student_outputs_dict, targets, **metas_for_criterion)
            main_loss_iter = sum(main_loss_dict.values())
            
            total_loss_iter = main_loss_iter + kd_loss_iter_value

        if scaler is not None:
            scaler.scale(total_loss_iter).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss_iter.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        # Logging
        reduced_main_loss_dict = dist_utils.reduce_dict(main_loss_dict)
        # Reduce kd_loss_iter_value (scalar)
        if dist_utils.is_dist_available_and_initialized():
            dist_utils.all_reduce(kd_loss_iter_value) # all_reduce cho scalar
            kd_loss_iter_reduced = kd_loss_iter_value / dist_utils.get_world_size()
        else:
            kd_loss_iter_reduced = kd_loss_iter_value
        
        # total_loss_iter_reduced = main_loss_iter_reduced + kd_loss_iter_reduced (cần reduce main_loss_iter trước)
        main_loss_iter_reduced_scalar = sum(reduced_main_loss_dict.values()) # Scalar
        total_loss_iter_reduced_scalar = main_loss_iter_reduced_scalar + kd_loss_iter_reduced


        metric_logger.update(loss=total_loss_iter_reduced_scalar.item(),
                             main_loss=main_loss_iter_reduced_scalar.item())
        if teacher_model and kd_loss_obj:
            metric_logger.update(kd_loss=kd_loss_iter_reduced.item())
        
        for k_reduced, v_reduced in reduced_main_loss_dict.items():
             metric_logger.update(**{k_reduced: v_reduced.item()}) # Unpack dict vào update
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        epoch_losses.append(total_loss_iter_reduced_scalar.item())

        if writer and dist_utils.is_main_process() and global_step % print_freq == 0: # Log thường xuyên hơn
            writer.add_scalar("TrainIter/TotalLoss", total_loss_iter_reduced_scalar.item(), global_step)
            writer.add_scalar("TrainIter/MainLoss", main_loss_iter_reduced_scalar.item(), global_step)
            if teacher_model and kd_loss_obj:
                writer.add_scalar("TrainIter/KDLoss", kd_loss_iter_reduced.item(), global_step)
                writer.add_scalar("TrainIter/KDFinalRatio", current_kd_final_ratio, global_step)
            for k_log, v_log in reduced_main_loss_dict.items():
                writer.add_scalar(f"TrainIter/MainLoss_{k_log}", v_log.item(), global_step)
            writer.add_scalar("Lr/Current", optimizer.param_groups[0]["lr"], global_step)

    # Kết thúc epoch
    metric_logger.synchronize_between_processes() # Đồng bộ hóa các meter
    print("Averaged stats for epoch:", metric_logger)
    
    avg_epoch_total_loss = np.mean(epoch_losses) if epoch_losses else 0.0 # Tính trung bình loss trong epoch
    
    if use_wandb and wandb is not None and dist_utils.is_main_process():
        wandb_log_data = {
            "train_epoch/avg_total_loss": avg_epoch_total_loss,
            "train_epoch/avg_main_loss": metric_logger.meters['main_loss'].global_avg,
            "lr": optimizer.param_groups[0]["lr"], # Lr cuối epoch
            "epoch": epoch
        }
        if teacher_model and kd_loss_obj:
             wandb_log_data["train_epoch/avg_kd_loss"] = metric_logger.meters['kd_loss'].global_avg
        
        for k_meter, meter_obj in metric_logger.meters.items():
            # Chỉ log các loss chi tiết của main_loss, tránh ghi đè các key đã có
            if k_meter not in ['loss', 'main_loss', 'kd_loss', 'lr']:
                 wandb_log_data[f"train_epoch/avg_main_{k_meter}"] = meter_obj.global_avg
        wandb.log(wandb_log_data)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}