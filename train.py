import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import FullDataset, TestDataset
from eval import (
    MIOU,
    evaluate_dataset,
    evaluate_segmentation_performance,
    print_eval_report,
)
from SAM3UNet import SAM3UNet


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def main(args):
    # 1. Load train data
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 672, mode="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # 2. Load test data
    test_loader = TestDataset(args.test_image_path, args.test_gt_path, 672)
    # 3. Set device
    device = torch.device("cuda")
    # 4. Load model to device
    model = SAM3UNet(args.sam3_path, 672)
    model.to(device)
    # 5. Set optimizer
    optim = opt.AdamW(
        [{"params": model.parameters(), "initia_lr": args.lr}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # 6. Set scheduler
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    # 7. Train
    os.makedirs(args.save_path, exist_ok=True)
    epoch_loss = 2.0
    base_mean_iou = args.base_mean_iou
    auto_save_iou = args.auto_save_iou
    save_interval = args.save_interval
    for epoch in range(args.epoch):
        # 7.1. Train phase
        print("Training:")
        model.train()  # Set model to training mode
        for i, batch in enumerate(dataloader):
            x = batch["image"]
            target = batch["label"]
            x = x.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = structure_loss(pred, target)
            epoch_loss = loss.item()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                print("epoch-{}-{}: loss:{}".format(epoch + 1, i + 1, epoch_loss))
        scheduler.step()

        # 7.2. Evaluation phase
        print("Evaluating", end="")
        # Set model to evaluation mode
        model.eval()
        # Disable gradient calculations for efficiency and safety
        results = []
        for i in range(test_loader.size):
            with torch.no_grad():
                # 7.2.1. Load test image, gt, name and padding
                image, gt, name, padding = test_loader.load_data()
                image = image.to(device)
                # 7.2.2. Model predict, Save process time
                res_padded = model(image)
                # Remove padding
                pad_left, pad_top, pad_right, pad_bottom = padding
                res = res_padded[:, :, pad_top : 672 - pad_bottom, pad_left : 672 - pad_right]
                # Output conversion
                res = F.interpolate(res, size=gt.shape, mode="bilinear", align_corners=False)
                res = res.sigmoid().data.cpu()
                res = res.numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = (res * 255).astype(np.uint8)
                gt = np.asarray(gt, np.float32)
                # Evaluate
                result = evaluate_segmentation_performance(res, gt)
                # 2.4. Save result
                results.append(result)
                # Print for status
                if i % 10 == 0:
                    print(".", end="", flush=True)

        # Reset test_loader index
        test_loader.reset_index()
        final_result = evaluate_dataset(results)
        print_eval_report(final_result, title=f"epoch-{epoch + 1}: loss: {epoch_loss}")

        # 7.3. Save checkpoint
        mean_iou = final_result[MIOU]

        if mean_iou > base_mean_iou:
            base_mean_iou = mean_iou
            save_model_path = os.path.join(
                args.save_path,
                f"SAM3-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth",
            )
            torch.save(model.state_dict(), save_model_path)
            print("Saving Snapshot best:", save_model_path)
        elif mean_iou > auto_save_iou:
            save_model_path = os.path.join(
                args.save_path,
                f"SAM3-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth",
            )
            torch.save(model.state_dict(), save_model_path)
            print("Auto Saving Good Snapshot:", save_model_path)
        elif (epoch + 1) % save_interval == 0 or (epoch + 1) == args.epoch:
            save_model_path = os.path.join(
                args.save_path,
                f"SAM3-UNet_epoch-{epoch + 1}_loss-{epoch_loss:.3f}_iou-{mean_iou:.3f}.pth",
            )
            torch.save(model.state_dict(), save_model_path)
            print("Saving Snapshot:", save_model_path)


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM3-UNet")
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to store the checkpoint"
    )
    parser.add_argument(
        "--sam3_path",
        type=str,
        default="/data1/workspace/ai_shared_workspace/model_zoo_shared/sam3/sam3.pt",
        help="path to the sam3 pretrained pth",
    )
    parser.add_argument(
        "--train_image_path",
        type=str,
        default="/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_train/images/",
        help="path to the image that used to train the model",
    )
    parser.add_argument(
        "--train_mask_path",
        type=str,
        default="/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_train/masks/",
        help="path to the mask file for training",
    )
    parser.add_argument(
        "--test_image_path",
        type=str,
        default="/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/images/",
        help="path to the image that used to evaluate the model",
    )
    parser.add_argument(
        "--test_gt_path",
        type=str,
        default="/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/masks/",
        help="path to the mask file for evaluating",
    )
    parser.add_argument("--epoch", type=int, default=500, help="training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--batch_size", default=36, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--base_mean_iou", default=0.8, type=float)
    parser.add_argument("--auto_save_iou", default=0.85, type=float)
    args = parser.parse_args()
    # seed_torch(1024)
    main(args)
