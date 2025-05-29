import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main():
    parser = argparse.ArgumentParser(description="Finetune script for cropmark classification.")

    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model. If not provided, ImageNet weights will be used.")
    parser.add_argument("--dataset_path", type=str, default="../data/cropmark_dataset", help="Path to the root dataset folder containing 'train' and 'val' subfolders.")
    parser.add_argument("--save_dir", type=str, default="../models/", help="Directory to save trained models.")
    parser.add_argument("--saved_model_prefix", type=str, default="", help= "Prefix for saved model files.")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "step"], default=None, help="Type of learning rate scheduler to use (cosine or step). If not specified, no scheduler will be used.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 regularization) coefficient.")
    parser.add_argument("--save_best_F1", type=str, choices=["single", "all"], default="single", help="Saving mode for best F1 models: 'single' keeps one file, 'all' saves all best checkpoints.")
    parser.add_argument("--wandb_project", type=str, default=None, help="If set, logs training metrics to Weights & Biases with given project name.")

    # Augmentation probabilities
    parser.add_argument("--p_randomscale", type=float, default=0.3, help="Probability of RandomScale in training augmentations.")
    parser.add_argument("--p_rotate", type=float, default=0.2, help="Probability of small Rotate in training augmentations.")
    parser.add_argument("--p_center_crop", type=float, default=0.7, help="Probability of MaxCenterCropResize in training augmentations.")
    parser.add_argument("--p_90rotate", type=float, default=0.3, help="Probability of 90 degree rotation in training augmentations.")
    parser.add_argument("--p_hflip", type=float, default=0.5, help="Probability of horizontal flip.")
    parser.add_argument("--p_vflip", type=float, default=0.5, help="Probability of vertical flip.")
    parser.add_argument("--p_blur", type=float, default=0.25, help="Probability of Gaussian blur.")
    parser.add_argument("--p_distort", type=float, default=0.15, help="Probability of GridDistortion.")
    parser.add_argument("--p_brightness", type=float, default=0.65, help="Probability of RandomBrightnessContrast.")
    parser.add_argument("--p_hsv", type=float, default=0.2, help="Probability of HueSaturationValue.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224

    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.model_path) if args.model_path else "./models"
    os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    class MaxCenterCrop(A.ImageOnlyTransform):
        def __init__(self, always_apply=True, p=1.0):
            super(MaxCenterCrop, self).__init__(always_apply, p)

        def apply(self, img, **params):
            min_side = min(img.shape[:2])
            center = [s // 2 for s in img.shape[:2]]
            return img[center[0] - min_side // 2: center[0] + min_side // 2,
                   center[1] - min_side // 2: center[1] + min_side // 2]

    class MaxCenterCropResize(A.ImageOnlyTransform):
        def __init__(self, always_apply=False, p=1.0):
            super(MaxCenterCropResize, self).__init__(always_apply, p)

        def apply(self, img, **params):
            min_side = min(img.shape[:2])
            center = [s // 2 for s in img.shape[:2]]
            cropped = img[center[0] - min_side // 2: center[0] + min_side // 2,
                      center[1] - min_side // 2: center[1] + min_side // 2]
            resized = cv2.resize(cropped, (256, 256))
            return resized

    class ConditionalResize(A.ImageOnlyTransform):
        def __init__(self, target_size, always_apply=True, p=1.0):
            super(ConditionalResize, self).__init__(always_apply, p)
            self.target_size = target_size

        def apply(self, img, **params):
            h, w = img.shape[:2]
            min_side = min(h, w)
            if min_side < self.target_size:
                scale = self.target_size / min_side
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return img

    train_transform = A.Compose([
        A.RandomScale(scale_limit=(0.1, 0.2), p=args.p_randomscale),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=args.p_rotate),
        MaxCenterCropResize(always_apply=False, p=args.p_center_crop),
        A.OneOf([
            A.Rotate(limit=[90, 90]),
            A.Rotate(limit=[-90, -90])
        ], p=args.p_90rotate),
        A.HorizontalFlip(p=args.p_hflip),
        A.VerticalFlip(p=args.p_vflip),
        A.GaussianBlur(blur_limit=(2, 4), sigma_limit=0.5, p=args.p_blur),
        A.GridDistortion(num_steps=5, distort_limit=0.03, p=args.p_distort),
        A.RandomBrightnessContrast(0.1, 0.1, p=args.p_brightness),
        A.HueSaturationValue(10, 15, 10, p=args.p_hsv),
        ConditionalResize(target_size=img_size, always_apply=True, p=1.0),
        A.RandomCrop(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        MaxCenterCrop(),
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    class AlbumentationsDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.dataset = datasets.ImageFolder(root=root)
            self.transform = transform

        def __getitem__(self, index):
            path, label = self.dataset.samples[index]
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image=image)['image'].float()
            return image, label

        def __len__(self):
            return len(self.dataset)

    data_dirs = {
        'train': os.path.join(args.dataset_path, 'cropmark_train_set'),
        'val': os.path.join(args.dataset_path, 'cropmark_validation_set')
    }

    datasets_ = {
        'train': AlbumentationsDataset(data_dirs['train'], transform=train_transform),
        'val': AlbumentationsDataset(data_dirs['val'], transform=val_transform)
    }

    dataloaders = {
        split: DataLoader(datasets_[split], batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=0)
        for split in ['train', 'val']
    }

    model = models.resnet18(weights=None if args.model_path else models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.fc = nn.Linear(model.fc[0].in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    def evaluate(model, loader):
        model.eval()
        y_true, y_pred = [], []
        loss_total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs) > 0.5
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                loss_total += loss.item() * inputs.size(0)
        acc = (torch.tensor(y_true) == torch.tensor(y_pred).int()).float().mean().item()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return loss_total / len(loader.dataset), acc, prec, rec, f1

    best_f1 = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        y_true_train, y_pred_train = [], []
        loss_total = 0
        loop = tqdm(dataloaders['train'], desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = torch.sigmoid(outputs) > 0.5
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())
            loss_total += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())

        train_acc = (torch.tensor(y_true_train) == torch.tensor(y_pred_train).int()).float().mean().item()
        train_prec = precision_score(y_true_train, y_pred_train, zero_division=0)
        train_rec = recall_score(y_true_train, y_pred_train, zero_division=0)
        train_f1 = f1_score(y_true_train, y_pred_train, zero_division=0)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, dataloaders['val'])

        if args.wandb_project:
            import wandb
            wandb.log({
                "train_loss": loss_total / len(dataloaders['train'].dataset),
                "train_accuracy": train_acc,
                "train_precision": train_prec,
                "train_recall": train_rec,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        if val_f1 >= best_f1:
            best_f1 = val_f1
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            if args.save_best_F1 == "all":
                name = f"{args.saved_model_prefix}best_model_epoch{epoch + 1}_{timestamp}_f1{val_f1:.3f}_recall{val_rec:.3f}.pt"
            else:
                name = f"{args.saved_model_prefix}best_model_F1.pt"
                args.saved_model_prefix
            torch.save(model.state_dict(), os.path.join(args.save_dir, name))

        if scheduler:
            scheduler.step()

    print("\n✅ Finetuning complete.")
    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
