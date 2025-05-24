import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Testovací skript.")

    parser.add_argument("--model_base_path", type=str, default="../models/", help="Path to model directory.")
    parser.add_argument("--model_name", type=str, default="model_7.pt", help="Name of the file with the model weights.")
    parser.add_argument("--dataset_path", type=str, default="../data/cropmark_dataset/cropmark_test_set", help="Dataset path.")
    parser.add_argument("--plot_FN_FP", type=bool, default=False, help="Plot figure with examples of False Negatives and False Positives.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for testing.")

    args = parser.parse_args()

    model_path = args.model_base_path + args.model_name
    if not os.path.exists(model_path):
        model_path = model_path[1:]

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        dataset_path = dataset_path[1:]

    print(f"Model path: {model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Plot FN and FP: {args.plot_FN_FP}")
    print(f"Device preferred: {'cuda' if args.use_gpu else 'cpu'}")

    # ======= SEED =======
    SEED = 77
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    plot_fn_fp = args.plot_FN_FP



    model_name = "resnet18"
    # dataset_path = "../data/cropmark_dataset/cropmark_test_set"


    img_size = 224
    batch_size = 32

    filename = os.path.basename(model_path)  # "model_5.pt"
    filenames = {"model_2.pt", "model_3.pt", "model_4.pt", "model_5.pt", "model_6.pt", "model_7.pt", "model_8.pt", "model_9.pt", "model_10.pt", "model_11.pt", "model_12.pt", "model_13.pt", "best_accuracy_model_aerial.pth", "best_accuracy_model_field.pth"}
    filenames_1 = {"model_2.pt", "model_3.pt", "model_4.pt"}
    # filenames_2 = {'model_5.pt', "model_6.pt"}
    filenames_3 = {"model_7.pt", "model_8.pt", "model_9.pt"}
    filenames_4 = {"model_10.pt", "model_11.pt", "model_12.pt", "model_13.pt"}
    filenames_5 = {"best_accuracy_model_aerial.pth", "best_accuracy_model_field.pth"}

    if filename in filenames_5:
        raise ValueError("Model is not supported for this test. It should be evaluated on aerial test set or agricultural test set INSTEAD OF cropmark test set. Please select another model.")

    if filename in filenames:
        custom_model = False
    else:
        custom_model = True

    class MaxCenterCrop(A.ImageOnlyTransform):
        def __init__(self, always_apply=True, p=1.0):
            super(MaxCenterCrop, self).__init__(always_apply, p)

        def apply(self, img, **params):
            min_side = min(img.shape[:2])
            center = [s // 2 for s in img.shape[:2]]
            return img[center[0] - min_side // 2 : center[0] + min_side // 2,
                       center[1] - min_side // 2 : center[1] + min_side // 2]


    if filename in filenames_4 or custom_model:
        test_transform = Compose([
            MaxCenterCrop(always_apply=True, p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        test_transform = Compose([
            MaxCenterCrop(always_apply=True, p=1.0),
            A.Resize(img_size, img_size),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
    ])

    # ======= DATASET =======
    class AlbumentationsDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.dataset = datasets.ImageFolder(root=root)
            self.transform = transform

        def __getitem__(self, index):
            path, label = self.dataset.samples[index]
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image=image)['image'].float()
            return image, label, path

        def __len__(self):
            return len(self.dataset)

    # ======= LOADERS =======
    test_dataset = AlbumentationsDataset(dataset_path, transform=test_transform)



    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ======= MODEL LOADING =======
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")


    try:
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
        else:
            raise ValueError("Unsupported model")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except RuntimeError:
        if model_name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError("Unsupported model")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Testing model ({model_name}) on cropmark test set...")
    model = model.to(device)
    model.eval()

    # ======= TEST =======
    y_true, y_pred, paths = [], [], []

    # ======= TEST =======
    with torch.no_grad():
        for inputs, labels, batch_paths in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze(1)
            if filename in filenames_1 or filename in filenames_3:
                probs = torch.sigmoid(outputs)
                if filename in filenames_1:
                    probs = probs > 0.5
                preds = probs > 0.5  # logity převedené na pravděpodobnost
            else:
                preds = outputs > 0.5  # výstupy už jsou sigmoidované
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            paths.extend(batch_paths)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]

    sample_fp = random.sample(list(false_positives), min(6, len(false_positives)))
    sample_fn = random.sample(list(false_negatives), min(6, len(false_negatives)))

        # ======= MAX CENTRAL CROP FUNKCE =======
    if plot_fn_fp:
        def max_center_crop(img):
            h, w, _ = img.shape
            min_side = min(h, w)
            startx = w//2 - min_side//2
            starty = h//2 - min_side//2
            return img[starty:starty+min_side, startx:startx+min_side]

        # ======= ZOBRAZENÍ =======
        fig, axs = plt.subplots(2, 6, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.02, hspace=0.02)
        fig.suptitle('Examples of Type I (False Positive) and Type II (False Negative) Errors', fontsize=18, y=0.88)

        for i, idx in enumerate(sample_fp):
            img = cv2.cvtColor(cv2.imread(paths[idx]), cv2.COLOR_BGR2RGB)
            img = max_center_crop(img)
            axs[0, i].imshow(img)
            axs[0, i].axis('off')

        for i, idx in enumerate(sample_fn):
            img = cv2.cvtColor(cv2.imread(paths[idx]), cv2.COLOR_BGR2RGB)
            img = max_center_crop(img)
            axs[1, i].imshow(img)
            axs[1, i].axis('off')

        fig.text(0.005, 0.63, 'FP', va='center', ha='left', fontsize=14, weight='bold')
        fig.text(0.005, 0.25, 'FN', va='center', ha='left', fontsize=14, weight='bold')

        plt.tight_layout(rect=[0.03, 0, 1, 0.95])
        plt.show()

    # ======= VÝPIS METRIK A KONFUZNÍ MATICE =======
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['negative', 'positive'], digits=3))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)


if __name__ == "__main__":
    main()