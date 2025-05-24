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
    parser = argparse.ArgumentParser(description="Predict script.")

    parser.add_argument("--model_base_path", type=str, default="../models/", help="Path to model directory.")
    parser.add_argument("--model_name", type=str, default="model_7.pt", help="Name of the file with the model weights.")
    parser.add_argument("--image_folder", type=str, default="../data/cropmark_dataset/cropmark_test_set", help="Folder with input images.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for testing.")
    parser.add_argument("--output_csv_path", type=str, default="predictions.csv", help="Path to save predictions CSV.")
    parser.add_argument("--plot_random_preds", action="store_true", help="Plot a few random predictions.")

    args = parser.parse_args()

    model_path = args.model_base_path + args.model_name
    if not os.path.exists(model_path):
        model_path = model_path[1:]

    image_folder = args.image_folder
    if not os.path.exists(image_folder):
        image_folder = image_folder[1:]

    print(f"Model path: {model_path}")
    print(f"Image folder: {args.image_folder}")
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
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, folder, transform=None):
            self.image_paths = [os.path.join(folder, fname)
                                for fname in os.listdir(folder)
                                if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.transform = transform

        def __getitem__(self, index):
            path = self.image_paths[index]
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image=image)['image'].float()
            return image, path

        def __len__(self):
            return len(self.image_paths)

    # ======= LOADERS =======
    test_dataset = InferenceDataset(image_folder, transform=test_transform)
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


    results = []
    with torch.no_grad():
        for inputs, batch_paths in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(1)

            if filename in filenames_1 or filename in filenames_3:
                probs = torch.sigmoid(outputs).cpu().numpy()
            else:
                probs = outputs.cpu().numpy()

            for path, prob in zip(batch_paths, probs):
                results.append((path, float(prob)))

    import pandas as pd
    df = pd.DataFrame(results, columns=["image_path", "predicted_probability"])
    df.to_csv(args.output_csv_path, index=False)
    print(f"Saved predictions to {args.output_csv_path}")

    if args.plot_random_preds:
        sample = random.sample(results, min(6, len(results)))

        fig, axs = plt.subplots(1, len(sample), figsize=(15, 4))
        fig.suptitle("Random Examples of Predictions", fontsize=16)

        for i, (path, prob) in enumerate(sample):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            min_side = min(h, w)
            startx = w // 2 - min_side // 2
            starty = h // 2 - min_side // 2
            img = img[starty:starty + min_side, startx:startx + min_side]
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(f"{prob:.2f}")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()