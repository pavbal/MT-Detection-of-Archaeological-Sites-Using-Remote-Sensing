#  Detection (scene classification) of archaeological sites (cropmarks) using remote sensing (aerial imagery)
#  WORK IN PROGRESS - this repository is not yet finished and is being updated
## Repository introduction
This repository contains the representative code for the Master's Thesis "Detection of archaeological sites using remote sensing"
by Pavel Balda.

The repository contains the representative version of thesis' code, models, figures, and data. Detailed info is provided in [the thesis PDF](Pavel%20Balda%20-%20Master's%20Thesis.pdf).

Only a part of the data and code is included in the repository. The full dataset and code are available on request from the author.

## Environvment setup

To run scripts, please install requirements as follows:
```bash
pip install -r requirements.txt
```

## Training and prediction


### Test script
To reproduce the results of the paper, please run the following command (for models `model_2` - `model_13`). Argument `plot_FN_FP` 
will plot the example subset of false negatives and false positives of the model on the test set. Argument `use_gpu` will use GPU for training if available. 
```bash
python src/test.py \
        --model_path ./models/ \
        --model_name model_7 \
        --dataset_path ./data/cropmark_dataset/cropmark_test_set \
        --plot_FN_FP True \
        --use_gpu

```

### Predict
The following command will predict the images from given `image_folder` using the given model (`model_7.pt`) and save the 
predictions to `predictions_model7.csv` (in a form of output probabilities).
```bash
python predict.py \
  --model_base_path ../models/ \
  --model_name model_7.pt \
  --image_folder ./data/cropmark_dataset/cropmark_test_set/positive \
  --output_csv_path ./results/predictions.csv \
  --plot_random_preds \
  --use_gpu

```
### Train

### Neural data generation
Neural data generation (via SDXL 1.0 LoRA finetuning) is not implemented in the repository. The modification of the [Colab Notebook by jhj0517](https://colab.research.google.com/github/jhj0517/finetuning-notebooks/blob/master/sdxl/finetuning_notebooks_sdxl_lora_dreambooth.ipynb)
was used to generate all the neural images.