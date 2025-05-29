#  Detection (scene classification) of archaeological sites (cropmarks) using remote sensing (aerial imagery)
#  WORK IN PROGRESS - this repository is not yet finished and is being updated
## Repository introduction
This repository serves as a supplement to the Master's Thesis "Detection of archaeological sites using remote sensing"
by Pavel Balda.

Repository contains the representative version of thesis' code, models, figures, and data. Detailed info is provided in [the thesis PDF](Pavel%20Balda%20-%20Master's%20Thesis.pdf).

Only a part of the data and code is included in the repository. The full dataset and code are available on request from the author.

## Environvment setup

To run scripts, please install requirements as follows:
```bash
pip install -r requirements.txt
```

## Prediction and training


### Test on cropmark test set
To reproduce the results presented in the paper, please run the following command (for models `model_2` - `model_13`). Argument `plot_FN_FP` 
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
The following command will predict the images from `image_folder` using the given model (`model_7.pt`) and save the 
predictions to `./results/predictions.csv` (in a form of output probabilities).
```bash
python src/predict.py \
  --model_base_path ../models/ \
  --model_name model_7.pt \
  --image_folder ./data/cropmark_dataset/cropmark_test_set/positive \
  --output_csv_path ./results/predictions.csv \
  --plot_random_preds \
  --use_gpu

```
### Train
To train the model on the cropmark dataset, please run the following command. The model will be saved to `./models/finetuned` directory. 
For more info about the arguments, please run the script with `--help` argument. 
The training will use WandB for logging and visualization of the training process. 
The model will be trained on the cropmark dataset. The pretrained model is expected to be in the `model_path` directory.
You can use sample model pretrained on procedrual dataset [here](./models/model_pretrained_procedural.pt).

```bash
python src/train.py \
  --dataset_path ../data/cropmark_dataset \
  --model_path ./models/pretrained.pt \
  --save_dir ./models/finetuned \
  --learning_rate 1e-5 \
  --num_epochs 15 \
  --weight_decay 5e-5 \
  --wandb_project cropmark_classification \
  --save_best_F1 all \
  --scheduler cosine \
  --p_randomscale 0.3 \
  --p_rotate 0.2 \
  --p_center_crop 0.7 \
  --p_90rotate 0.3 \
  --p_hflip 0.5 \
  --p_vflip 0.5 \
  --p_blur 0.25 \
  --p_distort 0.15 \
  --p_brightness 0.65 \
  --p_hsv 0.2


```

## Synthetic data generation

### Procedural data genration
To generate the procedural masks and apply them to real geoportal imagery, run the following command. For more info
about the arguments, please run the script with `--help` argument.
```bash
python src/mask_generation/create_masked_images.py \
  --input_folder data/geoportal_mock_dataset/negative \
  --output_folder data/geoportal_mock_dataset/positive_procedural \
  --positive_mask_prob 0.85 \
  --multi_type_probs 0.8 0.15 0.05 \
  --repeat_single_mask_prob 0.1 \
  --mask_type_weights 0.2 0.2 0.2 0.2 0.2 \
  --alpha_random_min 0.1 \
  --alpha_random_max 0.25 \
  --save_masks \
  --plot_mask_stats \
  --plot_examples \
  --plot_mask_examples \
  --seed 43
```

### Neural data generation
Neural data generation (via SDXL 1.0 LoRA finetuning) is not implemented in the repository. The modification of the [Colab Notebook by jhj0517](https://colab.research.google.com/github/jhj0517/finetuning-notebooks/blob/master/sdxl/finetuning_notebooks_sdxl_lora_dreambooth.ipynb)
was used to generate all the neural images. The examples of the generated images can be found [here](figures/image_tile_visualizations), specifically
[negative examples](figures/image_tile_visualizations/neural_synth_cropmark_examples_negatives.pdf) and [positive examples](figures/image_tile_visualizations/neural_synth_cropmark_examples_types.pdf) figures.