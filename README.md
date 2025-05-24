# Detection (scene classification) of archaeological sites (cropmarks) using remote sensing (aerial imagery)

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
The above code will reproduce KILE results, to reproduce LIR results, `epochs` should be changed to `50` and `seed` to `1`

### Predict
```bash
python predict.py \
  --model_base_path ../models/ \
  --model_name model_7.pt \
  --image_folder ./data/cropmark_dataset/cropmark_test_set/positive \
  --output_csv_path ./results/predictions_model7.csv \
  --plot_random_preds \
  --use_gpu

```
