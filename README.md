# Water Segmentation from Multispectral Satellite Imagery with a Channel-Attention U-Net

A PyTorch pipeline that segments surface water in 128 x 128 satellite tiles. It combines remote-sensing feature engineering (spectral indices, terrain and land-cover features) with a U-Net that includes a learned channel-attention gate, and it analyses which inputs the model actually relies on.

## Results

Evaluated on a held-out test set of 47 images (global pixel-level metrics):

| Split | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| Validation | 0.7646 | 0.8413 | 0.8934 | 0.8666 |
| Test | 0.8004 | 0.8684 | 0.9109 | 0.8891 |
| Test + TTA | 0.7990 | 0.8670 | 0.9106 | 0.8883 |

Additional findings:

- With all ancillary channels (DEM, slope, relief, water occurrence, WorldCover) set to zero, test IoU is 0.7841, so the model relies mainly on spectral information.
- A three-model ensemble with a validation-tuned threshold (0.55) reached a test IoU of 0.7996, with no gain over the single model.
- Test-time augmentation (flips) did not improve results.
- Recall depends strongly on water body size: about 0.90 for regions larger than 1000 pixels, but only about 0.34 for regions of 10 pixels or fewer.

## Dataset

- 306 tiles of shape 128 x 128 x 12, stored as GeoTIFF, with binary water masks stored as PNG (1 = water, 0 = background).
- Channel order: Coastal aerosol, Blue, Green, Red, NIR, SWIR1, SWIR2, QA band, MERIT DEM, Copernicus DEM, ESA WorldCover, Water occurrence probability.
- Class balance: 25.98% water pixels; 14.71% of images contain no water.
- Split: random 70 / 15 / 15, giving 214 training, 45 validation and 47 test images.

Expected layout:

```
data/
  images/   *.tif
  labels/   *.png   (same filename as the image)
```

Set `DATA_DIR` at the top of the notebook to your data location. The default points to a Kaggle input path.

## Method

### 1. Channel selection and feature engineering

Each channel was reviewed using remote-sensing knowledge:

| Channel | Decision |
|---|---|
| Coastal aerosol | Dropped (atmospheric correction band, correlated with Blue) |
| QA band | Dropped (bit flags, not a physical measurement) |
| MERIT DEM | Dropped (nearly identical to Copernicus DEM, contains a -9999 no-data value) |
| ESA WorldCover (raw code) | Replaced by one-hot water and wetland channels |
| Blue, Green, Red, NIR, SWIR1, SWIR2 | Kept |
| Copernicus DEM, Water occurrence | Kept |

Features added:

- Spectral indices: NDWI, MNDWI, NDVI, AWEI.
- Slope: gradient magnitude of the Copernicus DEM, in degrees.
- Relief: DEM minus its local 15 x 15 mean (negative in depressions).
- WC water and WC wetland: binary channels from WorldCover classes 80 and 90/95.

### 2. Normalisation

Each channel is clipped to its 1st to 99th percentile and standardised using statistics computed on the training split only. Binary channels are left as 0/1. Statistics are saved to `normalization_stats.csv`.

### 3. Model

- U-Net with four encoder and four decoder stages, skip connections, BatchNorm, and a 512-channel bottleneck (about 7.8 M parameters).
- A squeeze-and-excitation channel-attention gate on the input, which learns a weight in (0, 1) for every input channel and image.

### 4. Training

- Loss: binary cross-entropy plus Tversky loss (alpha = 0.3 for false positives, beta = 0.7 for misses), which favours recall.
- Optimiser: AdamW (learning rate 1e-3, weight decay 1e-4) with one epoch of warmup and cosine decay.
- Exponential moving average (EMA) of the weights, used for validation and checkpointing.
- Augmentation: random horizontal and vertical flips.
- Ancillary-channel dropout: DEM, slope, relief, water occurrence and WorldCover channels are zeroed for 30% of training images to reduce over-reliance on static priors.
- Checkpoint selection by best validation IoU.

### 5. Ablation study

Six input sets were trained for 30 epochs each:

| Feature set | Channels | Validation IoU |
|---|---|---|
| plus_terrain_landcover | 16 | 0.7450 |
| bands_indices | 10 | 0.7293 |
| plus_terrain | 14 | 0.7236 |
| bands_indices_dem | 11 | 0.7094 |
| all_12 | 12 | 0.7046 |
| bands_indices_dem_occurrence | 12 | 0.6990 |

On the best set, removing channel attention lowered validation IoU from 0.7450 to 0.7034. The best set was then retrained for 100 epochs.

## Analysis

- Channel attention weights: mean gate value per channel on the test set (`attention_weights.csv`).
- Permutation importance: IoU drop when each channel is shuffled. NDWI, Green and NIR matter most; wetland has almost no effect (`permutation_importance.csv`).
- Per land-cover class IoU, recall and false positive rate (`iou_per_landcover.csv`).
- Error maps for the six worst test images (false positives, missed water, correct water).
- Recall by connected water-body size (`recall_by_size.csv`).

## Limitations

- The dataset is small (306 tiles). Each ablation result comes from a single seed and a 45-image validation set, so small differences may be noise.
- The split is random. If tiles overlap or are spatially adjacent, there may be mild spatial leakage.
- The pixel size used for slope (30 m) is an assumption and should be set to the true resolution.
- Small water bodies (about 50 pixels or fewer) are the main failure mode.
- The 0.7646 validation IoU is slightly optimistic because the validation set was used for checkpoint selection. The test numbers are the unbiased estimate.

## Possible improvements

- Multi-seed runs or cross-validation to quantify variance.
- A spatially aware split.
- Loss or sampling changes that emphasise small objects.
- Architectures with denser skip connections (for example U-Net++) or higher-resolution inputs.

## Requirements

- Python 3.9+
- torch
- numpy, pandas, scipy, scikit-learn
- matplotlib, tifffile, Pillow, tqdm

```
pip install torch numpy pandas scipy scikit-learn matplotlib tifffile pillow tqdm
```

A CUDA-capable GPU is recommended. The notebook was run on Kaggle.

## Usage

1. Place the dataset in the layout above and update `DATA_DIR`.
2. Open the notebook and run all cells in order.
3. Outputs are written to the working directory:
   - `unet_water.pt` (final model weights)
   - `normalization_stats.csv`
   - `final_metrics.csv`
   - `attention_weights.csv`
   - `permutation_importance.csv`
   - `iou_per_landcover.csv`
   - `recall_by_size.csv`

Hyperparameters (batch size, learning rate, epochs, EMA decay, seed) are set in the first code cell.
