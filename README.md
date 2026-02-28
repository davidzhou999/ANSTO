20260219 update:

Uploaded the trained machine learning model file (n2same_het_pretrained.pt) to Google Drive because it is over 500MB in size, and provided the download link:  https://drive.google.com/file/d/1lCSJam-xlGZqkF9WZYNYaz6nodVxDI_g/view?usp=drive_link

n2same_het_pretrained.pt was obtained by training on about 4000 original neutron images.

Original neutron image vs denoised image: 

<img width="1080" height="1018" alt="image" src="https://github.com/user-attachments/assets/2450ae9c-1fa5-4883-82ab-a5f32ab54bfd" />

Three screenshots of 3D reconstruction image:
<img width="3833" height="2149" alt="Screenshot from 2026-02-19 15-57-06" src="https://github.com/user-attachments/assets/6c2b1802-b874-4a42-8c5a-a530730f63fe" />

<img width="3835" height="2151" alt="Screenshot from 2026-02-20 16-27-24" src="https://github.com/user-attachments/assets/65886f2c-ba2d-46bb-9b7b-ee7d6fe2f989" />

<img width="3837" height="2159" alt="Screenshot from 2026-02-21 10-53-53" src="https://github.com/user-attachments/assets/2e6f450c-10d4-4739-95e7-09c3ea61f4aa" />



202511:

From three of the most effective despeckle methods used in deep learning pipelines or image post-processing, the Outlier-Based Masking + Morphological Cleanup approach has been implemented compared in terms of removal power, preservation of structure, and suitability for speckle-like noise (e.g., bright/dark dots in microscopy or satellite imagery), which are better than Non-Local Means (Post-Inference Only), and “Remove Outliers” Filter (e.g. ImageJ / Fiji Inspired):

python train_n2same_het_multi_progress_metrics_stable.py \
  --roots data/train_data \
  --epochs 50 \
  --iters_per_epoch 2000 \
  --eval_iters 50 \
  --batch 12 \
  --patch 320 \
  --workers 8 \
  --amp \
  --save_path data/n2same_het_speckle_trained.pt \
  --speckle_weight 0.15 \
  --charb_w 0.3 \
  --hi_pctl 99.98 --lo_pctl 0.02 \
  --z_tau_hi 4.0 --z_tau_lo 4.0 \
  --speckle_dilate 4 \
  --ring_w 0.01 --ring_sigma 7 --ring_axis x \
  --lr 1e-4 --min_lr 1e-6 \
  --sched cosine --warmup_steps 500
  
python predict_tiff-despeckle.py \
  --ckpt data/n2same_het_pretrained.pt \
  --inp data/noisy \
  --out_dir data/preds_desp_full \
  --tile 1600 \
  --overlap 192 \
  --autocast \
  --norm percentile \
  --norm_pctl 0.5 99.5 \
  --global_norm \
  --global_take 64 \
  --brightness_match \
  --despeckle \
  --desp_source denoised \
  --desp_tau 3.0 \
  --hi_pctl 99.95 \
  --lo_pctl 0.05 \
  --z_tau_hi 6.0 \
  --z_tau_lo 6.0 \
  --desp_max_blob 600 \
  --desp_dilate 1 \
  --suffix _den \
  --out_dtype uint16 \
  --print_stats \
  --save_debug

python predict_tiff-despeckle.py \
  --ckpt data/n2same_het_pretrained.pt \
  --inp data/noisy \
  --out_dir data/preds_desp_aggressive \
  --tile 1600 \
  --overlap 192 \
  --autocast \
  --norm percentile \
  --norm_pctl 0.5 99.5 \
  --global_norm \
  --global_take 64 \
  --brightness_match \
  --despeckle \
  --desp_source denoised \
  --desp_tau 1.5 \                     # Lower MAD threshold → more masking
  --hi_pctl 99.9 \                     # High cutoff includes more bright outliers
  --lo_pctl 0.1 \                      # Low cutoff includes more dark outliers
  --z_tau_hi 3.0 \                     # Lower Z-score → more aggressive detection
  --z_tau_lo 3.0 \                     # Same for dark
  --desp_max_blob 200 \               # Limit blob size to remove small spots
  --desp_dilate 3 \                   # Expand mask for broader cleanup
  --desp_ksize 11 \                   # Larger median filter area for inpainting
  --suffix _den_aggr \
  --out_dtype uint16 \
  --print_stats \
  --save_debug

python predict_tiff-despeckle.py \
  --ckpt data/n2same_het_pretrained.pt \
  --inp data/noisy \
  --out_dir data/preds_desp_ultra \
  --tile 1600 \
  --overlap 192 \
  --autocast \
  --norm percentile \
  --norm_pctl 0.5 99.5 \
  --global_norm \
  --global_take 64 \
  --brightness_match \
  --despeckle \
  --desp_source denoised \
  --desp_tau 1.0 \                      # Very sensitive to deviations  -Detect even mild residual intensity anomalies
  --hi_pctl 99.85 \                     # Even more inclusive of bright pixels -Catch broader pixel intensity extremes..
  --lo_pctl 0.15 \                      # Broader inclusion of dark spots
  --z_tau_hi 2.0 \                      # Very aggressive bright outlier mask -Capture extreme bright/dark statistical outliers..
  --z_tau_lo 2.0 \                      # Same for dark outliers
  --desp_max_blob 100 \                # Removes all small speckles and blobs -Remove any small clusters that resemble speckles
  --desp_dilate 5 \                    # Strong expansion of the speckle mask -Expand the speckle mask aggressively
  --desp_ksize 15 \                    # Wide median filter inpainting -Strong smoothing/inpainting for masked regions
  --suffix _den_ultra \
  --out_dtype uint16 \
  --print_stats \
  --save_debug

 python predict_tiff-despeckle.py \
  --ckpt data/n2same_het_pretrained.pt \
  --inp data/noisy \
  --out_dir data/preds_desp_ultra_max \
  --tile 1600 \
  --overlap 192 \
  --autocast \
  --norm percentile \
  --norm_pctl 0.5 99.5 \
  --global_norm \
  --global_take 64 \
  --brightness_match \
  --despeckle \
  --desp_source denoised \
  --desp_tau 0.5 \                      # Most sensitive threshold
  --hi_pctl 99.80 \                     # Even broader bright spot range
  --lo_pctl 0.20 \                      # Same for dark
  --z_tau_hi 1.5 \                      # Capture faint bright outliers
  --z_tau_lo 1.5 \                      # Capture faint dark outliers
  --desp_max_blob 50 \                 # Remove even medium-sized blobs
  --desp_dilate 7 \                    # Wide mask expansion
  --desp_ksize 21 \                    # Strongest median filter patch
  --suffix _den_ultra_max \
  --out_dtype uint16 \
  --print_stats \
  --save_debug 



✅ Despeckle Parameters in Training

(from train_n2same_het_multi_progress_metrics_stable-despeckle.py)

Argument    Description Default     Notes
--speckle_weight  Downweighting factor for loss on speckle pixels 0.0   0.0 means no downweighting; no masking
--hi_pctl   High intensity percentile threshold for masking 99.98 Higher = more aggressive
--lo_pctl   Low intensity percentile threshold for masking  0.02  Lower = more aggressive
--z_tau_hi  Z-score threshold for high outliers 5.0   Lower = more aggressive
--z_tau_lo  Z-score threshold for low outliers  5.0   Lower = more aggressive
--speckle_dilate  Dilation kernel size for mask smoothing   3     0 disables dilation
--charb_w   Charbonnier loss weight on masked speckle pixels      0.2   0 disables it

Setting --speckle_weight 0 disables the masking during training, i.e. pure denoising without speckle suppression.

✅ Despeckle Parameters in Inference

(from predict_tiff-despeckle.py)

Argument    Description Default     Notes
--despeckle Enable despeckling      off   Must be set to enable
--desp_tau  Residual MAD threshold (both tails) 3.0   Higher = more aggressive
--hi_pctl   High percentile cutoff on input (normalized 0–1)      99.95 Higher = stricter (more mask)
--lo_pctl   Low percentile cutoff   0.05  Lower = stricter (more mask)
--z_tau_hi  Z-score cutoff for bright outliers  6.0   Lower = more aggressive
--z_tau_lo  Z-score cutoff for dark outliers    6.0   Lower = more aggressive
--desp_max_blob   Max size of blobs to keep (0 disables size filter)    600   Smaller = more aggressive
--desp_dilate     Dilation iterations for the speckle mask  1     0 disables dilation
--desp_source     Use denoised or noisy as input for mask computation   denoised    Can affect mask shape

Setting --despeckle off means no despeckle at inference — useful for evaluating clean denoising quality alone.

🔍 Summary: How to Control Despeckling

Disable entirely (pure denoising):

Training: --speckle_weight 0 --charb_w 0

Inference: Omit --despeckle

Make it more aggressive:

Lower: --z_tau_hi, --z_tau_lo, --hi_pctl, --lo_pctl

Increase: --desp_dilate, --desp_tau

(Training) Use higher --charb_w for masked pixel guidance.




Part 1: For the Best balance of speed + quality + reliability of our neutron instrument pipelines: We implemented TomoPy (preprocess + COR) → ASTRA FBP_CUDA / SIRT_CUDA. The other two options:
A. If the dataset is noisy or contains artifacts: 👉 CIL + ASTRA (TV/TGV regularised iterative reconstruction)
B. If we need production-line reconstruction: 👉 HTTomo with distributed execution.

Step 0: One “core recon” env for all 3D recon experiments:
TomoPy (preprocess + COR) → ASTRA FBP_CUDA / SIRT_CUDA

conda create -n recon python=3.10 -y
conda activate recon

# core numerics & IO
conda install -c conda-forge h5py tifffile -y
conda install numpy=1.26.4 -y
conda install scipy=1.11.4 -y
# Either via conda:
conda install -c conda-forge nibabel -y

# Or via pip (also ok):
# python -m pip install nibabel

# | Library           | Version         |
| ----------------- | --------------- |
| **numpy**         | **1.26.4**      |
| **scipy**         | **1.11.4**      |
| **h5py**          | 3.10–3.15       |
| **tomopy**        | Latest (1.14.x) |
| **astra-toolbox** | 2.4.0           |

# tomopy (CPU) – if we want GPU via ASTRA, ASTRA handles it
conda install -c conda-forge tomopy -y

# astra toolbox (check docs for CUDA support)
conda install -c astra-toolbox astra-toolbox -y

# CIL (and its regularisation toolkit)
conda install -c conda-forge -c ccpi cil ccpi-regulariser -y


Step 1: A standalone COR sweep script that tries many centers, reconstructs one slice per center, and saves them so we can visually pick the sharpest, three options:

python recon_tomopy_astra_fbp.py \
  --projs_dir data/scan \
  --pattern "to_tooth_*.tif" \
  --out_dir data/recon_preview \
  --preview_rows 600 900 \
  --center 3124 \
  --median 3 \
  --stripe fw \
  --downsample 2 \
  --mask 0.98 \
  --save tiff

python center_sweep_tomopy_astra.py \
  --projs_dir data/scan \
  --pattern "to_tooth_*.tif" \
  --out_dir data/center_sweep \
  --center_start 3000 \
  --center_end 3260 \
  --center_step 2 \
  --row 750 \
  --downsample 2 \
  --median 3 \
  --stripe fw \
  --algorithm gridrec

python auto_center_detect.py \
  --projs_dir data/scan \
  --pattern "to_tooth_*.tif" \
  --flats_dir data/scan \
  --darks_dir data/scan \
  --row 1500 \
  --downsample 2 \
  --center_start 3000 \
  --center_end 3260 \
  --center_step 2 \
  --median 3 \
  --stripe fw \
  --out_dir data/auto_center

Step 2 – Full-resolution recon once COR is known
suppose center is ≈ 3174.0:

python recon_tomopy_astra_fbp.py \
  --projs_dir data/scan \
  --pattern "to_tooth_*.tif" \
  --flats_dir data/scan \
  --darks_dir data/scan \
  --out_dir data/recon_full_nii \
  --center 3174.0 \
  --median 3 \
  --stripe fw \
  --downsample 2 \
  --mask 0.98 \
  --save nii \
  --nii_name tooth_recon.nii.gz

Part 2:
already got a full 3D tooth volume:
data/recon_full_nii/tooth_recon.nii.gz   # 1024 × 1024 × 1024, downsample=2
monai_benchmark_suite.py with these sub-commands:
1. QA + labelling
2. Organise into MONAI dataset (imagesTr / labelsTr)
3. Make folds (even if tiny)
4. Train 3D UNet / UNETR
5. Infer + evaluate

0. Environment (suggested)
Use a separate env from recon for MONAI + PyTorch:
conda create -n monai-bench python=3.10 -y

# Either via conda:
conda install -c conda-forge nibabel -y
# Or via pip (also ok):
# python -m pip install nibabel

conda install -c conda-forge napari[all] -y
or
python -m pip install napari[all]

conda activate monai-bench

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install monai==1.3.1 nibabel==5.2.1 tifffile==2024.5.22 \
  scikit-image==0.22.0 pandas==2.2.2 numpy==1.26.4 scipy==1.11.4 tqdm==4.66.4 einops==0.7.0 timm==0.9.12

1.Convert stack to 5 blocks:
python monai_benchmark_suite.py tiff_to_blocks \
  --images_dir data/recon_full \
  --out_images_dir data/nii/imagesTr \
  --k 5 \
  --prefix tooth \
  --pattern "recon_z*.tif" \
  --spacing 1 1 1

2. Create labels: manually or via Slicer/ITK-SNAP:
data/nii/labelsTr/tooth_00.nii.gz
data/nii/labelsTr/tooth_01.nii.gz
...

3. Generate folds:
python monai_benchmark_suite.py generate_folds \
  --images_dir data/nii/imagesTr \
  --labels_dir data/nii/labelsTr \
  --out_dir folds_tooth3d --k 5

4. Train 3D UNet on fold1:
python monai_benchmark_suite.py train \
  --fold_json folds_tooth3d/fold1.json \
  --spatial_dims 3 --model unet \
  --roi 128 128 128 --batch 2 --epochs 200 --amp \
  --out_dir runs/tooth_3dunet_fold1 --num_classes 2

5. Inference on the same case
python monai_benchmark_suite.py infer \
  --fold_json folds_tooth/fold1.json \
  --ckpt runs/tooth_3dunet_fold1/best.pt \
  --spatial_dims 3 \
  --model unet \
  --roi 128 128 128 \
  --overlap 0.5 \
  --amp \
  --out_dir preds/tooth_3dunet_fold1 \
  --num_classes 2 \
  --spacing 1 1 1 \
  --workers 4

get:
preds/tooth_3dunet_fold1/tooth_00_pred.nii.gz

Open that in 3D Slicer/ITK-SNAP and compare with tooth_00.nii.gz label.

6. Evaluate (Dice / IoU)
python monai_benchmark_suite.py eval \
  --fold_json folds_tooth/fold1.json \
  --preds preds/tooth_3dunet_fold1 \
  --metrics dice iou \
  --csv results/tooth_3dunet_fold1.csv

This will print Dice/IoU for our one case and save to CSV.



The current scripts, gridrec use a fixed Parzen filter, and the ASTRA FBP call does not set any filter (so it uses ASTRA’s default)
Hamming will improve our reconstruction for noisy neutron data, and it’s a well trade-off:
•     Ram-Lak (ramp): sharpest edges, max noise amplification.
•     Hamming / Hann / Cosine: apodized ramps → less noise/ringing, slightly lower high-frequency contrast.
•     Parzen: even stronger smoothing (what the current scripts use now), typically lowest noise, but can blur fine details.

Given our high-resolution, denoised 16-bit projections, Hamming (or Hann) usually gives a better noise-detail balance than Parzen.
I updated the both scripts.
Setup  ASTRA FBP and Hamming only:
python center_sweep_tomopy_astra_patched.py ^
  --projs_dir scan ^
  --pattern "DINGO_*.tif" ^
  --out_dir sweep_test_astra ^
  --theta_start 0 --theta_end 180 ^
  --center_start 3100 --center_end 3150 --center_step 2 ^
  --row 1600 ^
  --algorithm astra_fbp ^
  --filter_sweep
(For ASTRA, unsupported filters are skipped, so we will see warnings for parzen and ram-lak is fine, etc.)

python recon_tomopy_astra_fbp_patched.py ^
  --projs_dir scan ^
  --pattern "DINGO_*.tif" ^
  --out_dir recon_hamming ^
  --center 3124.0 ^
  --median 3 --stripe fw --downsample 1 ^
  --mask 0.98 ^
  --filter hamming ^
  --save nii --nii_name recon_dingo_hamming.nii.gz

  


