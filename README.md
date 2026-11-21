# ColonSpec8

ColonSpec8 learns an eight-channel projection of 351 spectral bands jointly with a local 2D encoder–decoder for segmentation of colorectal liver metastases. The training objective uses cross-entropy and Dice loss on revealed pixels. The implementation name of the proposed model is `local_only_supervised`. The repository also includes the matched frozen-random and PCA8 controls, five additional reference methods, a crossed mask/initialization experiment and a separate patient-grouped FTIR core classification study.

## Published results

Official ColonCancerHSI TEST included ten densely annotated images. For each seed, binary macro-F1 was calculated per image and averaged with equal image weights. The table gives the mean and sample standard deviation across ten paired training seeds on the same split. These are not patient-level confidence intervals.

| Method | Macro-F1 (%) |
| --- | ---: |
| ColonSpec8 | 86.47 ± 0.70 |
| Frozen random projection | 85.56 ± 0.70 |
| PCA8 | 84.92 ± 0.83 |
| nnU-Net v2 2D backbone adaptation | 84.46 ± 1.24 |
| Mamba-MT | 83.45 ± 2.11 |
| U-Net-MT | 80.27 ± 2.84 |
| Inductive GM–TensorSSA | 49.67 ± 0.50 |
| HybridSN dense adaptation | 37.58 ± 8.56 |

[Per-seed metrics](results/segmentation_per_seed.csv) include tumor F1, IoU, accuracy, sensitivity, specificity, ranking, calibration and boundary metrics when defined. The [model and contrast summaries](results/segmentation_summary.json) include all eight methods. [Checkpoint index](results/segmentation_checkpoints.json) identifies the selected models by SHA-256. SpecTr reached the profiling time limit and has no measured TEST score.

See [figures](figures/) for the plotting scripts and their source tables. The [provenance index](results/provenance.json) links each main result file to the source report digest.

In the separate 3 × 3 sparse-validation control, the learned projection exceeded the frozen random projection in nine matched conditions. The mean fixed-epoch difference was 2.72 percentage points. [Paired endpoints](results/mask_initialization.json).

On the independent FTIR collection, the three out-of-fold core-level macro-F1 means across three initializations were 76.84% (learned), 69.18% (random) and 67.93% (PCA8). This experiment trained new classifiers on FTIR spectra. It evaluates method transfer to a different task. External segmentation performance of the original frozen model remains unknown. [FTIR classification results and patient-cluster intervals](results/ftir_classification.json).

## Data

* **ColonCancerHSI.** Download the 57 original files from the [authors' data directory](https://data.fulir.irb.hr/data/ColonCancerHSI_Data/) into `data/raw/` using their published names. The [input checksum list](configs/data_checksums.json) contains file names, sizes and SHA-256 digests. Preparation uses the 17 official development images, including 14 for training and three for sparse validation. Each sparse mask reveals 0.5% of the pixels for the reported experiments. The official TEST masks enter only the evaluation command. [Dataset record](https://data.fulir.irb.hr/object/irb:538) and [original method](https://doi.org/10.1016/j.compbiomed.2025.110841).
* **FTIR.** Obtain `multimodal_CRC_dataset.zip` from the [Borkovits et al. dataset](https://doi.org/10.5281/zenodo.17790337). The importer checks member CRCs and the published spectral MD5 values, selects 117 normal and 121 cancer cores, and groups them by the 75 reported patients. All source data, patient mappings, generated predictions and weights stay local. [Data descriptor](https://doi.org/10.1038/s41597-026-07873-1).

## Installation

The study used Linux, Python 3.11, PyTorch 2.4.1 with CUDA 12.4, an NVIDIA GPU and compatible CUDA extensions. A CUDA 12.4 toolchain, a C++ compiler and a compatible driver are needed when building the extensions from source. From the repository root in a Linux shell:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation causal-conv1d==1.5.3.post1
python -m pip install --no-build-isolation mamba-ssm==2.2.6.post3
```

The historical constructor initializes the parent network before retaining its local branch. This preserves paired initial weights and requires the Mamba package during construction. ColonSpec8's forward pass uses only the local 2D network. [Third-party credits and applicable licenses](THIRD_PARTY.md).

## Reproduction

Check the released numbers using only Python's standard library:

```sh
python verify_results.py
```

Prepare the public files, create the training-only normalization and PCA transforms, and train the proposed model for one reported seed:

```sh
python run.py prepare
python run.py train --model local_only_supervised --seed 938
```

The full grid uses the eight method identifiers listed by `python run.py train --help` and seeds `938 476 993 518 663 628 828 353 621 306`. Run each pair once. Run the freeze command after all 80 fits, then evaluate the frozen pairs individually. For example:

```sh
python run.py freeze
python run.py evaluate --model local_only_supervised --seed 938
```

Training outputs go to `outputs/segmentation/<model>/<seed>/`. Dense evaluation writes the per-image metrics and primary endpoint to `outputs/test/<model>/<seed>/metrics.json`. Neural fits use at most 100 epochs, with 128 steps per epoch, validation-based checkpoint selection and patience 15. GM–TensorSSA fits a class subspace from sparse training labels. The reader needs disk space for both the raw collection and the development and test caches.

The original timed GM–TensorSSA execution shared TensorSSA features across the ten classifiers. These single-seed commands recalculate features for each seed. Their combined runtime therefore cannot be compared directly with the original shared-feature timing.

Run a crossed control on prepared development data with:

```sh
python run.py mask-control --model local_only_supervised --mask 938 --init 476
```

Run each of the two methods (`local_only_supervised`, `local_frozen_projection`) for all combinations of mask and initialization in `938 476 993`. Start the first control alone to create the shared plan and initialization files. Use sequential execution for the remaining controls. Each fit has a fixed 100-epoch budget. Outputs are in `outputs/maskstudy/`.

For the FTIR adaptation:

```sh
python run.py ftir-prepare /path/to/multimodal_CRC_dataset.zip
python run.py ftir-train --arm learned --fold 0 --init 938
```

Run the three arms (`learned`, `random`, `pca8`) for folds `0 1 2` and initializations `938 476 993`. After all 27 fits, run `python run.py ftir-evaluate`. Checkpoints and held-out predictions are written to `outputs/ftir/`. The local aggregate analysis is written to `experiments/borkovits_v1/reports/analysis_report.json`. This local analysis file contains patient-linked predictions and must be kept private. The distributed FTIR results contain aggregates only.
