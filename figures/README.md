# Manuscript figures

The figures use Turkish axis labels to match the manuscript. `plots/` contains the model diagram, the two-panel ROC figure, training trajectories and the paired differences for the ten official segmentation test images.

From the repository root, with NumPy and Matplotlib 3.10 installed, run:

```bash
python figures/plot_curves.py figures/data figures/plots
python figures/model_diagram.py figures/plots
```

`data/roc.csv` stores the interpolated mean true-positive rate at 501 false-positive rates and the mean of the original, unrounded AUROCs. For ColonCancerHSI, each image has equal weight within a training seed and each of the ten seeds has equal weight within a method. The GM–TensorSSA curve uses the difference between the two residual-based decision scores. For the Borkovits classification task, the three held-out folds are combined within each initialization before averaging three initialization curves. No patient or tissue-core predictions are distributed here.

`data/loss.csv` contains epoch-level losses from accepted fits. The two ColonCancerHSI panels compare the three local projection variants. Their training loss is cross-entropy plus Dice loss and their validation loss is sparse cross-entropy. The Borkovits panel shows training cross-entropy for the three classification variants. Thin lines end at the last recorded epoch. Thick means end when the first contributing run ends. Folds are only defined for Borkovits. No early-stopped history was padded.

`data/goruntu_farklari.csv` contains ten image-level macro-F1 differences in percentage points. The second and third columns give the learned projection minus the fixed random projection and PCA8, respectively, averaged over ten matched seeds.

To reproduce the paired image figure, install ReportLab and pypdfium2 and provide the paths to regular and bold TrueType fonts with Turkish glyphs. The published plot used Arial.

```bash
python figures/plot_image_differences.py figures/data/goruntu_farklari.csv figures/plots/per_image_differences.png /path/to/regular.ttf /path/to/bold.ttf
```
