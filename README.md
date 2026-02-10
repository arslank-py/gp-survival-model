# Bayesian GP survival model for TCGA pan-cancer multi-omics data

A hierarchical Bayesian survival analysis model integrating multi-omics genomic data with pathway activity signatures for pan-cancer survival prediction across 32 cancer types in The Cancer Genome Atlas (TCGA).

## Model Architecture

- **Input**: Multi-omics features (expression, copy number, mutations, RPPA) + pathway activity scores.
- **Dimensionality Reduction**: 
  - Supervised VAE for survival-aware multi-omics reduction (86k → 50 latent dimensions).
  - PCA for pathway activity reduction (1387 → 200 components).
- **Hazard Model**: Hybrid Cox proportional hazards architecture with:
  - **Baseline**: Piecewise constant hazard (50 intervals);
  - **Main Effects**: 4-layer MLP (512→256→128→64→1);
  - **Random Effects**: Gaussian Process with separable spatiotemporal covariance (Matern-3/2 × RBF kernel, 200 inducing points).
- **Training**: Variational Inference with ELBO optimization, KL warm-up, separate learning rates for GP and MLP components.

## Requirements

- Python 3.8+
- PyTorch (with CUDA support)
- NumPy, Pandas
- scikit-learn
- matplotlib, seaborn
- lifelines (optional for Kaplan-Meier baseline)

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn lifelines
```

## Data Download

The model requires TCGA Pan-Cancer multi-omics data files. Download the following files and place them in the `data/` directory:

### Required Files:

1. **Clinical/Survival Data**
   - File: `tcga_pancan_clinical`
   - Source: [TCGA Pan-Cancer Atlas](https://gdc.cancer.gov/about-data/publications/pancanatlas)
   - Direct link: [Xena Browser - TCGA Pan-Cancer Clinical Data](https://xenabrowser.net/datapages/?dataset=TCGA_pancan_clinical&host=https%3A%2F%2Ftcga.xenahubs.net)

2. **Gene Expression Data**
   - File: `tcga_pancan_expression.xena`
   - Source: [Xena Browser](https://xenabrowser.net/)
   - Direct link: [TCGA Pan-Cancer Expression](https://xenabrowser.net/datapages/?dataset=EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena&host=https%3A%2F%2Fpancanatlas.xenahubs.net)

3. **Copy Number Variation Data**
   - File: `tcga_pancan_copy_number.by_genes`
   - Source: [Xena Browser](https://xenabrowser.net/)
   - Direct link: [TCGA Pan-Cancer Copy Number (GISTIC2)](https://xenabrowser.net/datapages/?dataset=TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes&host=https%3A%2F%2Ftcga.xenahubs.net)

4. **Mutation Data**
   - File: `tcga_pancan_mutations.xena`
   - Source: [Xena Browser](https://xenabrowser.net/)
   - Direct link: [TCGA Pan-Cancer Mutations (MC3)](https://xenabrowser.net/datapages/?dataset=mc3.v0.2.8.PUBLIC.xena&host=https%3A%2F%2Ftcga.xenahubs.net)

5. **RPPA (Protein) Data** (optional)
   - File: `tcga_pancan_rppa.xena`
   - Source: [Xena Browser](https://xenabrowser.net/)
   - Direct link: [TCGA Pan-Cancer RPPA](https://xenabrowser.net/datapages/?dataset=TCGA.RPPA.sampleMap%2FRPPA&host=https%3A%2F%2Ftcga.xenahubs.net)

6. **Pathway Activity Scores (ssGSEA)**
   - File: `tcga_pancan_ssGSEA.txt`
   - Source: [TCGA Pan-Cancer Atlas](https://gdc.cancer.gov/about-data/publications/pancanatlas)
   - Note: This file contains pre-computed ssGSEA pathway activity scores.

### Alternative: BRCA Multi-Omics Data

If you prefer to use a single cancer type for testing:
- File: `tcga_brca_multiomics_survival.csv`
- Source: [TCGA BRCA Dataset](https://www.cancer.gov/tcga)

## Usage

1. **Download all required data files** and place them in the `data/` directory.

2. **Run the model**:
   ```bash
   python main.py
   ```

3. **Results** will be saved in the `results/` directory:
   - `model.pkl` - Trained model
   - `predictions.npz` - Survival predictions
   - `evaluation_metrics.csv` - Performance metrics
   - `survival_curves.png` - Visualization dashboard
   - `posterior_summary.csv` - Bayesian posterior summaries

## File Structure

```
.
├── main.py                      # Main training script
├── gp_survival_model_vi.py      # Core GP survival model
├── dimensionality_reduction.py  # VAE for dimensionality reduction
├── data_loader_pancan.py        # TCGA data loading utilities
├── evaluation_metrics.py        # Survival evaluation metrics
├── data/                        # Data directory 
└── results/                     # Output directory
```

