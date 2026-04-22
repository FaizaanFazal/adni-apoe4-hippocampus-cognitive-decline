# APOE ε4 as a Predictor of Cognitive Decline and Its Interaction with Hippocampal Volume in Alzheimer's Disease

> Code repository for my paper published in *Frontiers in Aging Neuroscience* (2026)  doi: 10.3389/fnagi.2026.1730265

---

## Overview

This is the full analysis pipeline I built for my study examining how APOE ε4 allele dosage and ICV-adjusted hippocampal volume jointly predict the rate of cognitive decline across cognitively normal (CN), mild cognitive impairment (MCI), and Alzheimer's disease (AD) diagnostic groups.

The study uses longitudinal data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**, spanning nearly 20 years of follow-up. After merging and cleaning 10 raw ADNI tables, I arrived at a final analysis sample of **2,417 subjects** across **11,793 longitudinal visits**.

### What I found

My primary analysis is a 3-way interaction: **Time × APOE4_DOSE × HIPPO_ICV_ADJ** — asking whether the protective effect of larger hippocampal volume against cognitive decline is moderated by APOE ε4 dosage. Key findings:

- The 3-way interaction was significant for **MMSE** (β = −0.79, p = 0.030) and **CDR-SB** (β = +0.47, p = 0.037)
- APOE ε4 homozygotes (dose 2) had 50% conversion rate from MCI → AD, compared to 25.6% in non-carriers
- Hippocampal volume × time interaction was highly significant across all three cognitive outcomes (p < 0.001)
- Cox PH survival model: 845 conversion events (35%) among 2,417 subjects over max 18.8 years

### Sample at a glance

| Group | N | % |
|---|---:|---:|
| APOE ε4 dose 0 (non-carrier) | 1,322 | 54.7% |
| APOE ε4 dose 1 (heterozygous) | 870 | 36.0% |
| APOE ε4 dose 2 (homozygous) | 225 | 9.3% |
| Baseline CN | 901 | 37.3% |
| Baseline MCI | 1,108 | 45.8% |
| Baseline AD | 408 | 16.9% |

---

## Data Availability

I am **not able to include the raw ADNI data** in this repository. ADNI data is governed by a Data Use Agreement that prohibits redistribution. If you want to reproduce my analysis, you will need to apply for ADNI access yourself at:

**https://adni.loni.usc.edu**

Once approved, download the following tables and place them in a `tables/` folder at the project root:

| Table | File pattern |
|---|---|
| Demographics | `All_Subjects_PTDEMOG_*.csv` |
| APOE Genotype | `All_Subjects_APOERES_*.csv` |
| MMSE | `All_Subjects_MMSE_*.csv` |
| CDR | `All_Subjects_CDR_*.csv` |
| ADAS-Cog13 | `All_Subjects_ADAS_*.csv` |
| Diagnosis | `All_Subjects_DXSUM_*.csv` |
| GDS (Depression) | `All_Subjects_GDSCALE_*.csv` |
| NPI-Q | `All_Subjects_NPIQ_*.csv` |
| FreeSurfer Imaging | `All_Subjects_UCSFFSX7_*.csv` |
| Survival/Baseline | `baseline_subj.csv` |

The `reports/` folder in this repo contains all my **aggregated outputs** (model coefficients, figures, Table 1, sensitivity summaries) — no individual subject data.

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/adni-apoe4-hippocampus-cognitive-decline.git
cd adni-apoe4-hippocampus-cognitive-decline
```

### 2. Set up the Python environment

I built and ran everything on **Python 3.10.20** using Conda. I recommend creating a dedicated environment to avoid any package conflicts.

**Using Conda (recommended):**

```bash
# Create environment with Python 3.10.20
conda create -n adni_notebooks python=3.10.20 -y

# Activate it
conda activate adni_notebooks

# Install all required packages
pip install -r requirements_notebooks.txt
```

**Using pip only (no Conda):**

```bash
# Python 3.10.x required — newer versions may have compatibility issues
# with some statsmodels internals used in the LME models
pip install -r requirements_notebooks.txt
```

> I specifically used Python **3.10.20** because some f-string syntax restrictions in 3.10 affect how Jupyter formats expressions — 3.11+ would also work but I have not tested it.

### 3. Add your ADNI data

Place all downloaded ADNI CSVs into `tables/` at the project root. The notebooks read from that folder using absolute paths — you may need to update the `BASE` path variable at the top of each notebook to match your local directory.

```python
# At the top of each notebook, update this:
BASE = '/your/local/path/to/adni-apoe4-hippocampus-cognitive-decline'
```

### 4. Run the notebooks in order

```bash
jupyter notebook
```

Then run each notebook sequentially — each one depends on the outputs of the previous:

| Notebook | Purpose | Output |
|---|---|---|
| `01_data_pipeline.ipynb` | Merge all 10 ADNI tables, clean, filter complete cases | `reports/ADNI_*.csv` |
| `02_descriptive_stats.ipynb` | Table 1, group comparisons, exploratory figures | Figs 1–6 |
| `03_lme_analysis.ipynb` | Linear Mixed-Effects models + forest plots + trajectories | LME summaries, Figs 7–15 |
| `04_survival_sensitivity.ipynb` | Cox PH, Kaplan-Meier, 4 sensitivity analyses | Figs 16–20, sensitivity tables |
| `05_detailed_counts.py` | Exact conversion counts, GDS, age 85+, follow-up by Dx | Printed to console |

All processed outputs go into `reports/`. The `reports/ADNI_*.csv` files (subject-level) are excluded from this repo via `.gitignore`.

### Requirements

Full package list is in `requirements_notebooks.txt`. Key dependencies:

```
pandas==2.2.3
numpy==1.26.4
scipy==1.13.1
statsmodels>=0.14.0      # LME models via mixedlm
lifelines>=0.27.0        # Cox PH and Kaplan-Meier
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
patsy>=0.5.6             # formula parsing for LME
```

---

## Plan and Implementation

My original analysis plan is in the `plan/` folder — I kept it versioned here because it went through several rounds of revision based on reviewer feedback.

### What I originally planned (`plan/ADNI_Revision_Plan.md`)

The revision plan addresses comments from Frontiers reviewers. The main asks were:

1. **Expand sample size** — from the original N=133 to the full ADNI cohort (target: 2,000+ subjects)
2. **Keep hippocampal volume continuous** — the original paper split by median; Reviewer 1 correctly pushed back on this
3. **Add GDS and NPI-Q** as depression and neuropsychiatric covariates (Reviewer 2)
4. **Use APOE ε4 allele dose (0/1/2)** instead of binary carrier/non-carrier
5. **Report survival analysis** — time to conversion, not just cross-sectional differences

### How I implemented it (`plan/adni_pipeline.py`)

The `plan/adni_pipeline.py` file was my first draft of the full pipeline as a single CLI script. I later broke it into the 4 notebooks for better interactivity and reproducibility — but the `.py` file is still there as a reference for the logic.

### Key implementation decisions I made

**QC filtering**: The UCSFFSX7 `OVERALLQC` column has values `Pass`, `Partial`, `Hippocampus Only`, `Fail`, and `NaN`. I exclude only `Fail` (4 observations). The 11,091 NaN rows all have valid hippocampal measurements — I confirmed this by checking ST29SV/ST30SV completeness. Using `== 'Pass'` (the naive filter) would have dropped 97% of valid imaging data.

**Merge strategy**: I use MMSE as the backbone with left joins for all other tables. Outer joins were creating ~582 extra null-MMSE rows that failed the complete-case filter — left join preserves the MMSE visit structure correctly.

**Time variable**: The `baseline_subj.csv` `Time` column is in **months** (max = 226.1), not years. I convert to `TIME_YEARS` in notebook 01.

**GDS missing code**: ADNI uses `-4` as a sentinel for missing data in GDTOTAL. I replace these with `NaN` before any analysis.

**LME index bug**: statsmodels 0.14.x has an out-of-bounds index error when Patsy drops NaN rows internally but the `groups` array retains original length. My fix: use `patsy.dmatrices()` to get the post-NaN-drop index, then align the data frame to that index before fitting.

### Folder structure

```
.
├── notebooks2/              # Main analysis notebooks (run in order)
│   ├── 01_data_pipeline.ipynb
│   ├── 02_descriptive_stats.ipynb
│   ├── 03_lme_analysis.ipynb
│   ├── 04_survival_sensitivity.ipynb
│   └── 05_detailed_counts.py
├── plan/                    # Original pipeline plan and revision notes
│   ├── ADNI_Revision_Plan.md
│   └── adni_pipeline.py
├── reports/                 # Outputs: figures, model summaries, aggregate tables
│   ├── Fig*.png
│   ├── LME_*.csv / *.txt
│   └── Sensitivity_*.csv
├── tables/                  # Raw ADNI CSVs — NOT included (see Data Availability)
├── requirements_notebooks.txt
├── requirements.txt
└── .gitignore
```

---

## Citation

If you use my pipeline or build on it, please cite the paper:

```
doi: 10.3389/fnagi.2026.1730265
```

---

## Contact

If you have questions about the analysis or run into issues reproducing results, feel free to open a GitHub Issue.
