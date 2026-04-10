#!/usr/bin/env python3
"""
05_detailed_counts.py — Detailed breakdown of conversion, GDS, age, and follow-up
Run: python notebooks2/05_detailed_counts.py
"""

import pandas as pd
import numpy as np
import os

BASE    = '/media/faizaan/4TB/1_DATA_PROJECTS/Projects/Multimodel_study'
REPORTS = os.path.join(BASE, 'reports')

complete = pd.read_csv(os.path.join(REPORTS, 'ADNI_Complete_Cases.csv'))
baseline = pd.read_csv(os.path.join(REPORTS, 'ADNI_Baseline_Analysis.csv'))
longit   = pd.read_csv(os.path.join(REPORTS, 'ADNI_Longitudinal_Analysis.csv'))

SEP = '=' * 70

# ============================================================
# 1. EXACT CONVERSION COUNTS
# ============================================================
print(SEP)
print('1. EXACT CONVERSION COUNTS')
print(SEP)

# For each subject: baseline Dx and worst (latest) Dx reached
subj_dx = complete.sort_values(['RID', 'YEARS_FROM_BL']).copy()

# Map diagnosis labels to ordinal for progression tracking
dx_order = {'CN': 0, 'MCI': 1, 'AD': 2}
subj_dx['DX_ORD'] = subj_dx['DX_LABEL'].map(dx_order)

# Per-subject: baseline Dx, worst Dx reached, and whether Event=1
subj_summary = []
for rid, grp in subj_dx.groupby('RID'):
    bl_dx = grp['BL_DX_LABEL'].iloc[0]
    event = grp['Event'].iloc[0]
    # All diagnoses observed for this subject (drop NaN)
    dx_sequence = grp['DX_LABEL'].dropna().tolist()
    worst_dx_ord = grp['DX_ORD'].dropna().max()
    worst_dx = {0: 'CN', 1: 'MCI', 2: 'AD'}.get(worst_dx_ord, 'Unknown')
    # Check specific transitions
    subj_summary.append({
        'RID': rid,
        'BL_DX': bl_dx,
        'Event': event,
        'Worst_DX': worst_dx,
        'N_visits': len(grp),
        'DX_sequence': dx_sequence,
    })

sdf = pd.DataFrame(subj_summary)

print(f'\nTotal subjects: {len(sdf):,}')
print(f'Total with Event=1: {(sdf["Event"]==1).sum():,} ({(sdf["Event"]==1).mean()*100:.1f}%)')
print(f'Total with Event=0: {(sdf["Event"]==0).sum():,} ({(sdf["Event"]==0).mean()*100:.1f}%)')

# Event=1 broken down by baseline diagnosis
print(f'\n{"─"*50}')
print('Event=1 (conversion) by BASELINE diagnosis:')
print(f'{"─"*50}')
for bl_dx in ['CN', 'MCI', 'AD']:
    sub = sdf[sdf['BL_DX'] == bl_dx]
    events = sub[sub['Event'] == 1]
    print(f'\n  Baseline {bl_dx}: {len(sub):,} subjects, {len(events):,} with Event=1 ({len(events)/len(sub)*100:.1f}%)')

# Detailed transition matrix
print(f'\n{"─"*50}')
print('DIAGNOSIS TRANSITION MATRIX (Baseline → Worst observed Dx):')
print(f'{"─"*50}')
ct = pd.crosstab(sdf['BL_DX'], sdf['Worst_DX'], margins=True)
ct = ct.reindex(index=['CN', 'MCI', 'AD', 'All'], columns=['CN', 'MCI', 'AD', 'All'])
print(f'\n{ct.to_string()}')

# Specific transition counts
cn_to_mci = len(sdf[(sdf['BL_DX'] == 'CN') & (sdf['Worst_DX'] == 'MCI')])
cn_to_ad  = len(sdf[(sdf['BL_DX'] == 'CN') & (sdf['Worst_DX'] == 'AD')])
mci_to_ad = len(sdf[(sdf['BL_DX'] == 'MCI') & (sdf['Worst_DX'] == 'AD')])
cn_stable = len(sdf[(sdf['BL_DX'] == 'CN') & (sdf['Worst_DX'] == 'CN')])
mci_stable = len(sdf[(sdf['BL_DX'] == 'MCI') & (sdf['Worst_DX'] == 'MCI')])
mci_revert = len(sdf[(sdf['BL_DX'] == 'MCI') & (sdf['Worst_DX'] == 'CN')])

print(f'\n  Specific transitions:')
print(f'    CN → CN (stable):       {cn_stable:>5,}')
print(f'    CN → MCI (progression): {cn_to_mci:>5,}')
print(f'    CN → AD  (progression): {cn_to_ad:>5,}')
print(f'    MCI → CN (reversion):   {mci_revert:>5,}')
print(f'    MCI → MCI (stable):     {mci_stable:>5,}')
print(f'    MCI → AD (progression): {mci_to_ad:>5,}')

# By APOE dose
print(f'\n{"─"*50}')
print('Conversions (Event=1) by BASELINE DX × APOE4_DOSE:')
print(f'{"─"*50}')
bl = baseline.copy()
for bl_dx in ['CN', 'MCI', 'AD']:
    sub = bl[bl['BL_DX_LABEL'] == bl_dx]
    print(f'\n  Baseline {bl_dx} (n={len(sub):,}):')
    for dose in [0.0, 1.0, 2.0]:
        dsub = sub[sub['APOE4_DOSE'] == dose]
        events = dsub['Event'].sum()
        n = len(dsub)
        pct = events / n * 100 if n > 0 else 0
        print(f'    APOE4 dose {int(dose)}: {n:>5,} subjects, {int(events):>4,} events ({pct:.1f}%)')


# ============================================================
# 2. GDS DISTRIBUTION
# ============================================================
print(f'\n{SEP}')
print('2. GDS (GERIATRIC DEPRESSION SCALE) DISTRIBUTION')
print(SEP)

gds_bl = baseline['GDTOTAL'].dropna()
gds_all = complete['GDTOTAL'].dropna()

print(f'\n  Baseline (N={len(baseline):,}):')
print(f'    GDS available:  {len(gds_bl):,} ({len(gds_bl)/len(baseline)*100:.1f}%)')
print(f'    GDS missing:    {len(baseline)-len(gds_bl):,} ({(len(baseline)-len(gds_bl))/len(baseline)*100:.1f}%)')
print(f'    Mean (SD):      {gds_bl.mean():.2f} ({gds_bl.std():.2f})')
print(f'    Median [IQR]:   {gds_bl.median():.0f} [{gds_bl.quantile(.25):.0f}–{gds_bl.quantile(.75):.0f}]')
print(f'    Range:          [{gds_bl.min():.0f}, {gds_bl.max():.0f}]')

# Score distribution
print(f'\n  Baseline GDS score distribution:')
for score in range(int(gds_bl.max()) + 1):
    n = (gds_bl == score).sum()
    pct = n / len(gds_bl) * 100
    bar = '█' * int(pct)
    print(f'    GDS={score:>2}: {n:>5,} ({pct:>5.1f}%) {bar}')

# Clinical cutoffs
gds_5_bl = (gds_bl >= 5).sum()
gds_10_bl = (gds_bl >= 10).sum()
print(f'\n  Clinical cutoffs (baseline):')
print(f'    GDS ≥ 5 (mild depression):    {gds_5_bl:>5,} ({gds_5_bl/len(gds_bl)*100:.1f}%)')
print(f'    GDS ≥ 10 (moderate-severe):   {gds_10_bl:>5,} ({gds_10_bl/len(gds_bl)*100:.1f}%)')

# By diagnosis
print(f'\n  GDS ≥ 5 by baseline diagnosis:')
for dx in ['CN', 'MCI', 'AD']:
    sub_gds = baseline[baseline['BL_DX_LABEL'] == dx]['GDTOTAL'].dropna()
    n5 = (sub_gds >= 5).sum()
    print(f'    {dx}: {n5:>4,} / {len(sub_gds):,} ({n5/len(sub_gds)*100:.1f}%) with GDS≥5,  mean GDS={sub_gds.mean():.2f}')

# Longitudinal (all visits)
gds_5_all = (gds_all >= 5).sum()
gds_10_all = (gds_all >= 10).sum()
print(f'\n  All visits (N={len(gds_all):,}):')
print(f'    GDS ≥ 5:   {gds_5_all:>6,} ({gds_5_all/len(gds_all)*100:.1f}%)')
print(f'    GDS ≥ 10:  {gds_10_all:>6,} ({gds_10_all/len(gds_all)*100:.1f}%)')


# ============================================================
# 3. AGE 85+ COUNTS
# ============================================================
print(f'\n{SEP}')
print('3. AGE 85+ SUBJECTS')
print(SEP)

age_bl = baseline['AGE'].dropna()
print(f'\n  Overall age at baseline (N={len(age_bl):,}):')
print(f'    Mean (SD):    {age_bl.mean():.1f} ({age_bl.std():.1f})')
print(f'    Median [IQR]: {age_bl.median():.0f} [{age_bl.quantile(.25):.0f}–{age_bl.quantile(.75):.0f}]')
print(f'    Range:        [{age_bl.min():.0f}, {age_bl.max():.0f}]')

# Age brackets
brackets = [(0, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 200)]
print(f'\n  Age distribution (baseline):')
for lo, hi in brackets:
    label = f'{lo}–{hi-1}' if hi < 200 else f'{lo}+'
    n = ((age_bl >= lo) & (age_bl < hi)).sum()
    pct = n / len(age_bl) * 100
    bar = '█' * int(pct / 2)
    print(f'    {label:>7}: {n:>5,} ({pct:>5.1f}%) {bar}')

# 85+ deep dive
old = baseline[baseline['AGE'] >= 85].copy()
n_old = len(old)
print(f'\n  AGE ≥ 85 deep dive:')
print(f'    Total:            {n_old:>5,} subjects ({n_old/len(baseline)*100:.1f}%)')
print(f'    Mean age:         {old["AGE"].mean():.1f}')

# APOE4+ among 85+
apoe4_pos_old = (old['APOE4_DOSE'] > 0).sum()
apoe4_neg_old = (old['APOE4_DOSE'] == 0).sum()
print(f'    APOE4+ (≥1 allele): {apoe4_pos_old:>4,} ({apoe4_pos_old/n_old*100:.1f}%)')
print(f'    APOE4-  (0 alleles): {apoe4_neg_old:>4,} ({apoe4_neg_old/n_old*100:.1f}%)')

# By dose
for dose in [0, 1, 2]:
    n = (old['APOE4_DOSE'] == dose).sum()
    print(f'    APOE4 dose {dose}:     {n:>4,} ({n/n_old*100:.1f}%)')

# Diagnosis in 85+
print(f'\n    Diagnosis in 85+ group:')
for dx in ['CN', 'MCI', 'AD']:
    n = (old['BL_DX_LABEL'] == dx).sum()
    print(f'      {dx}: {n:>4,} ({n/n_old*100:.1f}%)')

# Event rate in 85+
events_old = old['Event'].sum()
print(f'\n    Events (conversions): {int(events_old):>4,} ({events_old/n_old*100:.1f}%)')

# Comparison: 85+ vs <85
young = baseline[baseline['AGE'] < 85]
print(f'\n  Comparison: Age <85 vs ≥85:')
print(f'    {"Metric":<25} {"<85 (n={})".format(len(young)):>20} {"≥85 (n={})".format(n_old):>20}')
print(f'    {"─"*65}')
for var, label in [('MMSCORE', 'MMSE'), ('HIPPO_ICV_ADJ', 'Hippo (ICV-adj)'),
                   ('APOE4_DOSE', 'APOE4 dose'), ('Event', 'Event rate')]:
    m_y = young[var].mean()
    m_o = old[var].mean()
    print(f'    {label:<25} {m_y:>20.2f} {m_o:>20.2f}')


# ============================================================
# 4. FOLLOW-UP BY DIAGNOSIS
# ============================================================
print(f'\n{SEP}')
print('4. FOLLOW-UP BY BASELINE DIAGNOSIS')
print(SEP)

# Visits per subject
visit_stats = complete.groupby('RID').agg(
    N_visits=('VISCODE2', 'count'),
    Max_years=('YEARS_FROM_BL', 'max'),
    BL_DX=('BL_DX_LABEL', 'first')
).reset_index()

print(f'\n  {"Metric":<30} {"CN":>12} {"MCI":>12} {"AD":>12} {"Overall":>12}')
print(f'  {"─"*78}')

for dx in ['CN', 'MCI', 'AD', 'Overall']:
    sub = visit_stats if dx == 'Overall' else visit_stats[visit_stats['BL_DX'] == dx]
    n = len(sub)

    if dx == 'CN':
        cn_stats = sub
    elif dx == 'MCI':
        mci_stats = sub
    elif dx == 'AD':
        ad_stats = sub

# Print table
def fmt(val, dec=1):
    return f'{val:.{dec}f}'

for metric, col, dec in [
    ('N subjects',          None, 0),
    ('Total visits',        None, 0),
    ('Visits/subject mean', 'N_visits', 1),
    ('Visits/subject median','N_visits', 0),
    ('Visits/subject range', 'N_visits', 0),
    ('Follow-up years mean','Max_years', 1),
    ('Follow-up years median','Max_years', 1),
    ('Follow-up years SD',  'Max_years', 1),
    ('Follow-up years range','Max_years', 1),
]:
    vals = []
    for dx in ['CN', 'MCI', 'AD', 'Overall']:
        sub = visit_stats if dx == 'Overall' else visit_stats[visit_stats['BL_DX'] == dx]
        if metric == 'N subjects':
            vals.append(f'{len(sub):,}')
        elif metric == 'Total visits':
            vals.append(f'{sub["N_visits"].sum():,}')
        elif 'mean' in metric:
            vals.append(fmt(sub[col].mean(), dec))
        elif 'median' in metric:
            vals.append(fmt(sub[col].median(), dec))
        elif 'SD' in metric:
            vals.append(fmt(sub[col].std(), dec))
        elif 'range' in metric:
            vals.append(f'[{sub[col].min():.0f}–{sub[col].max():.0f}]')
    print(f'  {metric:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}')

# Visit distribution by diagnosis
print(f'\n  Visit count distribution by baseline diagnosis:')
print(f'  {"Visits":<10} {"CN":>8} {"MCI":>8} {"AD":>8} {"Total":>8}')
print(f'  {"─"*42}')
for nv in range(1, 11):
    cn_n = (visit_stats[visit_stats['BL_DX']=='CN']['N_visits'] == nv).sum()
    mci_n = (visit_stats[visit_stats['BL_DX']=='MCI']['N_visits'] == nv).sum()
    ad_n = (visit_stats[visit_stats['BL_DX']=='AD']['N_visits'] == nv).sum()
    tot = cn_n + mci_n + ad_n
    if tot > 0:
        print(f'  {nv:<10} {cn_n:>8,} {mci_n:>8,} {ad_n:>8,} {tot:>8,}')
nv10 = visit_stats['N_visits'] >= 10
cn_10 = (visit_stats[visit_stats['BL_DX']=='CN']['N_visits'] >= 10).sum()
mci_10 = (visit_stats[visit_stats['BL_DX']=='MCI']['N_visits'] >= 10).sum()
ad_10 = (visit_stats[visit_stats['BL_DX']=='AD']['N_visits'] >= 10).sum()
print(f'  {"10+":<10} {cn_10:>8,} {mci_10:>8,} {ad_10:>8,} {cn_10+mci_10+ad_10:>8,}')

# Follow-up years distribution
print(f'\n  Follow-up duration distribution by baseline diagnosis:')
print(f'  {"Years":<12} {"CN":>8} {"MCI":>8} {"AD":>8} {"Total":>8}')
print(f'  {"─"*44}')
year_brackets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20)]
for lo, hi in year_brackets:
    cn_n = ((visit_stats[visit_stats['BL_DX']=='CN']['Max_years'] >= lo) &
            (visit_stats[visit_stats['BL_DX']=='CN']['Max_years'] < hi)).sum()
    mci_n = ((visit_stats[visit_stats['BL_DX']=='MCI']['Max_years'] >= lo) &
             (visit_stats[visit_stats['BL_DX']=='MCI']['Max_years'] < hi)).sum()
    ad_n = ((visit_stats[visit_stats['BL_DX']=='AD']['Max_years'] >= lo) &
            (visit_stats[visit_stats['BL_DX']=='AD']['Max_years'] < hi)).sum()
    label = f'{lo}–{hi}y'
    print(f'  {label:<12} {cn_n:>8,} {mci_n:>8,} {ad_n:>8,} {cn_n+mci_n+ad_n:>8,}')

print(f'\n{SEP}')
print('DONE')
print(SEP)
