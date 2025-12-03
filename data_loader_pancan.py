"""
Data loader for TCGA Pan-Cancer datasets w/ alignment and integration of pan-cancer omics data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import gzip
import warnings
warnings.filterwarnings('ignore')


class MultiOmicsDataLoader:
    """Base class for loading, integrating multi-omics TCGA data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.survival_data = None
        self.omics_data = {}
        self.last_valid_sample_ids = None
    
    def align_samples(self, omics_dict: Dict[str, pd.DataFrame], 
                     patient_id_col: str = None) -> Dict[str, pd.DataFrame]:
        """
        Align samples across different omics datasets;
        standardizes TCGA sample IDs -> TCGA-XX-XXXX
        """        
        sample_sets = {}
        aligned_omics = {}
        
        for omics_type, df in omics_dict.items():
            if df is None or df.empty:
                continue
            
            samples = None
            
            if len(df.columns) > 0:
                first_col = df.columns[0]
                if 'sample' in str(first_col).lower() or 'patient' in str(first_col).lower():
                    samples = df.iloc[:, 0].values
                elif any('TCGA' in str(c) for c in df.columns[:10]):
                    samples = np.array(df.columns)
                elif any('TCGA' in str(idx) for idx in df.index[:10]):
                    samples = df.index.values
                else:
                    samples = df.iloc[:, 0].values
            
            if samples is None:
                print(f"Warning: Could not extract sample IDs for {omics_type}")
                continue
            
            normalized_samples = []
            for s in samples:
                s_str = str(s)
                if 'TCGA' in s_str:
                    parts = s_str.split('-')
                    if len(parts) >= 3:
                        normalized_samples.append('-'.join(parts[:3]))
                    else:
                        normalized_samples.append(s_str)
                else:
                    normalized_samples.append(s_str)
            
            sample_sets[omics_type] = set(normalized_samples)
            
            if any('TCGA' in str(c) for c in df.columns[:10]):
                df_aligned = df.T.copy()
                df_aligned['normalized_id'] = normalized_samples
            else:
                df_aligned = df.copy()
                df_aligned['normalized_id'] = normalized_samples
            
            aligned_omics[omics_type] = df_aligned
        
        if sample_sets:
            common_samples = set.intersection(*sample_sets.values())
            print(f"Found {len(common_samples)} common samples across all omics datasets")
            
            if len(common_samples) == 0:
                print("Warning: No common samples found. Trying pairwise alignment...")
                all_samples = set()
                for samples in sample_sets.values():
                    all_samples.update(samples)
                
                sample_counts = {}
                for samples in sample_sets.values():
                    for s in samples:
                        sample_counts[s] = sample_counts.get(s, 0) + 1
                
                common_samples = {s for s, count in sample_counts.items() if count >= 2}
                print(f"Using {len(common_samples)} samples common to at least 2 datasets")
            
            for omics_type in aligned_omics:
                aligned_omics[omics_type] = aligned_omics[omics_type][
                    aligned_omics[omics_type]['normalized_id'].isin(common_samples)
                ]
        
        return aligned_omics
    
    def prepare_survival_outcomes(self, survival_df: pd.DataFrame,
                                  time_col: str = None,
                                  event_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract survival times and event indicators
        
        Returns:
            times: Array of survival/censoring times
            events: Array of event indicators (1 = event, 0 = censored)
        """
        
        if time_col is None:
            pancan_time_cols = [c for c in survival_df.columns if c in ['OS.time', 'PFI.time', 'DSS.time', 'DFI.time']]
            if pancan_time_cols:
                time_col = pancan_time_cols[0]
            else:
                time_cols_with_dot = [c for c in survival_df.columns if c.endswith('.time') and c in survival_df.columns]
                if time_cols_with_dot:
                    time_col = time_cols_with_dot[0]
                else:
                    time_cols = [c for c in survival_df.columns if ('time' in c.lower() or 'survival' in c.lower() 
                                or 'os_time' in c.lower() or 'pfs_time' in c.lower()) 
                                and 'birth' not in c.lower() and 'age' not in c.lower() 
                                and 'diagnosis' not in c.lower() and 'collection' not in c.lower()]
                    if time_cols:
                        time_col = time_cols[0]
                    else:
                        numeric_cols = survival_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols[:50]: 
                            values = survival_df[col].dropna()
                            if len(values) > 100: 
                                median_val = values.median()
                                if 0 < median_val < 10000:
                                    time_col = col
                                    break
                        if time_col is None:
                            if len(numeric_cols) > 0:
                                first_num_col = numeric_cols[0]
                                values = survival_df[first_num_col].dropna()
                                if len(values) > 100:
                                    time_col = first_num_col
                                else:
                                    time_col = survival_df.columns[0]
        
        if event_col is None:
            pancan_event_cols = [c for c in survival_df.columns if c in ['OS', 'PFI', 'DSS', 'DFI']]
            if pancan_event_cols:
                event_col = pancan_event_cols[0]
            else:
                event_cols = [c for c in survival_df.columns if ('event' in c.lower() or 'status' in c.lower()
                             or 'death' in c.lower() or 'cens' in c.lower() or 'vital' in c.lower()
                             or 'os_event' in c.lower() or 'pfs_event' in c.lower())
                             and 'menopause' not in c.lower() and 'alcohol' not in c.lower()
                             and 'tobacco' not in c.lower() and 'gender' not in c.lower()]
                if event_cols:
                    event_col = event_cols[0]
                else:
                    binary_cols = []
                numeric_cols = survival_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:100]:  
                    if col == time_col:
                        continue
                    unique_vals = survival_df[col].dropna().unique()
                    if len(unique_vals) <= 4:  
                        val_counts = pd.Series(unique_vals).value_counts()
                        if len(val_counts) <= 3:
                            binary_cols.append((col, unique_vals))
                
                if binary_cols:
                    for col, vals in binary_cols:
                        if set(vals).issubset({0, 1, -1, True, False, 2}) or len(vals) == 2:
                            event_col = col
                            break
                
                if event_col is None and len(survival_df.columns) > 1:
                    if time_col != survival_df.columns[1]:
                        event_col = survival_df.columns[1]
                    else:
                        event_col = None
        
        if event_col is None:
            raise ValueError("Could not identify event column. Please specify event_col parameter.")
        
        times = survival_df[time_col].values
        events_raw = survival_df[event_col].values
        
        times = pd.to_numeric(times, errors='coerce').astype(float)
        
        if events_raw.dtype == bool:
            events = events_raw.astype(int)
        elif events_raw.dtype == object:
            unique_strs = pd.Series(events_raw).unique()
            events = np.array([1 if str(v).lower() in ['1', 'true', 'event', 'dead', 'death', 'yes'] 
                             else 0 for v in events_raw])
        else:
            events = pd.to_numeric(events_raw, errors='coerce').astype(float)
            unique_vals = np.unique(events[~np.isnan(events)])
            if len(unique_vals) == 2:
                if -1 in unique_vals or 2 in unique_vals:
                    events = (events == unique_vals.max()).astype(int)
                else:
                    events = events.astype(int)
            elif len(unique_vals) > 2:
                median_val = np.nanmedian(events)
                events = (events > median_val).astype(int)
            else:
                events = (events > 0).astype(int)
                
        valid_mask = np.isfinite(times) & (times >= 0) & np.isfinite(events) & (events >= 0) & (events <= 1)
        
        if np.sum(valid_mask) == 0:
            valid_mask = np.isfinite(times) & np.isfinite(events)
            print("Warning: Using lenient filtering (allowing times=0)")
        
        try:
            if 'normalized_id' in survival_df.columns:
                raw_ids = survival_df['normalized_id'].astype(str).values
            elif 'sample' in survival_df.columns:
                raw_ids = survival_df['sample'].astype(str).values
            elif 'patient' in survival_df.columns:
                raw_ids = survival_df['patient'].astype(str).values
            else:
                raw_ids = survival_df.index.astype(str).values
        except Exception:
            raw_ids = survival_df.index.astype(str).values
        self.last_valid_sample_ids = raw_ids[valid_mask] if len(raw_ids) == len(valid_mask) else None
        
        times = times[valid_mask]
        events = events[valid_mask]
        
        n_events = events.sum()
        n_censored = len(events) - n_events
        
        if n_events == 0:
            raise ValueError("No events found in survival data. Check event column or specify --event_col manually")
        
        if len(times) == 0:
            raise ValueError("No valid survival times found after filtering.")
        
        return times, events
    
    def integrate_omics(self, aligned_omics: Dict[str, pd.DataFrame],
                       max_features_per_omics: int = 1000) -> pd.DataFrame:
        """
        Integrate multiple omics datasets into a single feature matrix;
        each omics type is processed then concatenated
        """
        
        integrated_features = []
        feature_names = []
        
        for omics_type, df in aligned_omics.items():
            if df is None or df.empty:
                continue
            
            exclude_cols = ['normalized_id', 'sample', 'patient', 'gene', 'gene_id']
            feature_cols = [c for c in df.columns 
                           if not any(exc in str(c).lower() for exc in exclude_cols)]
            
            if len(feature_cols) == 0:
                continue
            
            omics_features = df[feature_cols].select_dtypes(include=[np.number])
            
            if omics_features.empty:
                continue
            
            if omics_features.shape[1] > max_features_per_omics:
                variances = omics_features.var(axis=0)
                top_indices = variances.nlargest(max_features_per_omics).index
                omics_features = omics_features[top_indices]
            
            omics_features = omics_features.fillna(omics_features.median())
            
            omics_features = (omics_features - omics_features.mean()) / (omics_features.std() + 1e-8)
            
            omics_features.columns = [f"{omics_type}_{col}" for col in omics_features.columns]
            
            integrated_features.append(omics_features)
            feature_names.extend(omics_features.columns.tolist())
        
        if not integrated_features:
            raise ValueError("No valid omics features found for integration")
        
        integrated_df = pd.concat(integrated_features, axis=1)
        print(f"Integrated feature matrix shape: {integrated_df.shape}")
        print(f"Total features: {len(integrated_df.columns)}")
        
        return integrated_df


class PanCancerOmicsDataLoader(MultiOmicsDataLoader):
    """Loader for TCGA Pan-Cancer unified datasets"""
    
    def __init__(self, data_dir: str = "data", include_cancer_type: bool = True):
        """
        Args:
            data_dir: Directory containing pan-cancer TCGA data
            include_cancer_type: If True, add cancer_type as one-hot features
        """
        super().__init__(data_dir)
        self.include_cancer_type = include_cancer_type
        
    def load_pancan_data(self, 
                        clinical_file: str = None,
                        expression_file: str = None,
                        copy_number_file: str = None,
                        mutation_file: str = None,
                        rppa_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load and integrate TCGA Pan-Cancer data.
        
        Returns:
            Integrated DataFrame with features and survival data
        """
        
        data_files = [f for f in self.data_dir.iterdir() if f.is_file()]
        
        if clinical_file is None:
            clinical_files = [f for f in data_files if f.name.startswith('tcga_pancan_clinical')]
            if not clinical_files:
                clinical_files = [f for f in data_files if 'survival' in f.name.lower() or 
                                'supplemental' in f.name.lower() or 
                                ('clinical' in f.name.lower() and 'pancan' in f.name.lower())]
            if clinical_files:
                clinical_file = clinical_files[0].name
            else:
                raise FileNotFoundError("Could not find clinical/survival file. Look for 'tcga_pancan_clinical' or 'Survival' or 'clinical' in filename.")
        
        clinical_path = self.data_dir / clinical_file
        if not clinical_path.exists():
            raise FileNotFoundError(f"Clinical file not found: {clinical_path}")
        
        try:
            clinical = pd.read_csv(clinical_path, sep='\t', low_memory=False)
        except:
            try:
                clinical = pd.read_csv(clinical_path, low_memory=False)
            except Exception as e:
                try:
                    with open(clinical_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                    if '\t' in first_line:
                        clinical = pd.read_csv(clinical_path, sep='\t', low_memory=False)
                    else:
                        clinical = pd.read_csv(clinical_path, low_memory=False)
                except:
                    raise ValueError(f"Could not read clinical file {clinical_path}. Error: {e}")
        
        sample_col = None
        for col in ['sample', 'Sample', 'patient', 'Patient', 'bcr_patient_barcode']:
            if col in clinical.columns:
                sample_col = col
                break
        
        if sample_col is None:
            clinical.reset_index(inplace=True)
            sample_col = clinical.columns[0]
        
        def normalize_id(sid):
            sid = str(sid).upper()
            if 'TCGA' in sid:
                parts = sid.split('-')
                if len(parts) >= 3:
                    return '-'.join(parts[:3])
            return sid
        
        clinical['normalized_id'] = clinical[sample_col].apply(normalize_id)
        
        cancer_type_col = None
        for col in ['cancer type abbreviation', 'cancer_type', 'cancer.type', 'type', 
                   'project', 'primary_disease', 'sample_type', 'DISEASE', 'acronym',
                   'acronym', 'tumor_type']:
            if col in clinical.columns:
                cancer_type_col = col
                break
        
        if cancer_type_col is None:
            clinical['cancer_type'] = 'UNKNOWN'
            cancer_type_col = 'cancer_type'
        else:
            clinical['cancer_type'] = clinical[cancer_type_col]
        
        if expression_file is None:
            expr_files = [f for f in data_files if f.name.startswith('tcga_pancan_expression')]
            if not expr_files:
                expr_files = [f for f in data_files if 'expression' in f.name.lower() or 
                             ('geneexp' in f.name.lower() and 'pancan' in f.name.lower())]
            if expr_files:
                expression_file = expr_files[0].name
        
        expr = None
        if expression_file:
            expression_path = self.data_dir / expression_file
            if expression_path.exists():
                if expression_path.suffix == '.gz' or '.gz' in expression_path.name:
                    with gzip.open(expression_path, 'rt') as f:
                        expr = pd.read_csv(f, sep='\t', index_col=0, low_memory=False)
                else:
                    try:
                        expr = pd.read_csv(expression_path, sep='\t', index_col=0, low_memory=False)
                    except:
                        expr = pd.read_csv(expression_path, index_col=0, low_memory=False)
                
                
                expr.columns = [normalize_id(col) for col in expr.columns]
                
                common_samples = set(clinical['normalized_id']).intersection(set(expr.columns))
                
                if len(common_samples) > 0:
                    clinical = clinical[clinical['normalized_id'].isin(common_samples)].copy()
                    expr_aligned = expr[list(common_samples)]
                    
                    expr_t = expr_aligned.T
                    expr_t.columns = [f'expr_{col}' for col in expr_t.columns]
                    
                    clinical = clinical.merge(
                        expr_t.reset_index().rename(columns={'index': 'normalized_id'}),
                        on='normalized_id',
                        how='inner'
                    )
        
        cnv = None
        if copy_number_file is None:
            cnv_files = [f for f in data_files if f.name.startswith('tcga_pancan_copy_number')]
            if not cnv_files:
                cnv_files = [f for f in data_files if ('copy' in f.name.lower() and 'number' in f.name.lower()) or 
                            ('gistic' in f.name.lower() and 'pancan' in f.name.lower())]
            if cnv_files:
                copy_number_file = cnv_files[0].name
        
        if copy_number_file:
            cnv_path = self.data_dir / copy_number_file
            if cnv_path.exists():
                if cnv_path.suffix == '.gz' or '.gz' in cnv_path.name:
                    with gzip.open(cnv_path, 'rt') as f:
                        cnv = pd.read_csv(f, sep='\t', index_col=0, low_memory=False)
                else:
                    try:
                        cnv = pd.read_csv(cnv_path, sep='\t', index_col=0, low_memory=False)
                    except:
                        try:
                            cnv = pd.read_csv(cnv_path, index_col=0, low_memory=False)
                        except:
                            with open(cnv_path, 'r', encoding='utf-8') as f:
                                first_line = f.readline()
                            if '\t' in first_line:
                                cnv = pd.read_csv(cnv_path, sep='\t', index_col=0, low_memory=False)
                            else:
                                cnv = pd.read_csv(cnv_path, sep=',', index_col=0, low_memory=False)
                
                cnv.columns = [normalize_id(col) for col in cnv.columns]
                
                common_samples = set(clinical['normalized_id']).intersection(set(cnv.columns))
                
                if len(common_samples) > 0:
                    clinical = clinical[clinical['normalized_id'].isin(common_samples)].copy()
                    cnv_aligned = cnv[list(common_samples)]
                    cnv_t = cnv_aligned.T
                    cnv_t.columns = [f'cnv_{col}' for col in cnv_t.columns]
                    
                    clinical = clinical.merge(
                        cnv_t.reset_index().rename(columns={'index': 'normalized_id'}),
                        on='normalized_id',
                        how='inner'
                    )
        
        mut = None
        if mutation_file is None:
            mut_files = [f for f in data_files if f.name.startswith('tcga_pancan_mutations')]
            if not mut_files:
                mut_files = [f for f in data_files if ('mutation' in f.name.lower() or 'mc3' in f.name.lower() or 
                          'nonsilent' in f.name.lower()) and 'pancan' in f.name.lower()]
            if mut_files:
                mutation_file = mut_files[0].name
        
        if mutation_file:
            mut_path = self.data_dir / mutation_file
            if mut_path.exists():
                if mut_path.suffix == '.gz' or '.gz' in mut_path.name:
                    with gzip.open(mut_path, 'rt') as f:
                        mut = pd.read_csv(f, sep='\t', index_col=0, low_memory=False)
                else:
                    try:
                        mut = pd.read_csv(mut_path, sep='\t', index_col=0, low_memory=False)
                    except:
                        try:
                            mut = pd.read_csv(mut_path, index_col=0, low_memory=False)
                        except:
                            with open(mut_path, 'r', encoding='utf-8') as f:
                                first_line = f.readline()
                            if '\t' in first_line:
                                mut = pd.read_csv(mut_path, sep='\t', index_col=0, low_memory=False)
                            else:
                                mut = pd.read_csv(mut_path, sep=',', index_col=0, low_memory=False)
                
                
                mut = mut.astype(np.float32)
                
                mut.columns = [normalize_id(col) for col in mut.columns]
                
                common_samples = set(clinical['normalized_id']).intersection(set(mut.columns))
                
                if len(common_samples) > 0:
                    clinical = clinical[clinical['normalized_id'].isin(common_samples)].copy()
                    mut_aligned = mut[list(common_samples)]
                    mut_t = mut_aligned.T
                    mut_t.columns = [f'mut_{col}' for col in mut_t.columns]
                    
                    mut_t_reset = mut_t.reset_index().rename(columns={'index': 'normalized_id'})
                    
                    clinical = clinical.merge(
                        mut_t_reset,
                        on='normalized_id',
                        how='inner'
                    )
        
        rppa = None
        if rppa_file is None:
            rppa_files = [f for f in data_files if f.name.startswith('tcga_pancan_rppa')]
            if not rppa_files:
                rppa_files = [f for f in data_files if 'rppa' in f.name.lower() and 'pancan' in f.name.lower()]
            if rppa_files:
                rppa_file = rppa_files[0].name
        
        if rppa_file:
            rppa_path = self.data_dir / rppa_file
            if rppa_path.exists():
                if rppa_path.suffix == '.gz' or '.gz' in rppa_path.name:
                    with gzip.open(rppa_path, 'rt') as f:
                        rppa = pd.read_csv(f, sep='\t', index_col=0, low_memory=False)
                else:
                    try:
                        rppa = pd.read_csv(rppa_path, sep='\t', index_col=0, low_memory=False)
                    except:
                        try:
                            rppa = pd.read_csv(rppa_path, index_col=0, low_memory=False)
                        except:
                            with open(rppa_path, 'r', encoding='utf-8') as f:
                                first_line = f.readline()
                            if '\t' in first_line:
                                rppa = pd.read_csv(rppa_path, sep='\t', index_col=0, low_memory=False)
                            else:
                                rppa = pd.read_csv(rppa_path, sep=',', index_col=0, low_memory=False)
                
                rppa.columns = [normalize_id(col) for col in rppa.columns]
                
                common_samples = set(clinical['normalized_id']).intersection(set(rppa.columns))
                
                if len(common_samples) > 0:
                    clinical = clinical[clinical['normalized_id'].isin(common_samples)].copy()
                    rppa_aligned = rppa[list(common_samples)]
                    rppa_t = rppa_aligned.T
                    rppa_t.columns = [f'rppa_{col}' for col in rppa_t.columns]
                    
                    clinical = clinical.merge(
                        rppa_t.reset_index().rename(columns={'index': 'normalized_id'}),
                        on='normalized_id',
                        how='inner'
                    )
        
        if self.include_cancer_type and 'cancer_type' in clinical.columns:
            cancer_types = clinical['cancer_type']
            one_hot = pd.get_dummies(cancer_types, prefix='cancer_type')
            for col in one_hot.columns:
                clinical[col] = one_hot[col]
        
        self.survival_data = clinical
        return clinical


if __name__ == "__main__":
    loader = PanCancerOmicsDataLoader(data_dir="data", include_cancer_type=True)
    
    try:
        integrated_data = loader.load_pancan_data()
        
        times, events = loader.prepare_survival_outcomes(integrated_data)
        
        print(f"\n✓ Successfully loaded pan-cancer dataset:")
        print(f"  Total samples: {len(integrated_data)}")
        print(f"  Features: {integrated_data.shape[1]} columns")
        print(f"  Events: {events.sum()}, Censored: {(events == 0).sum()}")
        
        if 'cancer_type' in integrated_data.columns:
            print(f"  Cancer types: {integrated_data['cancer_type'].value_counts().sum()} unique")
            print(f"\n  Top cancer types:")
            print(integrated_data['cancer_type'].value_counts().head(10))
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have downloaded the pan-cancer files:")
        print("  1. Curated clinical data → tcga_pancan_clinical.csv")
        print("  2. Batch effects normalized mRNA → tcga_pancan_expression.tsv")
        print("  3. gene-level copy number (gistic2_thresholded) → tcga_pancan_copy_number.tsv")
        print("  4. Gene level non-silent mutation → tcga_pancan_mutations.tsv")
        print("  5. RPPA (optional) → tcga_pancan_rppa.tsv")

