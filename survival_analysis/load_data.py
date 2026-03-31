import ahocorasick
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import yaml
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, VarianceThreshold


def load_config(config_path="config.yaml"):
    """Load configuration file"""
    if not os.path.exists(config_path):
        config = {
            'paths': {
                'clinical_data': 'path/to/clinc_info.xlsx',
                'data_dirs': [
                    'path/to/images',
                ],
                'save_path': 'path/to/merged_features.csv',
                'feature_dir': 'features',
            }
        }
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    
    return config

# ----------------------
# Core data processing module
# ----------------------
class ClinicalDataMatcher:
    """Clinical data matcher (optimized with AC automaton)"""
    def __init__(self, clinical_data):
        self.automaton = ahocorasick.Automaton()
        self.clinical_data = clinical_data
        self._build_automaton()
    
    def _build_automaton(self):
        """Build AC automaton"""
        for idx, row in self.clinical_data.iterrows():
            # Hospital ID matching
            hospital_id = str(row['住院号']).strip()
            if hospital_id and hospital_id not in self.automaton:
                self.automaton.add_word(hospital_id, (idx, 'hospital_id'))
            
            # Name matching (remove spaces and special characters)
            name = str(row['姓名']).strip().replace(' ', '')
            if name and name not in self.automaton:
                self.automaton.add_word(name, (idx, 'name'))
        
        self.automaton.make_automaton()
    
    def match(self, image_name):
        """Execute matching"""
        matches = set()
        for _, (idx, key_type) in self.automaton.iter(image_name):
            matches.add(idx)
        
        if matches:
            return self.clinical_data.iloc[list(matches)]
        return pd.DataFrame()

# ----------------------
# Data loading and preprocessing
# ----------------------
def load_and_merge_data(config, force_reload=False):
    """Data loading and merging pipeline"""
    
    if not force_reload and os.path.exists(config['paths']['save_path']):
        # print("Loading merged data...")
        return pd.read_csv(config['paths']['save_path'])
    
    # Load clinical data
    clinical_data = pd.read_excel(
        config['paths']['clinical_data'],
        dtype={'MPR': str, '住院号': str}
    )
    
    # Load feature data
    features_dfs = []
    for data_dir in config['paths']['data_dirs']:
        features_dir = os.path.join(data_dir, config['paths']['feature_dir'])
        if not os.path.exists(features_dir):
            # logger.warning(f"Feature directory does not exist: {features_dir}")
            continue
        
        # Parallel read CSV files
        with ProcessPoolExecutor(max_workers=32) as executor:
            csv_files = [
                os.path.join(features_dir, f) 
                for f in os.listdir(features_dir) 
                if f.endswith('.csv')
            ]
            results = executor.map(pd.read_csv, csv_files)
        
        # Add data source identifier
        for df in results:
            df['data_source'] = os.path.basename(data_dir)
            features_dfs.append(df)
    
    features = pd.concat(features_dfs, ignore_index=True)

    feature_cols = [
        col for col in features.columns
        if col not in ['image_name', 'data_source']
    ]

    # Execute data merging
    matcher = ClinicalDataMatcher(clinical_data)
    merged_rows = []
    
    for _, row in features.iterrows():
        matched = matcher.match(row['image_name'])
        if not matched.empty:
            clinical_info = matched.iloc[0].add_prefix('clinical_')
            merged_rows.append(pd.concat([row, clinical_info]))
    
    merged_data = pd.DataFrame(merged_rows)
    
    # Save results
    merged_data.to_csv(config['paths']['save_path'], index=False)
    # print(f"Merged data saved to: {config['paths']['save_path']}")
    return merged_data, feature_cols


def process_data_source(data, feature_cols, file_path, prefix, source_name, 
                        id_column=None, id_processor=None, 
                        file_type='excel', sep='\t'):
    """Unified function to process data sources"""
    if not os.path.exists(file_path):
        # logger.warning(f"{source_name} data file not found: {file_path}")
        return data, feature_cols, False
    
    # print(f"Loading {source_name} data file: {file_path}")
    
    df = pd.read_csv(file_path, sep=sep)
    
    # Handle special characters
    df.replace({'-': np.nan, '/': np.nan}, inplace=True)
    
    # Handle ID column: if no hospital ID column, create one
    if 'clinical_住院号' not in df.columns:
        if id_column and id_column in df.columns:
            # Delete rows containing Pre in id_column
            df = df[~df[id_column].str.contains('Pre', na=False)]
            if id_processor:
                # Use custom processing function to handle ID
                df['clinical_住院号'] = df[id_column].apply(id_processor)
            else:
                # Default processing: take first part as hospital ID
                df['clinical_住院号'] = df[id_column].str.split('_').str[0]
            df = df.drop(columns=[id_column])
        else:
            # No ID column specified, use first column
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'clinical_住院号'})
    
    # Clean unnecessary columns
    cols_to_drop = []
    for col in df.columns:
        # Delete clinical_ prefixed columns except hospital ID
        if col.startswith('clinical_') and col != 'clinical_住院号':
            cols_to_drop.append(col)
        # Delete OS related columns
        elif col in ['OS_status', 'OS_time', 'OS_Group', 'IO_weight_signature']:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Add prefix to non-hospital ID columns
    non_id_cols = [col for col in df.columns if col != 'clinical_住院号']
    df = df.rename(columns={col: f"{prefix}_{col}" for col in non_id_cols})
    
    # Automatically convert numeric columns
    num_cols = [col for col in df.columns if col != 'clinical_住院号']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    
    # Merge to main data
    merged_data = data.merge(df, on='clinical_住院号', how='left')

    # When mutation_spectrum, set num_cols to 0 for rows where 'TMB' is not null
    if 'mutation_spectrum' in source_name:
        merged_data.loc[merged_data[f"{prefix}_TMB"].notnull(), num_cols] = 0
    
    # Update feature columns
    new_feature_cols = feature_cols + [col for col in df.columns if col != 'clinical_住院号']
    
    return merged_data, new_feature_cols, True

def load_data(merged_file='merged_features.csv',
              wes_info_data='WES_gene_signatures_all_sizes.csv',
              mutation_spectrum_data='mutation_spectrum.xls',
              tcr_diversity_data='TCR_diversity.xls',
              modal='clinical',
              ):
    """Load merged feature data"""
    
    if os.path.exists(merged_file):
        # print(f"Loading merged data file: {merged_file}")
        data = pd.read_csv(merged_file)
        feature_cols = [
            col for col in data.columns
            if col not in ['image_name', 'data_source'] and not col.startswith('clinical_') #  and re.search(r'(type_4|ratio_4_to|to_4|nn_4|nc_4)', col)
        ]
        # print(f"Pathological features have {data.shape[0]} records")
    else:
        # If no merged file, call data loading function from analysis_feature.py
        # print("No merged data found, starting to reload and merge...")
        config = load_config()
        data, feature_cols = load_and_merge_data(config, force_reload=True)

    use_all = 'all' in modal or 'muti_modal' in modal
    
    if 'path' not in modal and not use_all:
        feature_cols = []
    
    if 'clinical' in modal or use_all:
        data.rename(columns={'clinical_性别': 'clinical_gender',
                             'clinical_年龄': 'clinical_age',
                             'clinical_肿瘤分期': 'clinical_tumor_stage',
                             'clinical_新辅助治疗周期': 'clinical_new_treatment_period',
                             'clinical_pCR': 'clinical_pCR',
                             'clinical_MPR': 'clinical_MPR'}, inplace=True)
        feature_cols.extend([
            'clinical_gender',
            'clinical_age',
            'clinical_tumor_stage',
            'clinical_new_treatment_period',
            'clinical_pCR',
            'clinical_MPR'
        ])
    
    # Process WES info data
    if 'wes' in modal or use_all:
        data, feature_cols, success = process_data_source(
            data=data,
            feature_cols=feature_cols,
            file_path=wes_info_data,
            prefix='wes_info',
            source_name='WES info',
            sep=','
        )
    
        # Process mutation spectrum data
        data, feature_cols, success = process_data_source(
            data=data,
            feature_cols=feature_cols,
            file_path=mutation_spectrum_data,
            prefix='mutation_spectrum',
            source_name='Mutation spectrum',
            id_column='ID'
        )
    
    # Process TCR diversity data
    if 'tcr' in modal or use_all:
        data, feature_cols, success = process_data_source(
            data=data,
            feature_cols=feature_cols,
            file_path=tcr_diversity_data,
            prefix='tcr_diversity',
            source_name='TCR diversity'
        )
    
    return data, feature_cols

def filter_data(data, feature_cols):
    # Filter valid samples
    data = data.rename(columns={'clinical_status': 'status', 'clinical_die_days': 'time'})
    data = data.dropna(subset=['status', 'time'])
    data['status'] = data['status'].astype(bool)
    data['time'] = data['time'].astype(float) / 30.0  # Convert to months
    # data = data.dropna(subset=feature_cols)

    data = data[(data['clinical_术前新辅助治疗'] == '直接手术') | (((data['clinical_术前新辅助治疗'] == '免疫化疗') | (data['clinical_术前新辅助治疗'] == '单纯化疗')) & (data['data_source'] != '治疗前SVS'))]
    # data = data[data['data_source'] != '治疗前SVS']

    print(f"After filtering valid samples, total {data.shape[0]} records")
    # Duplicate clinical_住院号 rows in data[(data['clinical_术前新辅助治疗'] == '免疫化疗')& (data['data_source'] != '治疗前SVS')]['clinical_住院号']
    duplicated_hipnos = data[(data['clinical_术前新辅助治疗'] == '免疫化疗') & (data['data_source'] != '治疗前SVS')]['clinical_住院号'].duplicated()
    print(data[(data['clinical_术前新辅助治疗'] == '免疫化疗') & (data['data_source'] != '治疗前SVS')][duplicated_hipnos])
    print(f"Direct surgery has {data[data['clinical_术前新辅助治疗'] == '直接手术'].shape[0]} records")
    print(f"Immunotherapy has {data[data['clinical_术前新辅助治疗'] == '免疫化疗'].shape[0]} records")
    print(f"Chemotherapy alone has {data[data['clinical_术前新辅助治疗'] == '单纯化疗'].shape[0]} records")
    print(f"Duplicated clinical_住院号: {data[data.duplicated(subset='clinical_住院号', keep=False)]['clinical_住院号'].unique()}")
    
    data = data[feature_cols + ['status', 'time']]
    
    return data

def preprocess_data(X_train, X_val):
    """
    Preprocess training and validation set data, including missing value handling, variance filtering, and one-hot encoding.

    Args:
    X_train (DataFrame): Training set feature data
    X_val (DataFrame): Validation set feature data

    Returns:
    tuple: Preprocessed training and validation set feature data
    """

    # Separate numerical and categorical features
    numerical_features = X_train.select_dtypes(include=[np.number]).columns
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns

    # Numerical feature processing pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        # ('variance_threshold', VarianceThreshold(threshold=1e-10)),
    ])

    # Categorical feature processing pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        # ('variance_threshold', VarianceThreshold(threshold=1e-10))
    ])

    # Combine numerical and categorical feature processing pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Fit training data and transform training and validation sets
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_val_encoded = preprocessor.transform(X_val)

    # Get processed feature names
    # numerical_transformed_features = preprocessor.named_transformers_['num'].named_steps['variance_threshold'].get_feature_names_out(numerical_features)
    numerical_transformed_features = numerical_features
    if categorical_features.size > 0:
        # categorical_transformed_features = preprocessor.named_transformers_['cat'].named_steps['variance_threshold'].get_feature_names_out(
        #     preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        # )
        categorical_transformed_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_features = np.concatenate([numerical_transformed_features, categorical_transformed_features])
    else:
        all_features = numerical_transformed_features

    # Convert to DataFrame
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=all_features)
    X_val_encoded = pd.DataFrame(X_val_encoded, columns=all_features)

    # Check for null values
    assert X_train_encoded.isnull().sum().sum() == 0, "X_train_encoded has null values"
    assert X_val_encoded.isnull().sum().sum() == 0, "X_val_encoded has null values"
    
    return X_train_encoded, X_val_encoded


if __name__ == '__main__':
    load_and_merge_data(load_config(), force_reload=True) 