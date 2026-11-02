import pandas as pd
import numpy as np

class DataQualityAnalyzer:
    """Comprehensive data quality assessment framework"""
    
    def __init__(self, df):
        self.df = df
        self.quality_report = {}
        
    def assess_completeness(self):
        """Analyze missing values and data completeness"""
        completeness = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'completeness_rate': (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        }
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        completeness['missing_by_column'] = pd.DataFrame({
            'missing_count': missing_data,
            'missing_percentage': missing_percent
        }).sort_values('missing_percentage', ascending=False)
        
        return completeness
    
    def assess_dataTypes(self):
        return self.df.dtypes
    
    def assess_duplicates(self):
        """Analyze duplicate rows in the dataset"""
        
        duplicate_mask = self.df.duplicated()
        
        duplicate_report = {
            'total_duplicates': duplicate_mask.sum(),
            'duplicate_percentage': (duplicate_mask.sum() / len(self.df)) * 100 if len(self.df) > 0 else 0,
            'has_duplicates': duplicate_mask.any(),
            'duplicate_rows': self.df[duplicate_mask]
        }
        
        return duplicate_report
    
    def detect_outliers(self, numerical_columns=None):
        """Identify outliers using IQR method"""
        if numerical_columns is None:
            numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_report = {}
        
        for col in numerical_columns:
            if col not in self.df.columns:
                continue
                
            if self.df[col].dtype not in [np.int64, np.float64]:
                continue
                
            if self.df[col].notna().sum() == 0:
                outlier_report[col] = {'error': 'Column contains only NaN values'}
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                
                if std_val == 0:
                    outlier_report[col] = {'error': 'Constant column, no variability'}
                    continue
                    
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
            else:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            outlier_report[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(self.df)) * 100 if len(self.df) > 0 else 0,
                'min_value': self.df[col].min(),
                'max_value': self.df[col].max(),
                'has_issues': (len(outliers) / len(self.df)) * 100 > 5 if len(self.df) > 0 else False
            }
        
        return outlier_report
    
    def assess_categorical_consistency(self, categorical_columns=None):
        """Check categorical variables for consistency"""
        if categorical_columns is None:
            categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        categorical_report = {}
        
        for col in categorical_columns:
            value_counts = self.df[col].value_counts()
            
            categorical_report[col] = {
                'unique_values': len(value_counts),
                'value_distribution': value_counts.head(10),
            }
        
        return categorical_report

    def generate_quality_report(self):
        """Generate comprehensive data quality report"""
        
        completeness = self.assess_completeness()
        dTypes = self.assess_dataTypes()
        duplicates = self.assess_duplicates()
        outliers = self.detect_outliers()
        categorical_consistency = self.assess_categorical_consistency()
        
        self.quality_report = {
            'completeness': completeness,
            'Data Types': dTypes,
            'duplicates': duplicates,
            'outliers': outliers,
            'categorical_consistency': categorical_consistency,
        }
        
        return self.quality_report