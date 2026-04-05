"""
Generate synthetic social scoring dataset with intentional demographic variations.
Focus: Enable governance testing over realistic modeling.
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)


def generate_social_scoring_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic loan application data with protected attributes.
    Intentionally includes potential bias patterns for governance testing.
    """
    logger.info(f"Generating {n_samples} synthetic loan applications...")
    
    data = []
    
    for i in range(n_samples):
        # Protected attributes (for bias detection)
        gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
        age = np.random.randint(18, 75)
        age_group = (
            'Young Adult (18-25)' if age <= 25 else
            'Adult (26-40)' if age <= 40 else
            'Middle Age (41-60)' if age <= 60 else
            'Senior (60+)'
        )
        
        ethnicity = np.random.choice(
            ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'],
            p=[0.60, 0.13, 0.18, 0.06, 0.03]
        )
        
        # Socioeconomic features
        education_level = np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            p=[0.30, 0.45, 0.20, 0.05]
        )
        
        employment_status = np.random.choice(
            ['Employed Full-Time', 'Employed Part-Time', 'Self-Employed', 'Unemployed'],
            p=[0.65, 0.15, 0.12, 0.08]
        )
        
        # Financial features
        # Introduce subtle bias: income correlates with demographics
        base_income = 45000
        
        # Gender pay gap (intentional bias for detection)
        if gender == 'Male':
            income_multiplier = np.random.uniform(1.0, 1.15)
        else:
            income_multiplier = np.random.uniform(0.85, 1.0)
        
        # Education impact
        edu_multipliers = {'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.5}
        income_multiplier *= edu_multipliers[education_level]
        
        # Age impact
        if age < 25:
            income_multiplier *= 0.6
        elif age > 60:
            income_multiplier *= 0.9
        
        annual_income = int(base_income * income_multiplier * np.random.uniform(0.8, 1.3))
        
        # Credit history
        years_credit_history = min(age - 18, np.random.randint(0, 30))
        num_credit_cards = np.random.randint(0, 8)
        num_bank_accounts = np.random.randint(1, 5)
        
        # Debt and loans
        existing_debt = int(annual_income * np.random.uniform(0, 2.5))
        previous_loans = np.random.randint(0, 10)
        
        # Payment behavior
        payment_history_score = np.random.randint(300, 850)
        missed_payments_last_year = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.5, 0.25, 0.15, 0.06, 0.03, 0.01])
        
        # Loan application details
        loan_amount_requested = int(np.random.uniform(5000, 100000))
        loan_purpose = np.random.choice([
            'Home Purchase', 'Car Purchase', 'Education', 'Business', 
            'Debt Consolidation', 'Personal'
        ])
        
        # Additional risk factors
        debt_to_income_ratio = existing_debt / annual_income if annual_income > 0 else 999
        employment_length_years = np.random.randint(0, max(1, min(age - 18, 30)))
        has_savings = np.random.choice([True, False], p=[0.6, 0.4])
        savings_amount = int(annual_income * np.random.uniform(0, 1.5)) if has_savings else 0
        
        # Target variable: Loan approval (with bias)
        # Calculate base approval probability
        approval_prob = 0.5
        
        # Positive factors
        if payment_history_score > 700:
            approval_prob += 0.2
        if debt_to_income_ratio < 0.4:
            approval_prob += 0.15
        if annual_income > 60000:
            approval_prob += 0.1
        if education_level in ['Master', 'PhD']:
            approval_prob += 0.05
        if missed_payments_last_year == 0:
            approval_prob += 0.1
            
        # Negative factors
        if payment_history_score < 600:
            approval_prob -= 0.25
        if debt_to_income_ratio > 1.0:
            approval_prob -= 0.2
        if employment_status == 'Unemployed':
            approval_prob -= 0.3
        if missed_payments_last_year > 2:
            approval_prob -= 0.15
            
        # INTENTIONAL BIAS for detection (should be flagged by governance)
        # Age bias
        if age < 25:
            approval_prob -= 0.12
        elif age > 65:
            approval_prob -= 0.08
            
        # Gender bias (subtle)
        if gender == 'Female':
            approval_prob -= 0.05
            
        # Ethnicity bias (this should be detected!)
        if ethnicity in ['African American', 'Hispanic']:
            approval_prob -= 0.10
        
        # Clip probability
        approval_prob = max(0.05, min(0.95, approval_prob))
        
        # Generate approval decision
        approved = np.random.random() < approval_prob
        
        # Create record
        record = {
            # Identifiers
            'application_id': str(uuid.uuid4()),
            'application_date': fake.date_time_between(start_date='-2y', end_date='now'),
            
            # Protected attributes
            'gender': gender,
            'age': age,
            'age_group': age_group,
            'ethnicity': ethnicity,
            
            # Socioeconomic
            'education_level': education_level,
            'employment_status': employment_status,
            'employment_length_years': employment_length_years,
            'annual_income': annual_income,
            
            # Credit profile
            'years_credit_history': years_credit_history,
            'payment_history_score': payment_history_score,
            'num_credit_cards': num_credit_cards,
            'num_bank_accounts': num_bank_accounts,
            'existing_debt': existing_debt,
            'previous_loans': previous_loans,
            'missed_payments_last_year': missed_payments_last_year,
            
            # Financial metrics
            'debt_to_income_ratio': round(debt_to_income_ratio, 3),
            'has_savings': has_savings,
            'savings_amount': savings_amount,
            
            # Loan details
            'loan_amount_requested': loan_amount_requested,
            'loan_purpose': loan_purpose,
            
            # Target
            'approved': int(approved),
            'approval_probability': round(approval_prob, 4)
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} records")
    logger.info(f"Approval rate: {df['approved'].mean():.2%}")
    logger.info(f"Demographics distribution:")
    logger.info(f"  Gender: {df['gender'].value_counts().to_dict()}")
    logger.info(f"  Ethnicity: {df['ethnicity'].value_counts().to_dict()}")
    
    return df


def create_data_quality_report(df: pd.DataFrame) -> dict:
    """Generate data quality metrics for governance."""
    return {
        'total_records': len(df),
        'completeness': {
            col: (1 - df[col].isna().sum() / len(df)) 
            for col in df.columns
        },
        'unique_values': {
            col: df[col].nunique() 
            for col in df.columns
        },
        'numeric_ranges': {
            col: {'min': float(df[col].min()), 'max': float(df[col].max()), 'mean': float(df[col].mean())}
            for col in df.select_dtypes(include=[np.number]).columns
        },
        'categorical_distributions': {
            col: df[col].value_counts().to_dict()
            for col in df.select_dtypes(include=['object']).columns
            if df[col].nunique() < 20  # Only for low-cardinality categoricals
        }
    }


def save_dataset(df: pd.DataFrame, output_path: str = 'data/synthetic'):
    """Save dataset with versioning."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save main dataset
    csv_path = f"{output_path}/social_scoring_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset to {csv_path}")
    
    # Save latest version
    latest_path = f"{output_path}/social_scoring_latest.csv"
    df.to_csv(latest_path, index=False)
    logger.info(f"Saved latest version to {latest_path}")
    
    # Save data quality report
    quality_report = create_data_quality_report(df)
    import json
    report_path = f"{output_path}/data_quality_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Saved quality report to {report_path}")


def main():
    """Generate and save synthetic dataset."""
    logger.info("=" * 60)
    logger.info("Social Scoring Dataset Generator")
    logger.info("=" * 60)
    
    # Generate data
    df = generate_social_scoring_data(n_samples=10000)
    
    # Save dataset
    save_dataset(df)
    
    logger.info("\n" + "=" * 60)
    logger.info("Dataset generation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
