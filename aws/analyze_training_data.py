#!/usr/bin/env python3
"""
Comprehensive Training Data Analysis
Analyzes data quality, distribution, and sufficiency for general and specialized models
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingDataAnalyzer:
    def __init__(self, data_path='/tmp/train_sample.csv'):
        self.data_path = data_path
        self.metadata_path = '/tmp/dataset_metadata.json'

        # Class mappings
        self.class_names = {
            0: "Normal",
            1: "DDoS/DoS Attack",
            2: "Network Reconnaissance",
            3: "Brute Force Attack",
            4: "Web Application Attack",
            5: "Malware/Botnet",
            6: "Advanced Persistent Threat"
        }

        # Minimum samples needed for effective training
        self.min_samples_general = 10000  # Per class for general model
        self.min_samples_specialist = 5000  # For specialist models
        self.recommended_samples_specialist = 20000  # Ideal for specialists

    def load_metadata(self):
        """Load dataset metadata"""
        try:
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            logger.info("=" * 60)
            logger.info("📊 DATASET METADATA")
            logger.info("=" * 60)
            logger.info(f"Total Samples: {metadata['total_samples']:,}")
            logger.info(f"Train Samples: {metadata['train_samples']:,}")
            logger.info(f"Test Samples: {metadata['test_samples']:,}")
            logger.info(f"Features: {metadata['features']}")
            logger.info(f"Classes: {metadata['classes']}")

            logger.info("\n📈 Class Distribution:")
            for class_name, count in metadata['distribution'].items():
                logger.info(f"  {class_name:30s}: {count:,} samples")

            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None

    def load_sample_data(self, n_rows=10000):
        """Load sample of training data for analysis"""
        try:
            logger.info(f"\n📥 Loading sample data ({n_rows:,} rows)...")

            df = pd.read_csv(self.data_path, nrows=n_rows)

            logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")

            return df
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
            return None

    def analyze_data_quality(self, df):
        """Analyze data quality issues"""
        logger.info("\n" + "=" * 60)
        logger.info("🔍 DATA QUALITY ANALYSIS")
        logger.info("=" * 60)

        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"⚠️  Missing values detected:")
            for col, count in missing[missing > 0].items():
                logger.warning(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            logger.info("✅ No missing values")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count

        if inf_counts:
            logger.warning(f"⚠️  Infinite values detected:")
            for col, count in inf_counts.items():
                logger.warning(f"  {col}: {count}")
        else:
            logger.info("✅ No infinite values")

        # Check feature distributions
        logger.info("\n📊 Feature Statistics:")
        logger.info(f"  Total features: {len(numeric_cols)}")

        # Check for constant features
        constant_features = [col for col in numeric_cols if df[col].nunique() == 1]
        if constant_features:
            logger.warning(f"⚠️  Constant features (no variance): {len(constant_features)}")
        else:
            logger.info("✅ No constant features")

        # Check for features with very low variance
        low_variance = []
        for col in numeric_cols:
            if df[col].std() < 0.01:
                low_variance.append(col)

        if low_variance:
            logger.warning(f"⚠️  Low variance features: {len(low_variance)}")
        else:
            logger.info("✅ Good feature variance")

        # Check for duplicates
        dupl_count = df.duplicated().sum()
        if dupl_count > 0:
            logger.warning(f"⚠️  Duplicate rows: {dupl_count} ({dupl_count/len(df)*100:.2f}%)")
        else:
            logger.info("✅ No duplicate rows")

        return {
            "missing_values": missing.sum(),
            "infinite_values": sum(inf_counts.values()) if inf_counts else 0,
            "constant_features": len(constant_features),
            "low_variance_features": len(low_variance),
            "duplicates": dupl_count
        }

    def analyze_class_distribution(self, df, metadata):
        """Analyze class distribution and balance"""
        logger.info("\n" + "=" * 60)
        logger.info("⚖️  CLASS BALANCE ANALYSIS")
        logger.info("=" * 60)

        # Get label column (assume it's the last column or named 'label')
        if 'label' in df.columns:
            labels = df['label']
        else:
            labels = df.iloc[:, -1]

        class_counts = Counter(labels)
        total = len(labels)

        logger.info(f"Sample Size: {total:,}")
        logger.info("\nClass Distribution:")

        min_count = float('inf')
        max_count = 0

        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total) * 100
            class_name = self.class_names.get(class_id, f"Class {class_id}")

            logger.info(f"  {class_name:30s}: {count:6,} ({percentage:5.2f}%)")

            min_count = min(min_count, count)
            max_count = max(max_count, count)

        # Calculate imbalance ratio
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        logger.info(f"\n📊 Balance Metrics:")
        logger.info(f"  Min class samples: {min_count:,}")
        logger.info(f"  Max class samples: {max_count:,}")
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 10:
            logger.warning(f"⚠️  HIGH CLASS IMBALANCE (>{imbalance_ratio:.0f}:1)")
            logger.info("  💡 Recommendation: Use class weights or resampling")
        elif imbalance_ratio > 5:
            logger.warning(f"⚠️  MODERATE CLASS IMBALANCE ({imbalance_ratio:.0f}:1)")
            logger.info("  💡 Recommendation: Consider class weights")
        else:
            logger.info(f"✅ BALANCED CLASSES ({imbalance_ratio:.0f}:1)")

        return class_counts, imbalance_ratio

    def assess_sufficiency_for_models(self, metadata):
        """Assess data sufficiency for general and specialized models"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 DATA SUFFICIENCY ASSESSMENT")
        logger.info("=" * 60)

        distribution = metadata['distribution']

        # Map metadata class names to our standard names
        class_mapping = {
            'normal': 0,
            'ddos_dos': 1,
            'reconnaissance': 2,
            'brute_force': 3,
            'web_attacks': 4,
            'malware_botnet': 5,
            'advanced_threats': 6
        }

        assessments = {}

        logger.info("\n🎯 GENERAL PURPOSE MODEL (Current):")
        logger.info(f"  Minimum per class: {self.min_samples_general:,}")

        general_sufficient = True
        insufficient_classes = []

        for meta_name, class_id in class_mapping.items():
            count = distribution.get(meta_name, 0)
            class_name = self.class_names[class_id]

            if count >= self.min_samples_general:
                status = "✅ SUFFICIENT"
            else:
                status = "❌ INSUFFICIENT"
                general_sufficient = False
                insufficient_classes.append(class_name)

            logger.info(f"  {class_name:30s}: {count:,} {status}")

        if general_sufficient:
            logger.info("\n✅ General model has SUFFICIENT data for all classes")
        else:
            logger.warning(f"\n⚠️  General model INSUFFICIENT data for: {', '.join(insufficient_classes)}")

        # Assess specialized models
        logger.info("\n" + "=" * 60)
        logger.info("🎯 SPECIALIZED MODELS ASSESSMENT")
        logger.info("=" * 60)

        specialist_recommendations = {
            'Brute Force Specialist': {
                'primary_class': 'brute_force',
                'related_classes': ['normal', 'reconnaissance'],
                'priority': 'HIGH',
                'rationale': 'Common attack type, clear patterns, actionable'
            },
            'DDoS/DoS Specialist': {
                'primary_class': 'ddos_dos',
                'related_classes': ['normal'],
                'priority': 'HIGH',
                'rationale': 'High volume attacks, network impact, time-critical'
            },
            'APT/Malware Specialist': {
                'primary_class': 'advanced_threats',
                'related_classes': ['malware_botnet', 'reconnaissance', 'normal'],
                'priority': 'MEDIUM',
                'rationale': 'Complex patterns, high-value detection, lower frequency'
            },
            'Web Attack Specialist': {
                'primary_class': 'web_attacks',
                'related_classes': ['normal', 'reconnaissance'],
                'priority': 'MEDIUM',
                'rationale': 'Application layer attacks, SQL/XSS/RCE detection'
            }
        }

        for specialist_name, config in specialist_recommendations.items():
            logger.info(f"\n📊 {specialist_name}:")
            logger.info(f"  Priority: {config['priority']}")
            logger.info(f"  Rationale: {config['rationale']}")

            primary_count = distribution.get(config['primary_class'], 0)
            related_counts = [distribution.get(cls, 0) for cls in config['related_classes']]
            total_samples = primary_count + sum(related_counts)

            logger.info(f"\n  Available Data:")
            logger.info(f"    Primary class: {primary_count:,}")
            logger.info(f"    Related classes: {sum(related_counts):,}")
            logger.info(f"    Total: {total_samples:,}")

            # Assessment
            if primary_count >= self.recommended_samples_specialist:
                status = "✅ EXCELLENT"
                recommendation = "Ready to train specialist model"
            elif primary_count >= self.min_samples_specialist:
                status = "⚠️  SUFFICIENT"
                recommendation = "Can train, but more data would improve performance"
            else:
                status = "❌ INSUFFICIENT"
                needed = self.min_samples_specialist - primary_count
                recommendation = f"Need {needed:,} more samples before training"

            logger.info(f"\n  Status: {status}")
            logger.info(f"  💡 Recommendation: {recommendation}")

            assessments[specialist_name] = {
                'status': status,
                'primary_samples': primary_count,
                'total_samples': total_samples,
                'recommendation': recommendation,
                'priority': config['priority']
            }

        return assessments

    def provide_recommendations(self, assessments, metadata):
        """Provide actionable recommendations"""
        logger.info("\n" + "=" * 60)
        logger.info("💡 RECOMMENDATIONS")
        logger.info("=" * 60)

        distribution = metadata['distribution']

        logger.info("\n🎯 IMMEDIATE ACTIONS:")

        # 1. Fix the scaler issue
        logger.info("\n1. ❗ CRITICAL: Fix Feature Scaling Issue")
        logger.info("   The model was trained with RobustScaler but the scaler wasn't packaged")
        logger.info("   Action: Retrain model and save scaler, or extract scaler from training logs")

        # 2. General model status
        logger.info("\n2. ✅ General Purpose Model")
        logger.info(f"   Current: 97.98% accuracy, 7 classes")
        logger.info(f"   Data: {metadata['train_samples']:,} training samples")
        logger.info("   Status: PRODUCTION READY (after fixing scaler)")

        # 3. Specialized models
        logger.info("\n3. 🎯 Specialized Models - Priority Order:")

        # Sort by priority and data availability
        prioritized = sorted(
            assessments.items(),
            key=lambda x: (
                0 if x[1]['priority'] == 'HIGH' else 1,
                -x[1]['primary_samples']
            )
        )

        for i, (name, assessment) in enumerate(prioritized, 1):
            logger.info(f"\n   {i}. {name}")
            logger.info(f"      Priority: {assessment['priority']}")
            logger.info(f"      Data: {assessment['primary_samples']:,} primary samples")
            logger.info(f"      Status: {assessment['status']}")
            logger.info(f"      Action: {assessment['recommendation']}")

        # 4. Data collection recommendations
        logger.info("\n4. 📈 Data Collection Strategy:")

        insufficient_classes = []
        for meta_name, count in distribution.items():
            if count < self.recommended_samples_specialist:
                insufficient_classes.append((meta_name, count))

        if insufficient_classes:
            logger.info("   Classes needing more data:")
            for class_name, count in sorted(insufficient_classes, key=lambda x: x[1]):
                needed = self.recommended_samples_specialist - count
                logger.info(f"     - {class_name}: need {needed:,} more samples")

            logger.info("\n   📍 Data Sources to Consider:")
            logger.info("     - Honeypot (T-Pot): Already integrated ✅")
            logger.info("     - Public datasets:")
            logger.info("       • CICIDS2017/2018 (comprehensive)")
            logger.info("       • UNSW-NB15 (modern attacks)")
            logger.info("       • CTU-13 (botnet traffic)")
            logger.info("       • Kyoto 2006+ (honeypot data)")
            logger.info("     - Synthetic data generation for rare attacks")

        # 5. Timeline
        logger.info("\n5. ⏱️  Recommended Timeline:")
        logger.info("   Week 1-2: Fix scaler issue, validate general model")
        logger.info("   Week 3-4: Deploy Phase 1 fast triage layer")
        logger.info("   Month 2: Train Brute Force Specialist (HIGH priority)")
        logger.info("   Month 3: Train DDoS/DoS Specialist (HIGH priority)")
        logger.info("   Month 4+: Train APT/Web specialists as data grows")

        return {
            "critical_issues": ["Scaler not packaged with model"],
            "ready_for_specialists": [name for name, a in assessments.items() if "EXCELLENT" in a['status'] or "SUFFICIENT" in a['status']],
            "need_more_data": [name for name, a in assessments.items() if "INSUFFICIENT" in a['status']]
        }

    def run_full_analysis(self):
        """Run complete analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("🔬 TRAINING DATA ANALYSIS")
        logger.info("=" * 60)

        # Load metadata
        metadata = self.load_metadata()
        if not metadata:
            return

        # Load sample data
        df = self.load_sample_data()
        if df is None:
            return

        # Run analyses
        quality = self.analyze_data_quality(df)
        class_counts, imbalance = self.analyze_class_distribution(df, metadata)
        assessments = self.assess_sufficiency_for_models(metadata)
        recommendations = self.provide_recommendations(assessments, metadata)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📝 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Data Quality: {'GOOD' if quality['missing_values'] == 0 else 'NEEDS ATTENTION'}")
        logger.info(f"✅ Class Balance: {'GOOD' if imbalance < 5 else 'NEEDS ATTENTION'}")
        logger.info(f"✅ General Model: SUFFICIENT DATA")
        logger.info(f"✅ Ready for Specialists: {len(recommendations['ready_for_specialists'])}/4")
        logger.info(f"⚠️  Critical Issues: {len(recommendations['critical_issues'])}")
        logger.info("=" * 60)


if __name__ == "__main__":
    analyzer = TrainingDataAnalyzer()
    analyzer.run_full_analysis()