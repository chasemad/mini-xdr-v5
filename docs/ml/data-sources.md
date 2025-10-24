# Data Sources

## Bundled Datasets

- **CICIDS2017 Dataset**: `datasets/cicids2017_official/MachineLearningCSV.zip` - Official network intrusion detection dataset with 8+ million records across 15 attack types.
- **Real Cybersecurity Datasets**: `datasets/real_datasets/*.json` - Curated samples including KDD Cup 1999, honeypot logs, URLhaus malware feeds, and threat intelligence data.
- **Synthetic Attack Scenarios**: `datasets/*.json` - Generated attack patterns (brute force, DDoS, web attacks, lateral movement) for model augmentation and testing.
- **Windows Event Datasets**: `datasets/windows_*/` - Windows security event logs and AD datasets for specialized model training.
- **Threat Feed Datasets**: `datasets/threat_feeds/` - Real-time threat intelligence feeds integrated into training pipelines.

## Ingested Events & Training Data Collection

- **Real-time Event Ingestion**: Events via `/ingest/multi` stored in database with automatic feature extraction for continuous learning.
- **Training Data Collector**: `backend/app/training_data_collector.py` automatically captures and labels incident data for model improvement.
- **Online Learning Data**: Continuous data streaming for online learning adaptation without full retraining.
- **Federated Data Sources**: Privacy-preserving data aggregation from distributed tenants for collaborative model training.

## Feature Engineering

- **Base Feature Extraction**: `BaseMLDetector._extract_features` in `backend/app/ml_engine.py` handles core feature engineering.
- **Enhanced Pipeline**: `backend/app/enhanced_training_pipeline.py` includes advanced feature engineering with temporal features, behavioral patterns, and correlation analysis.
- **Dynamic Features**: Runtime feature extraction adapts to new data sources and threat patterns.
- **Feature Store**: Centralized feature storage and versioning for consistent model training and inference.
- **Custom Features**: Extensible feature engineering framework for domain-specific threat detection.

## Data Governance & Compliance

- **Dataset Provenance**: All datasets tracked with source attribution, collection date, and licensing information.
- **Data Quality**: Automated validation and cleaning pipelines ensure high-quality training data.
- **Privacy Compliance**: Data anonymization and aggregation techniques protect sensitive information.
- **Retention Policies**: Configurable data retention with automated cleanup for compliance requirements.
- **Audit Trails**: Complete audit logging of data usage, model training, and performance metrics.
- **Storage Optimization**: Large datasets stored in Git LFS or external artefact storage to manage repository size.
- **Access Control**: Role-based access to training data and model artefacts based on organizational policies.

Update this document when new datasets or data sources are introduced, ensuring compliance with data governance policies.
