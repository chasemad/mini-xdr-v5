"""
Unified Dataset Loaders for Revolutionary XDR Training

Loads and normalizes data from multiple cybersecurity datasets:
- CICIDS2017 (~2.8M events)
- KDD Cup 1999 (~4.9M events)
- UNSW-NB15 (~2.5M events)
- APT29 Evaluation (~50K events)
- Honeypot logs (variable)

Total: 10M+ events for comprehensive training
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset."""

    name: str
    num_samples: int
    num_features: int
    num_classes: int
    class_distribution: Dict[int, int]
    source_path: str


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders."""

    # Unified threat class mapping (7 classes)
    UNIFIED_CLASS_MAPPING = {
        0: "Normal",
        1: "DDoS",
        2: "Reconnaissance",
        3: "Brute Force",
        4: "Web Attack",
        5: "Malware",
        6: "APT",
    }

    # Standard 79 features for Mini-XDR
    NUM_FEATURES = 79
    NUM_CLASSES = 7

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.info: Optional[DatasetInfo] = None

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset and return (features, labels)."""
        pass

    @abstractmethod
    def _map_labels_to_unified(self, labels: np.ndarray) -> np.ndarray:
        """Map dataset-specific labels to unified 7-class scheme."""
        pass

    def _pad_or_truncate_features(self, features: np.ndarray) -> np.ndarray:
        """Ensure features have exactly NUM_FEATURES dimensions."""
        current_features = features.shape[1]

        if current_features == self.NUM_FEATURES:
            return features
        elif current_features < self.NUM_FEATURES:
            # Pad with zeros
            padding = np.zeros(
                (features.shape[0], self.NUM_FEATURES - current_features)
            )
            return np.hstack([features, padding])
        else:
            # Truncate (keep most important features)
            return features[:, : self.NUM_FEATURES]

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        if self.info is None:
            raise RuntimeError("Dataset not loaded yet")
        return self.info


class CICIDS2017Loader(BaseDatasetLoader):
    """
    Loader for CICIDS2017 dataset.

    Original classes:
    - BENIGN
    - Bot
    - DDoS
    - DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris
    - FTP-Patator, SSH-Patator
    - Heartbleed
    - Infiltration
    - PortScan
    - Web Attack – Brute Force, XSS, SQL Injection
    """

    # Map CICIDS2017 labels to unified classes
    LABEL_MAPPING = {
        "BENIGN": 0,
        "Bot": 5,
        "DDoS": 1,
        "DoS GoldenEye": 1,
        "DoS Hulk": 1,
        "DoS Slowhttptest": 1,
        "DoS slowloris": 1,
        "FTP-Patator": 3,
        "SSH-Patator": 3,
        "Heartbleed": 4,
        "Infiltration": 6,
        "PortScan": 2,
        "Web Attack – Brute Force": 4,
        "Web Attack – XSS": 4,
        "Web Attack – Sql Injection": 4,
        "Web Attack \x96 Brute Force": 4,
        "Web Attack \x96 XSS": 4,
        "Web Attack \x96 Sql Injection": 4,
    }

    def __init__(self, data_path: Union[str, Path] = None):
        if data_path is None:
            data_path = DATASETS_DIR / "cicids2017_official" / "MachineLearningCVE"
        super().__init__(data_path)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all CICIDS2017 CSV files."""
        logger.info(f"Loading CICIDS2017 from {self.data_path}")

        all_features = []
        all_labels = []

        csv_files = list(self.data_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")

        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")

            try:
                # Read CSV with error handling
                df = pd.read_csv(csv_file, low_memory=False, encoding="utf-8")

                # Clean column names
                df.columns = df.columns.str.strip()

                # Get label column
                label_col = None
                for col in [" Label", "Label", "label"]:
                    if col in df.columns:
                        label_col = col
                        break

                if label_col is None:
                    logger.warning(f"No label column found in {csv_file.name}")
                    continue

                # Extract labels
                labels = df[label_col].values

                # Extract features (drop label and non-numeric columns)
                feature_cols = [
                    col
                    for col in df.columns
                    if col
                    not in [
                        label_col,
                        "Flow ID",
                        "Source IP",
                        "Destination IP",
                        "Timestamp",
                    ]
                ]

                features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
                features = features.fillna(0).values

                all_features.append(features)
                all_labels.append(labels)

            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")
                continue

        if not all_features:
            raise RuntimeError("No data loaded from CICIDS2017")

        # Combine all data
        X = np.vstack(all_features)
        y_raw = np.concatenate(all_labels)

        # Map labels to unified classes
        y = self._map_labels_to_unified(y_raw)

        # Normalize features to standard 79
        X = self._pad_or_truncate_features(X)

        # Clean infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Create info
        unique, counts = np.unique(y, return_counts=True)
        self.info = DatasetInfo(
            name="CICIDS2017",
            num_samples=len(y),
            num_features=X.shape[1],
            num_classes=len(unique),
            class_distribution=dict(zip(unique.tolist(), counts.tolist())),
            source_path=str(self.data_path),
        )

        logger.info(f"CICIDS2017 loaded: {len(y)} samples, {X.shape[1]} features")
        return X, y

    def _map_labels_to_unified(self, labels: np.ndarray) -> np.ndarray:
        """Map CICIDS2017 labels to unified 7-class scheme."""
        unified_labels = np.zeros(len(labels), dtype=np.int64)

        for i, label in enumerate(labels):
            label_str = str(label).strip()
            unified_labels[i] = self.LABEL_MAPPING.get(
                label_str, 0
            )  # Default to Normal

        return unified_labels


class KDDLoader(BaseDatasetLoader):
    """
    Loader for KDD Cup 1999 dataset.

    Original attack categories:
    - normal
    - dos: back, land, neptune, pod, smurf, teardrop
    - probe: ipsweep, nmap, portsweep, satan
    - r2l: ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster
    - u2r: buffer_overflow, loadmodule, perl, rootkit
    """

    # KDD attack types to unified classes
    ATTACK_MAPPING = {
        "normal": 0,
        # DoS attacks -> DDoS
        "back": 1,
        "land": 1,
        "neptune": 1,
        "pod": 1,
        "smurf": 1,
        "teardrop": 1,
        "apache2": 1,
        "mailbomb": 1,
        "processtable": 1,
        "udpstorm": 1,
        # Probe attacks -> Reconnaissance
        "ipsweep": 2,
        "nmap": 2,
        "portsweep": 2,
        "satan": 2,
        "mscan": 2,
        "saint": 2,
        # R2L attacks -> Brute Force / Web Attack
        "ftp_write": 3,
        "guess_passwd": 3,
        "imap": 3,
        "multihop": 4,
        "phf": 4,
        "spy": 6,
        "warezclient": 5,
        "warezmaster": 5,
        "httptunnel": 4,
        "named": 4,
        "sendmail": 4,
        "snmpgetattack": 2,
        "snmpguess": 3,
        "worm": 5,
        "xlock": 3,
        "xsnoop": 2,
        # U2R attacks -> APT
        "buffer_overflow": 6,
        "loadmodule": 6,
        "perl": 6,
        "rootkit": 6,
        "ps": 6,
        "sqlattack": 4,
        "xterm": 6,
    }

    def __init__(self, data_path: Union[str, Path] = None):
        if data_path is None:
            data_path = DATASETS_DIR / "real_datasets" / "kdd_full_minixdr.json"
        super().__init__(data_path)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load KDD dataset."""
        logger.info(f"Loading KDD from {self.data_path}")

        if self.data_path.suffix == ".json":
            return self._load_json()
        else:
            return self._load_csv()

    def _load_json(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load from Mini-XDR JSON format."""
        with open(self.data_path, "r") as f:
            data = json.load(f)

        features = []
        labels = []

        for record in data:
            if "features" in record:
                features.append(record["features"])
            else:
                # Extract features from record fields
                feat = self._extract_features_from_record(record)
                features.append(feat)

            # Get label
            label = record.get("label", record.get("attack_type", "normal"))
            labels.append(label)

        X = np.array(features, dtype=np.float32)
        y_raw = np.array(labels)

        y = self._map_labels_to_unified(y_raw)
        X = self._pad_or_truncate_features(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        unique, counts = np.unique(y, return_counts=True)
        self.info = DatasetInfo(
            name="KDD Cup 1999",
            num_samples=len(y),
            num_features=X.shape[1],
            num_classes=len(unique),
            class_distribution=dict(zip(unique.tolist(), counts.tolist())),
            source_path=str(self.data_path),
        )

        logger.info(f"KDD loaded: {len(y)} samples")
        return X, y

    def _load_csv(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load from CSV format."""
        df = pd.read_csv(self.data_path)

        # Identify label column
        label_col = None
        for col in ["label", "attack_type", "class"]:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            label_col = df.columns[-1]  # Assume last column

        y_raw = df[label_col].values
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values

        y = self._map_labels_to_unified(y_raw)
        X = self._pad_or_truncate_features(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        unique, counts = np.unique(y, return_counts=True)
        self.info = DatasetInfo(
            name="KDD Cup 1999",
            num_samples=len(y),
            num_features=X.shape[1],
            num_classes=len(unique),
            class_distribution=dict(zip(unique.tolist(), counts.tolist())),
            source_path=str(self.data_path),
        )

        return X, y

    def _extract_features_from_record(self, record: Dict[str, Any]) -> List[float]:
        """Extract numerical features from a KDD record."""
        # Standard KDD features
        feature_keys = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
        ]

        features = []
        for key in feature_keys:
            val = record.get(key, 0)
            if isinstance(val, (int, float)):
                features.append(float(val))
            else:
                features.append(0.0)

        return features

    def _map_labels_to_unified(self, labels: np.ndarray) -> np.ndarray:
        """Map KDD labels to unified classes."""
        unified_labels = np.zeros(len(labels), dtype=np.int64)

        for i, label in enumerate(labels):
            label_str = str(label).strip().lower().rstrip(".")
            unified_labels[i] = self.ATTACK_MAPPING.get(label_str, 0)

        return unified_labels


class UNSWLoader(BaseDatasetLoader):
    """
    Loader for UNSW-NB15 dataset.

    Original categories:
    - Normal
    - Analysis, Backdoor, DoS, Exploits, Fuzzers
    - Generic, Reconnaissance, Shellcode, Worms
    """

    # UNSW attack categories to unified classes
    CATEGORY_MAPPING = {
        "normal": 0,
        "dos": 1,
        "reconnaissance": 2,
        "analysis": 2,
        "fuzzers": 2,
        "exploits": 6,
        "backdoor": 6,
        "backdoors": 6,
        "shellcode": 5,
        "worms": 5,
        "generic": 5,
    }

    def __init__(self, data_path: Union[str, Path] = None):
        if data_path is None:
            data_path = DATASETS_DIR / "real_datasets" / "unsw_nb15_sample_minixdr.json"
        super().__init__(data_path)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load UNSW-NB15 dataset."""
        logger.info(f"Loading UNSW-NB15 from {self.data_path}")

        if self.data_path.suffix == ".json":
            with open(self.data_path, "r") as f:
                data = json.load(f)

            features = []
            labels = []

            for record in data:
                if "features" in record:
                    features.append(record["features"])
                else:
                    feat = self._extract_features(record)
                    features.append(feat)

                label = record.get("attack_cat", record.get("label", "normal"))
                labels.append(label)

            X = np.array(features, dtype=np.float32)
            y_raw = np.array(labels)
        else:
            df = pd.read_csv(self.data_path)

            label_col = "attack_cat" if "attack_cat" in df.columns else "label"
            y_raw = df[label_col].values
            X = df.select_dtypes(include=[np.number]).values

        y = self._map_labels_to_unified(y_raw)
        X = self._pad_or_truncate_features(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        unique, counts = np.unique(y, return_counts=True)
        self.info = DatasetInfo(
            name="UNSW-NB15",
            num_samples=len(y),
            num_features=X.shape[1],
            num_classes=len(unique),
            class_distribution=dict(zip(unique.tolist(), counts.tolist())),
            source_path=str(self.data_path),
        )

        logger.info(f"UNSW-NB15 loaded: {len(y)} samples")
        return X, y

    def _extract_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract features from UNSW record."""
        features = []
        for key, val in record.items():
            if key not in ["attack_cat", "label", "id"] and isinstance(
                val, (int, float)
            ):
                features.append(float(val))
        return features

    def _map_labels_to_unified(self, labels: np.ndarray) -> np.ndarray:
        """Map UNSW labels to unified classes."""
        unified_labels = np.zeros(len(labels), dtype=np.int64)

        for i, label in enumerate(labels):
            label_str = str(label).strip().lower()
            unified_labels[i] = self.CATEGORY_MAPPING.get(label_str, 0)

        return unified_labels


class HoneypotLoader(BaseDatasetLoader):
    """
    Loader for Mini-XDR honeypot logs.

    Real-world attack data from T-Pot honeypots.
    """

    # Event types to unified classes
    EVENT_MAPPING = {
        "cowrie.login.failed": 3,  # Brute Force
        "cowrie.login.success": 3,  # Brute Force (successful)
        "cowrie.session.connect": 2,  # Reconnaissance
        "cowrie.command.input": 6,  # APT (command execution)
        "cowrie.session.file_download": 5,  # Malware
        "cowrie.session.file_upload": 5,  # Malware
        "dionaea": 5,  # Malware (dionaea honeypot)
        "elasticpot": 4,  # Web Attack
        "adbhoney": 5,  # Malware
        "ciscoasa": 2,  # Reconnaissance
        "conpot": 6,  # APT (ICS attack)
        "mailoney": 3,  # Brute Force (email)
        "rdpy": 3,  # Brute Force (RDP)
        "tanner": 4,  # Web Attack
    }

    def __init__(self, data_path: Union[str, Path] = None):
        if data_path is None:
            data_path = DATASETS_DIR / "real_datasets" / "honeypot_logs_minixdr.json"
        super().__init__(data_path)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load honeypot logs."""
        logger.info(f"Loading honeypot logs from {self.data_path}")

        with open(self.data_path, "r") as f:
            data = json.load(f)

        features = []
        labels = []

        for record in data:
            feat = self._extract_features(record)
            features.append(feat)

            event_type = record.get("eventid", record.get("type", "unknown"))
            label = self._classify_event(event_type, record)
            labels.append(label)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        X = self._pad_or_truncate_features(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        unique, counts = np.unique(y, return_counts=True)
        self.info = DatasetInfo(
            name="Honeypot Logs",
            num_samples=len(y),
            num_features=X.shape[1],
            num_classes=len(unique),
            class_distribution=dict(zip(unique.tolist(), counts.tolist())),
            source_path=str(self.data_path),
        )

        logger.info(f"Honeypot logs loaded: {len(y)} samples")
        return X, y

    def _extract_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract features from honeypot record."""
        # Standard honeypot features
        features = [
            record.get("dst_port", 0),
            len(record.get("src_ip", "")) > 0,
            record.get("session_duration", 0),
            len(record.get("commands", []))
            if isinstance(record.get("commands"), list)
            else 0,
            record.get("login_attempts", 0),
            record.get("file_downloads", 0),
            record.get("file_uploads", 0),
            1 if record.get("eventid", "").startswith("cowrie") else 0,
            1 if record.get("login_success", False) else 0,
            len(record.get("message", ""))
            if isinstance(record.get("message"), str)
            else 0,
        ]

        # Pad to standard feature length
        while len(features) < 79:
            features.append(0.0)

        return features[:79]

    def _classify_event(self, event_type: str, record: Dict[str, Any]) -> int:
        """Classify honeypot event to unified class."""
        # Check direct mapping
        for prefix, label in self.EVENT_MAPPING.items():
            if event_type.startswith(prefix):
                return label

        # Heuristic classification
        if record.get("login_attempts", 0) > 5:
            return 3  # Brute Force
        if record.get("file_downloads", 0) > 0:
            return 5  # Malware
        if len(record.get("commands", [])) > 3:
            return 6  # APT

        return 0  # Normal

    def _map_labels_to_unified(self, labels: np.ndarray) -> np.ndarray:
        """Already unified during load."""
        return labels


class UnifiedDatasetLoader:
    """
    Master loader that combines all datasets into unified format.

    Features:
    - Loads all available datasets
    - Normalizes to 79 features / 7 classes
    - Provides stratified sampling
    - Memory-efficient batch loading
    """

    CLASS_NAMES = {
        0: "Normal",
        1: "DDoS",
        2: "Reconnaissance",
        3: "Brute Force",
        4: "Web Attack",
        5: "Malware",
        6: "APT",
    }

    def __init__(self, datasets_dir: Union[str, Path] = None):
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DATASETS_DIR
        self.loaders: Dict[str, BaseDatasetLoader] = {}
        self.loaded_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def discover_datasets(self) -> List[str]:
        """Discover available datasets."""
        available = []

        # Check CICIDS2017
        cicids_path = self.datasets_dir / "cicids2017_official" / "MachineLearningCVE"
        if cicids_path.exists() and list(cicids_path.glob("*.csv")):
            available.append("cicids2017")
            self.loaders["cicids2017"] = CICIDS2017Loader(cicids_path)

        # Check KDD
        kdd_paths = [
            self.datasets_dir / "real_datasets" / "kdd_full_minixdr.json",
            self.datasets_dir / "real_datasets" / "kdd_10_percent_minixdr.json",
        ]
        for kdd_path in kdd_paths:
            if kdd_path.exists():
                available.append("kdd")
                self.loaders["kdd"] = KDDLoader(kdd_path)
                break

        # Check UNSW-NB15
        unsw_paths = [
            self.datasets_dir / "full_datasets" / "unsw_nb15_full",
            self.datasets_dir / "real_datasets" / "unsw_nb15_sample_minixdr.json",
        ]
        for unsw_path in unsw_paths:
            if unsw_path.exists():
                available.append("unsw")
                self.loaders["unsw"] = UNSWLoader(unsw_path)
                break

        # Check Honeypot
        honeypot_path = (
            self.datasets_dir / "real_datasets" / "honeypot_logs_minixdr.json"
        )
        if honeypot_path.exists():
            available.append("honeypot")
            self.loaders["honeypot"] = HoneypotLoader(honeypot_path)

        logger.info(f"Discovered {len(available)} datasets: {available}")
        return available

    def load_all(
        self,
        datasets: Optional[List[str]] = None,
        max_samples_per_dataset: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and combine all specified datasets.

        Args:
            datasets: List of dataset names to load (None = all available)
            max_samples_per_dataset: Maximum samples from each dataset

        Returns:
            Combined (features, labels) arrays
        """
        if datasets is None:
            datasets = self.discover_datasets()

        all_X = []
        all_y = []

        for name in datasets:
            if name not in self.loaders:
                logger.warning(f"Dataset {name} not available, skipping")
                continue

            try:
                logger.info(f"Loading {name}...")
                X, y = self.loaders[name].load()

                # Apply sample limit if specified
                if max_samples_per_dataset and len(y) > max_samples_per_dataset:
                    indices = np.random.choice(
                        len(y), max_samples_per_dataset, replace=False
                    )
                    X = X[indices]
                    y = y[indices]

                all_X.append(X)
                all_y.append(y)
                self.loaded_data[name] = (X, y)

                logger.info(f"{name}: {len(y)} samples loaded")

            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                continue

        if not all_X:
            raise RuntimeError("No datasets loaded successfully")

        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        logger.info(
            f"Total combined: {len(y_combined)} samples, {X_combined.shape[1]} features"
        )

        # Log class distribution
        unique, counts = np.unique(y_combined, return_counts=True)
        for cls, count in zip(unique, counts):
            pct = 100.0 * count / len(y_combined)
            logger.info(
                f"  Class {cls} ({self.CLASS_NAMES[cls]}): {count} ({pct:.1f}%)"
            )

        return X_combined, y_combined

    def get_stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified train/val/test splits.

        Returns:
            Dict with 'train', 'val', 'test' keys containing (X, y) tuples
        """
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_ratio,
            stratify=y_trainval,
            random_state=random_state,
        )

        logger.info(
            f"Data split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
        )

        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

    def batch_generator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 256,
        shuffle: bool = True,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Memory-efficient batch generator.

        Yields:
            (batch_X, batch_y) tuples
        """
        n_samples = len(y)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)

        return dict(zip(classes.tolist(), weights.tolist()))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of loaded datasets."""
        summary = {
            "datasets_loaded": list(self.loaded_data.keys()),
            "total_samples": sum(len(y) for _, y in self.loaded_data.values()),
            "per_dataset": {},
        }

        for name, (X, y) in self.loaded_data.items():
            unique, counts = np.unique(y, return_counts=True)
            summary["per_dataset"][name] = {
                "samples": len(y),
                "features": X.shape[1],
                "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
            }

        return summary


# Convenience function
def load_all_datasets(
    max_samples_per_dataset: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, UnifiedDatasetLoader]:
    """
    Load all available datasets.

    Returns:
        (X, y, loader) tuple
    """
    loader = UnifiedDatasetLoader()
    X, y = loader.load_all(max_samples_per_dataset=max_samples_per_dataset)
    return X, y, loader


if __name__ == "__main__":
    # Test loaders
    logging.basicConfig(level=logging.INFO)

    loader = UnifiedDatasetLoader()
    datasets = loader.discover_datasets()
    print(f"Available datasets: {datasets}")

    if datasets:
        X, y = loader.load_all(max_samples_per_dataset=10000)
        print(f"Combined shape: X={X.shape}, y={y.shape}")
        print(f"Summary: {loader.get_summary()}")
