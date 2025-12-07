#!/usr/bin/env python3
"""
Enhanced Windows/AD Dataset Converter for Mini-XDR
Parses real Windows datasets: APT29 Zeek logs, Atomic Red Team techniques
Generates 79-feature vectors mapped to 13 attack classes
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hashlib
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedWindowsConverter:
    """Enhanced converter for Windows attack datasets"""
    
    def __init__(self, 
                 source_dir="$(cd "$(dirname "$0")/../.." ${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))}${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))} pwd)/datasets/windows_ad_datasets",
                 output_dir="$(cd "$(dirname "$0")/../.." ${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))}${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))} pwd)/datasets/windows_converted"):
        
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Expanded 13-class mapping
        self.class_mapping = {
            'normal': 0,
            'ddos': 1,
            'dos': 1,
            'reconnaissance': 2,
            'scan': 2,
            'discovery': 2,
            'brute_force': 3,
            'bruteforce': 3,
            'credential_access': 3,
            'web_attack': 4,
            'malware': 5,
            'botnet': 5,
            'execution': 5,
            'apt': 6,
            'persistence': 6,
            'kerberos_attack': 7,
            'golden_ticket': 7,
            'silver_ticket': 7,
            'kerberoasting': 7,
            'lateral_movement': 8,
            'psexec': 8,
            'wmi': 8,
            'rdp': 8,
            'remote': 8,
            'credential_theft': 9,
            'credential_dumping': 9,
            'mimikatz': 9,
            'dcsync': 9,
            'ntds': 9,
            'lsass': 9,
            'privilege_escalation': 10,
            'escalation': 10,
            'uac': 10,
            'data_exfiltration': 11,
            'exfiltration': 11,
            'collection': 11,
            'insider_threat': 12,
            'defense_evasion': 12,
            'impact': 12
        }
        
        # MITRE ATT&CK technique to class mapping
        self.mitre_to_class = self._build_mitre_mapping()
        
        self.feature_names = self._get_feature_names()
        self.converted_samples = []
        self.stats = defaultdict(int)
    
    def _build_mitre_mapping(self) -> Dict[str, int]:
        """Map MITRE ATT&CK techniques to attack classes"""
        return {
            # Credential Access (Class 9 - Credential Theft)
            'T1003': 9, 'T1003.001': 9, 'T1003.002': 9, 'T1003.003': 9,
            'T1003.004': 9, 'T1003.005': 9, 'T1003.006': 9, 'T1003.007': 9,
            'T1110': 3, 'T1110.001': 3, 'T1110.002': 3, 'T1110.003': 3,
            
            # Lateral Movement (Class 8)
            'T1021': 8, 'T1021.001': 8, 'T1021.002': 8, 'T1021.003': 8,
            'T1021.004': 8, 'T1021.005': 8, 'T1021.006': 8,
            'T1550': 8, 'T1550.002': 8, 'T1550.003': 8,
            
            # Privilege Escalation (Class 10)
            'T1068': 10, 'T1134': 10, 'T1134.001': 10, 'T1134.002': 10,
            'T1548': 10, 'T1548.002': 10, 'T1543': 10,
            
            # Persistence (Class 6)
            'T1053': 6, 'T1053.005': 6, 'T1098': 6, 'T1136': 6,
            'T1543.003': 6, 'T1547': 6,
            
            # Discovery (Class 2)
            'T1007': 2, 'T1012': 2, 'T1018': 2, 'T1033': 2,
            'T1046': 2, 'T1049': 2, 'T1057': 2, 'T1069': 2,
            'T1082': 2, 'T1083': 2, 'T1087': 2, 'T1201': 2,
            
            # Execution (Class 5)
            'T1059': 5, 'T1059.001': 5, 'T1059.003': 5, 'T1059.005': 5,
            'T1106': 5, 'T1204': 5,
            
            # Collection/Exfiltration (Class 11)
            'T1005': 11, 'T1039': 11, 'T1074': 11, 'T1114': 11,
            'T1560': 11, 'T1567': 11, 'T1048': 11,
            
            # Defense Evasion/Insider (Class 12)
            'T1027': 12, 'T1036': 12, 'T1055': 12, 'T1070': 12,
            'T1140': 12, 'T1497': 12, 'T1562': 12,
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get the 79 feature names for Mini-XDR"""
        features = [
            # Network features (20)
            'src_ip_encoded', 'dst_ip_encoded', 'src_port', 'dst_port',
            'protocol', 'bytes_sent', 'bytes_received', 'packets_sent',
            'packets_received', 'duration', 'flags', 'connection_state',
            'ttl', 'window_size', 'tcp_flags', 'icmp_type',
            'flow_id', 'fwd_packets', 'bwd_packets', 'packet_rate',
            
            # Process features (15)
            'process_id', 'parent_process_id', 'process_name_hash',
            'command_line_length', 'process_privileges', 'cpu_usage',
            'memory_usage', 'thread_count', 'handle_count',
            'process_age', 'child_process_count', 'dll_count',
            'registry_access_count', 'file_access_count', 'network_connections',
            
            # Authentication features (12)
            'username_hash', 'domain_hash', 'logon_type', 'authentication_package',
            'failed_auth_count', 'successful_auth_count', 'session_id',
            'logon_time_hour', 'logon_time_day', 'source_workstation_hash',
            'impersonation_level', 'elevation_type',
            
            # File features (10)
            'file_path_hash', 'file_operation', 'file_size',
            'file_extension_hash', 'file_created', 'file_modified',
            'file_accessed', 'file_attributes', 'file_entropy',
            'file_signature_valid',
            
            # Registry features (8)
            'registry_key_hash', 'registry_value_hash', 'registry_operation',
            'registry_path_length', 'hive_type', 'registry_value_type',
            'registry_value_size', 'registry_persistence_indicator',
            
            # Kerberos features (8)
            'ticket_type', 'encryption_type', 'ticket_lifetime',
            'service_name_hash', 'ticket_flags', 'client_realm_hash',
            'server_realm_hash', 'pac_valid',
            
            # Behavioral features (6)
            'event_frequency', 'time_since_last_event', 'anomaly_score',
            'rare_operation_indicator', 'suspicious_timing', 'baseline_deviation'
        ]
        
        assert len(features) == 79, f"Expected 79 features, got {len(features)}"
        return features
    
    def _hash_string(self, value: str) -> int:
        """Hash string to integer"""
        if not value or value == '-':
            return 0
        return int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16) % 1000000
    
    def _parse_zeek_log(self, log_path: Path) -> List[Dict]:
        """Parse Zeek JSON logs"""
        samples = []
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        samples.append(event)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to parse {log_path.name}: {e}")
        
        return samples
    
    def _extract_features_from_zeek(self, event: Dict, is_malicious: bool = True) -> List[float]:
        """Extract 79 features from Zeek event"""
        features = np.zeros(79, dtype=np.float32)
        
        # Network features (0-19)
        features[0] = self._hash_string(event.get('id_orig_h', '')) / 1000000
        features[1] = self._hash_string(event.get('id_resp_h', '')) / 1000000
        features[2] = float(event.get('id_orig_p', 0)) / 65535
        features[3] = float(event.get('id_resp_p', 0)) / 65535
        features[4] = self._map_protocol(event.get('proto', event.get('service', '')))
        features[5] = float(event.get('orig_bytes', 0)) / 1e6
        features[6] = float(event.get('resp_bytes', 0)) / 1e6
        features[7] = float(event.get('orig_pkts', 0)) / 1000
        features[8] = float(event.get('resp_pkts', 0)) / 1000
        features[9] = float(event.get('duration', 0)) / 3600
        features[16] = self._hash_string(event.get('uid', '')) / 1000000
        features[17] = float(event.get('orig_pkts', 0)) / 1000
        features[18] = float(event.get('resp_pkts', 0)) / 1000
        
        # Process features (20-34) - derive from network events
        if 'service' in event:
            features[22] = self._hash_string(event.get('service', '')) / 1000000
        
        # Authentication features (35-46)
        if 'client' in event:  # Kerberos
            client_parts = str(event.get('client', '')).split('/')
            features[35] = self._hash_string(client_parts[0]) / 1000000
            if len(client_parts) > 1:
                features[36] = self._hash_string(client_parts[1]) / 1000000
        
        features[38] = self._map_auth_package(event.get('service', ''))
        
        # Kerberos features (65-72)
        if event.get('@stream') == 'kerberos' or 'request_type' in event:
            features[65] = self._map_request_type(event.get('request_type', ''))
            features[66] = self._map_encryption_type(event.get('cipher', ''))
            features[67] = float(event.get('till', 0)) / 2147483647  # Normalize timestamp
            features[68] = self._hash_string(event.get('service', '')) / 1000000
            features[70] = self._hash_string(event.get('client', '').split('/')[1] if '/' in str(event.get('client', '')) else '') / 1000000
            
            # Detect Kerberos anomalies
            if not event.get('success', True):
                features[75] += 0.3  # Anomaly score
            if 'error_msg' in event:
                features[75] += 0.2
        
        # Behavioral features (73-78)
        features[73] = 0.8 if is_malicious else 0.1  # Event frequency indicator
        features[75] = self._calculate_anomaly_score(event, is_malicious)
        features[76] = 1.0 if self._is_suspicious_timing(event) else 0.0
        features[78] = 0.7 if is_malicious else 0.1  # Baseline deviation
        
        return features.tolist()
    
    def _map_protocol(self, protocol: str) -> float:
        """Map protocol to float"""
        protocol_map = {
            'tcp': 0.33, 'udp': 0.66, 'icmp': 1.0,
            'kerberos': 0.4, 'smb': 0.5, 'dce_rpc': 0.6,
            'http': 0.7, 'dns': 0.8, 'ssl': 0.9
        }
        return protocol_map.get(str(protocol).lower(), 0.0)
    
    def _map_auth_package(self, package: str) -> float:
        """Map authentication package"""
        package_map = {'ntlm': 0.5, 'kerberos': 1.0, 'negotiate': 0.75}
        return package_map.get(str(package).lower(), 0.0)
    
    def _map_request_type(self, req_type: str) -> float:
        """Map Kerberos request type"""
        type_map = {'AS': 0.33, 'TGS': 0.66, 'AP': 1.0}
        return type_map.get(str(req_type), 0.0)
    
    def _map_encryption_type(self, enc_type: str) -> float:
        """Map encryption type"""
        enc_map = {
            'des': 0.25, 'rc4': 0.5, 'rc4-hmac': 0.5,
            'aes128': 0.75, 'aes256': 1.0,
            'aes128-cts-hmac-sha1-96': 0.75,
            'aes256-cts-hmac-sha1-96': 1.0
        }
        return enc_map.get(str(enc_type).lower(), 0.5)
    
    def _calculate_anomaly_score(self, event: Dict, is_malicious: bool) -> float:
        """Calculate basic anomaly score"""
        score = 0.0
        
        if is_malicious:
            score += 0.3
        
        # Check for Kerberos errors
        if event.get('error_msg'):
            score += 0.3
        
        # Check for suspicious patterns
        if not event.get('success', True):
            score += 0.2
        
        # High port numbers
        if event.get('id_resp_p', 0) > 49152:
            score += 0.1
        
        return min(score, 1.0)
    
    def _is_suspicious_timing(self, event: Dict) -> bool:
        """Check if event timing is suspicious"""
        # Check timestamp for off-hours activity
        ts = event.get('ts', 0)
        if ts > 0:
            hour = datetime.fromtimestamp(ts).hour
            # Suspicious hours: 10pm - 6am
            return hour >= 22 or hour <= 6
        return False
    
    def _infer_attack_class_from_zeek(self, event: Dict, source_file: str) -> int:
        """Infer attack class from Zeek event"""
        # Check stream type
        stream = event.get('@stream', '')
        
        # Kerberos attacks
        if stream == 'kerberos' or 'kerberos' in source_file.lower():
            # Check for failed authentication (potential Kerberoasting)
            if not event.get('success', True) and event.get('error_msg'):
                return 7  # Kerberos attack
            # Successful Kerberos with TGS (potential golden/silver ticket)
            if event.get('request_type') == 'TGS':
                return 7
        
        # Lateral movement indicators
        if stream in ['dce_rpc', 'smb'] or event.get('id_resp_p') in [445, 135, 139]:
            return 8  # Lateral movement
        
        # RDP
        if event.get('id_resp_p') == 3389:
            return 8  # Lateral movement
        
        # Data exfiltration (large uploads, HTTP/HTTPS)
        if stream == 'http' and event.get('resp_bytes', 0) > 1000000:
            return 11  # Data exfiltration
        
        # DNS reconnaissance
        if stream == 'dns':
            return 2  # Reconnaissance
        
        # Default to APT for unclassified APT29 events
        return 6
    
    def _parse_atomic_test(self, yaml_path: Path) -> List[Dict]:
        """Parse Atomic Red Team YAML test"""
        samples = []
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            technique_id = data.get('attack_technique', '')
            technique_name = data.get('display_name', '')
            
            # Map technique to attack class
            attack_class = self.mitre_to_class.get(technique_id, 2)  # Default to reconnaissance
            
            # Generate samples for each atomic test
            for test in data.get('atomic_tests', []):
                test_name = test.get('name', '')
                description = test.get('description', '')
                executor = test.get('executor', {})
                command = executor.get('command', '')
                
                # Generate synthetic features based on technique
                features = self._generate_features_from_technique(
                    technique_id, test_name, description, command
                )
                
                samples.append({
                    'features': features,
                    'label': attack_class,
                    'technique': technique_id,
                    'technique_name': technique_name,
                    'test_name': test_name,
                    'source': 'atomic_red_team'
                })
        
        except Exception as e:
            logger.debug(f"Failed to parse {yaml_path.name}: {e}")
        
        return samples
    
    def _generate_features_from_technique(self, technique_id: str, 
                                         test_name: str,
                                         description: str,
                                         command: str) -> List[float]:
        """Generate synthetic features from MITRE ATT&CK technique"""
        features = np.random.rand(79) * 0.1  # Base noise
        
        # Process features based on technique category
        if 'T1003' in technique_id:  # Credential dumping
            features[20] = 0.9  # process_id indicator
            features[22] = self._hash_string('lsass.exe') / 1000000
            features[23] = 0.8  # command_line_length
            features[24] = 1.0  # High privileges
            features[75] = 0.9  # High anomaly score
            
        elif 'T1021' in technique_id:  # Lateral movement
            features[2] = 0.8  # src_port
            features[3] = 445/65535 if 'smb' in test_name.lower() else 3389/65535
            features[4] = 0.5  # SMB/RDP protocol
            features[75] = 0.85
            
        elif 'T1053' in technique_id:  # Scheduled tasks (Persistence)
            features[22] = self._hash_string('schtasks.exe') / 1000000
            features[24] = 1.0  # System privileges
            features[75] = 0.7
            
        elif 'T1055' in technique_id:  # Process injection
            features[20] = 0.95
            features[21] = 0.9  # Parent process
            features[26] = 0.9  # Memory usage
            features[75] = 0.95
            
        elif 'T1059' in technique_id:  # Command execution
            features[22] = self._hash_string('cmd.exe' if '.001' in technique_id else 'powershell.exe') / 1000000
            features[23] = 0.7  # Long command line
            features[75] = 0.75
        
        # Authentication features for relevant techniques
        if 'T1110' in technique_id:  # Brute force
            features[39] = 10.0  # failed_auth_count
            features[75] = 0.9
        
        # Kerberos techniques
        if technique_id in ['T1558', 'T1550.003']:
            features[65] = 0.66  # TGS
            features[66] = 0.5  # RC4 (weak encryption)
            features[75] = 0.95
        
        # Behavioral indicators
        features[73] = np.random.uniform(0.6, 0.9)  # Event frequency
        features[76] = 1.0 if np.random.rand() > 0.5 else 0.0  # Suspicious timing
        features[78] = np.random.uniform(0.5, 0.9)  # Baseline deviation
        
        return features.tolist()
    
    def convert_apt29_dataset(self) -> int:
        """Convert APT29 Zeek logs"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 Converting APT29 Zeek Logs")
        logger.info("=" * 70)
        
        apt29_dir = self.source_dir / 'apt29' / 'datasets'
        zeek_logs = list(apt29_dir.rglob("*.log"))
        
        logger.info(f"Found {len(zeek_logs)} Zeek log files")
        
        total_events = 0
        for log_file in zeek_logs:
            if log_file.stat().st_size == 0:
                continue
            
            logger.info(f"Processing: {log_file.relative_to(apt29_dir)}")
            events = self._parse_zeek_log(log_file)
            
            for event in events:
                features = self._extract_features_from_zeek(event, is_malicious=True)
                label = self._infer_attack_class_from_zeek(event, log_file.name)
                
                self.converted_samples.append({
                    'features': features,
                    'label': label,
                    'source': 'apt29',
                    'source_file': log_file.name,
                    'attack_type': self._get_attack_name(label)
                })
                
                self.stats[f'class_{label}'] += 1
            
            total_events += len(events)
            logger.info(f"  ✅ Converted {len(events)} events")
        
        logger.info(f"\n✅ APT29 Total: {total_events:,} events")
        return total_events
    
    def convert_atomic_red_team(self, max_samples_per_technique: int = 5) -> int:
        """Convert Atomic Red Team techniques to training samples"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 Converting Atomic Red Team Techniques")
        logger.info("=" * 70)
        
        atomics_dir = self.source_dir / 'atomic_red_team' / 'atomics'
        yaml_files = list(atomics_dir.rglob("*.yaml"))
        
        logger.info(f"Found {len(yaml_files)} technique files")
        
        total_samples = 0
        technique_count = 0
        
        for yaml_file in yaml_files:
            samples = self._parse_atomic_test(yaml_file)
            
            # Limit samples per technique to avoid overwhelming one class
            samples = samples[:max_samples_per_technique]
            
            for sample in samples:
                self.converted_samples.append(sample)
                self.stats[f"class_{sample['label']}"] += 1
                total_samples += 1
            
            if samples:
                technique_count += 1
                if technique_count % 50 == 0:
                    logger.info(f"  Processed {technique_count} techniques...")
        
        logger.info(f"\n✅ Atomic Red Team Total: {total_samples:,} samples from {technique_count} techniques")
        return total_samples
    
    def _get_attack_name(self, label: int) -> str:
        """Get attack name from label"""
        attack_names = {
            0: 'normal', 1: 'ddos', 2: 'reconnaissance',
            3: 'brute_force', 4: 'web_attack', 5: 'malware',
            6: 'apt', 7: 'kerberos_attack', 8: 'lateral_movement',
            9: 'credential_theft', 10: 'privilege_escalation',
            11: 'data_exfiltration', 12: 'insider_threat'
        }
        return attack_names.get(label, 'unknown')
    
    def generate_normal_traffic(self, count: int = 5000) -> int:
        """Generate synthetic normal traffic samples"""
        logger.info("\n" + "=" * 70)
        logger.info(f"📊 Generating {count:,} Normal Traffic Samples")
        logger.info("=" * 70)
        
        for i in range(count):
            # Generate benign network features
            features = np.random.rand(79) * 0.3  # Low values for normal
            
            # Typical benign patterns
            features[2] = np.random.uniform(0.5, 1.0)  # High ports
            features[3] = np.random.choice([80, 443, 53, 22]) / 65535  # Common services
            features[4] = np.random.choice([0.33, 0.66, 0.7, 0.8])  # TCP/UDP/HTTP/DNS
            features[9] = np.random.exponential(0.1)  # Short duration
            
            # Low anomaly indicators
            features[73] = np.random.uniform(0.0, 0.2)
            features[75] = np.random.uniform(0.0, 0.1)
            features[76] = 0.0
            features[78] = np.random.uniform(0.0, 0.2)
            
            self.converted_samples.append({
                'features': features.tolist(),
                'label': 0,
                'source': 'synthetic_normal',
                'attack_type': 'normal'
            })
            
            self.stats['class_0'] += 1
        
        logger.info(f"✅ Generated {count:,} normal samples")
        return count
    
    def save_converted_data(self):
        """Save all converted samples"""
        logger.info("\n" + "=" * 70)
        logger.info("💾 Saving Converted Dataset")
        logger.info("=" * 70)
        
        # Save JSON
        output_json = self.output_dir / "windows_ad_enhanced.json"
        with open(output_json, 'w') as f:
            json.dump({
                'metadata': {
                    'total_samples': len(self.converted_samples),
                    'features': 79,
                    'classes': 13,
                    'class_distribution': dict(self.stats),
                    'converted_at': datetime.now().isoformat(),
                    'sources': ['apt29', 'atomic_red_team', 'synthetic_normal']
                },
                'samples': self.converted_samples
            }, f, indent=2)
        
        logger.info(f"✅ Saved JSON: {output_json}")
        
        # Save NumPy arrays for fast training
        X = np.array([s['features'] for s in self.converted_samples], dtype=np.float32)
        y = np.array([s['label'] for s in self.converted_samples], dtype=np.int64)
        
        np.save(self.output_dir / "windows_features.npy", X)
        np.save(self.output_dir / "windows_labels.npy", y)
        
        logger.info(f"✅ Saved NumPy arrays: windows_features.npy, windows_labels.npy")
        logger.info(f"   Shape: X={X.shape}, y={y.shape}")
        
        # Save CSV for inspection
        csv_data = []
        for sample in self.converted_samples:
            row = sample['features'] + [sample['label'], sample.get('attack_type', 'unknown')]
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data, columns=self.feature_names + ['label', 'attack_type'])
        csv_file = self.output_dir / "windows_ad_enhanced.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"✅ Saved CSV: {csv_file}")
    
    def print_summary(self):
        """Print conversion summary"""
        logger.info("\n" + "=" * 70)
        logger.info("📈 CONVERSION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total samples: {len(self.converted_samples):,}")
        logger.info(f"\nClass distribution:")
        
        for class_id in range(13):
            count = self.stats.get(f'class_{class_id}', 0)
            if count > 0:
                pct = (count / len(self.converted_samples)) * 100
                attack_name = self._get_attack_name(class_id)
                logger.info(f"  Class {class_id:2d} ({attack_name:20s}): {count:6,} samples ({pct:5.1f}%)")
        
        logger.info("=" * 70)


def main():
    """Main execution"""
    logger.info("=" * 70)
    logger.info("🎯 ENHANCED WINDOWS/AD DATASET CONVERTER")
    logger.info("=" * 70)
    logger.info("Generates 79-feature vectors for 13 attack classes\n")
    
    try:
        converter = EnhancedWindowsConverter()
        
        # Convert APT29 Zeek logs (real network/Kerberos events)
        apt29_count = converter.convert_apt29_dataset()
        
        # Convert Atomic Red Team techniques (MITRE ATT&CK)
        atomic_count = converter.convert_atomic_red_team(max_samples_per_technique=3)
        
        # Generate normal traffic baseline
        normal_count = converter.generate_normal_traffic(count=5000)
        
        # Save all data
        converter.save_converted_data()
        
        # Print summary
        converter.print_summary()
        
        logger.info("\n✅ CONVERSION COMPLETE!")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Review: datasets/windows_converted/windows_ad_enhanced.csv")
        logger.info("   2. Train Windows specialist: aws/train_windows_specialist.py")
        logger.info("   3. Integrate with ensemble: backend/app/ensemble_ml_detector.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

