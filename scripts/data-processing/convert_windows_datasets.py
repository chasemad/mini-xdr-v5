#!/usr/bin/env python3
"""
Convert Windows/AD Attack Datasets to Mini-XDR Format
Converts: Mordor, EVTX, OpTC, APT29, Atomic Red Team → 79-feature format
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WindowsDatasetConverter:
    """Convert Windows attack datasets to Mini-XDR format"""
    
    def __init__(self, 
                 source_dir="/Users/chasemad/Desktop/mini-xdr/datasets/windows_ad_datasets",
                 output_dir="/Users/chasemad/Desktop/mini-xdr/datasets/windows_converted"):
        
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Attack class mapping (expanding from 7 to 13 classes)
        self.class_mapping = {
            'normal': 0,
            'ddos': 1,
            'dos': 1,
            'reconnaissance': 2,
            'scan': 2,
            'brute_force': 3,
            'bruteforce': 3,
            'web_attack': 4,
            'malware': 5,
            'botnet': 5,
            'apt': 6,
            'kerberos_attack': 7,  # NEW
            'golden_ticket': 7,
            'silver_ticket': 7,
            'kerberoasting': 7,
            'lateral_movement': 8,  # NEW
            'psexec': 8,
            'wmi': 8,
            'rdp': 8,
            'credential_theft': 9,  # NEW
            'mimikatz': 9,
            'dcsync': 9,
            'ntds': 9,
            'privilege_escalation': 10,  # NEW
            'escalation': 10,
            'data_exfiltration': 11,  # NEW
            'exfiltration': 11,
            'insider_threat': 12  # NEW
        }
        
        self.feature_names = self._get_feature_names()
        self.converted_samples = []
    
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
        if not value:
            return 0
        return int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16) % 1000000
    
    def _extract_features_from_json(self, event: Dict) -> List[float]:
        """Extract 79 features from JSON event"""
        features = np.zeros(79, dtype=np.float32)
        
        # Network features (0-19)
        features[0] = self._hash_string(event.get('SourceAddress', '')) / 1000000
        features[1] = self._hash_string(event.get('DestinationAddress', '')) / 1000000
        features[2] = float(event.get('SourcePort', 0)) / 65535
        features[3] = float(event.get('DestinationPort', 0)) / 65535
        features[4] = self._map_protocol(event.get('Protocol', ''))
        features[5] = float(event.get('BytesSent', 0)) / 1e6
        features[6] = float(event.get('BytesReceived', 0)) / 1e6
        features[7] = float(event.get('PacketsSent', 0)) / 1000
        features[8] = float(event.get('PacketsReceived', 0)) / 1000
        features[9] = float(event.get('Duration', 0)) / 3600
        
        # Process features (20-34)
        features[20] = float(event.get('ProcessId', 0)) / 100000
        features[21] = float(event.get('ParentProcessId', 0)) / 100000
        features[22] = self._hash_string(event.get('ProcessName', '')) / 1000000
        features[23] = len(str(event.get('CommandLine', ''))) / 1000
        features[24] = self._map_privilege_level(event.get('IntegrityLevel', ''))
        
        # Authentication features (35-46)
        features[35] = self._hash_string(event.get('TargetUserName', '')) / 1000000
        features[36] = self._hash_string(event.get('TargetDomainName', '')) / 1000000
        features[37] = float(event.get('LogonType', 0)) / 11
        features[38] = self._map_auth_package(event.get('AuthenticationPackageName', ''))
        
        # File features (47-56)
        features[47] = self._hash_string(event.get('ObjectName', '')) / 1000000
        features[48] = self._map_file_operation(event.get('AccessMask', ''))
        
        # Registry features (57-64)
        features[57] = self._hash_string(event.get('TargetObject', '')) / 1000000
        features[58] = self._hash_string(event.get('Details', '')) / 1000000
        
        # Kerberos features (65-72)
        if 'TicketEncryptionType' in event:
            features[65] = float(event.get('ServiceName', 0)) / 10
            features[66] = self._map_encryption_type(event.get('TicketEncryptionType', ''))
            features[67] = float(event.get('TicketOptions', 0)) / 1000
        
        # Behavioral features (73-78)
        features[73] = np.random.uniform(0, 1)  # Placeholder for real behavioral analysis
        features[74] = np.random.uniform(0, 1)
        features[75] = self._calculate_anomaly_score(event)
        features[76] = 1.0 if self._is_suspicious_timing(event) else 0.0
        
        return features.tolist()
    
    def _map_protocol(self, protocol: str) -> float:
        """Map protocol to float"""
        protocol_map = {'tcp': 0.33, 'udp': 0.66, 'icmp': 1.0}
        return protocol_map.get(str(protocol).lower(), 0.0)
    
    def _map_privilege_level(self, level: str) -> float:
        """Map privilege level to float"""
        level_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'system': 1.0}
        return level_map.get(str(level).lower(), 0.5)
    
    def _map_auth_package(self, package: str) -> float:
        """Map authentication package"""
        package_map = {'ntlm': 0.5, 'kerberos': 1.0, 'negotiate': 0.75}
        return package_map.get(str(package).lower(), 0.0)
    
    def _map_file_operation(self, access: str) -> float:
        """Map file operation"""
        return float(hash(str(access)) % 1000) / 1000
    
    def _map_encryption_type(self, enc_type: str) -> float:
        """Map encryption type"""
        enc_map = {'des': 0.25, 'rc4': 0.5, 'aes128': 0.75, 'aes256': 1.0}
        return enc_map.get(str(enc_type).lower(), 0.5)
    
    def _calculate_anomaly_score(self, event: Dict) -> float:
        """Calculate basic anomaly score"""
        score = 0.0
        
        # Check for known suspicious indicators
        suspicious_keywords = ['mimikatz', 'psexec', 'powershell', 'cmd', 'wmic']
        command_line = str(event.get('CommandLine', '')).lower()
        
        for keyword in suspicious_keywords:
            if keyword in command_line:
                score += 0.2
        
        return min(score, 1.0)
    
    def _is_suspicious_timing(self, event: Dict) -> bool:
        """Check if event timing is suspicious"""
        # Placeholder - would need real timestamp analysis
        return False
    
    def _infer_attack_class(self, event: Dict, filename: str) -> int:
        """Infer attack class from event data and filename"""
        filename_lower = filename.lower()
        
        # Check filename patterns
        for attack_type, class_id in self.class_mapping.items():
            if attack_type in filename_lower:
                return class_id
        
        # Check event content
        event_str = json.dumps(event).lower()
        
        # Kerberos attacks
        if any(k in event_str for k in ['golden_ticket', 'silver_ticket', 'kerberoasting']):
            return 7
        
        # Lateral movement
        if any(k in event_str for k in ['psexec', 'wmi', 'remote', 'lateral']):
            return 8
        
        # Credential theft
        if any(k in event_str for k in ['mimikatz', 'lsass', 'dcsync', 'ntds']):
            return 9
        
        # Privilege escalation
        if any(k in event_str for k in ['escalation', 'privilege', 'uac']):
            return 10
        
        # Data exfiltration
        if any(k in event_str for k in ['exfiltration', 'download', 'upload']):
            return 11
        
        # Default to reconnaissance if unclear
        return 2
    
    def convert_json_file(self, json_file: Path) -> List[Dict]:
        """Convert single JSON file"""
        samples = []
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                events = data
            elif isinstance(data, dict):
                events = data.get('value', data.get('events', [data]))
            else:
                return samples
            
            for event in events:
                if not isinstance(event, dict):
                    continue
                
                features = self._extract_features_from_json(event)
                label = self._infer_attack_class(event, json_file.name)
                
                samples.append({
                    'features': features,
                    'label': label,
                    'source_file': str(json_file.name),
                    'attack_type': self._get_attack_name(label),
                    'metadata': {
                        'original_event_id': event.get('EventID', event.get('@timestamp', '')),
                        'source_dataset': json_file.parent.name
                    }
                })
        
        except Exception as e:
            logger.warning(f"Failed to parse {json_file.name}: {e}")
        
        return samples
    
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
    
    def convert_all_datasets(self):
        """Convert all downloaded datasets"""
        logger.info("🔄 Converting Windows/AD datasets to Mini-XDR format")
        logger.info("=" * 70)
        
        json_files = list(self.source_dir.rglob("*.json"))
        logger.info(f"📁 Found {len(json_files)} JSON files to convert")
        
        total_samples = 0
        class_counts = {i: 0 for i in range(13)}
        
        for json_file in json_files:
            logger.info(f"Processing: {json_file.relative_to(self.source_dir)}")
            samples = self.convert_json_file(json_file)
            
            for sample in samples:
                class_counts[sample['label']] += 1
            
            self.converted_samples.extend(samples)
            total_samples += len(samples)
            
            if len(samples) > 0:
                logger.info(f"  ✅ Converted {len(samples)} samples")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Total samples converted: {total_samples:,}")
        logger.info("\n📊 Class distribution:")
        
        for class_id, count in sorted(class_counts.items()):
            if count > 0:
                attack_name = self._get_attack_name(class_id)
                logger.info(f"   Class {class_id} ({attack_name}): {count:,} samples")
        
        # Save converted data
        self._save_converted_data()
        
        return total_samples
    
    def _save_converted_data(self):
        """Save converted samples to file"""
        output_file = self.output_dir / "windows_ad_converted.json"
        
        logger.info(f"\n💾 Saving converted data to: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'total_samples': len(self.converted_samples),
                    'features': 79,
                    'classes': 13,
                    'converted_at': datetime.now().isoformat(),
                    'source': 'Windows/AD attack datasets'
                },
                'events': self.converted_samples
            }, f, indent=2)
        
        logger.info(f"✅ Saved {len(self.converted_samples):,} samples")
        
        # Also save CSV version for easy inspection
        csv_file = self.output_dir / "windows_ad_converted.csv"
        df_data = []
        
        for sample in self.converted_samples:
            row = sample['features'] + [sample['label']]
            df_data.append(row)
        
        df = pd.DataFrame(df_data, columns=self.feature_names + ['label'])
        df.to_csv(csv_file, index=False)
        
        logger.info(f"✅ Also saved CSV version: {csv_file}")


def main():
    """Main execution"""
    logger.info("🎯 Windows/AD Dataset Converter")
    logger.info("Converting to Mini-XDR format (79 features, 13 classes)\n")
    
    try:
        converter = WindowsDatasetConverter()
        total_samples = converter.convert_all_datasets()
        
        logger.info("\n✅ Conversion complete!")
        logger.info(f"📊 Total samples: {total_samples:,}")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Review converted data in: datasets/windows_converted/")
        logger.info("   2. Merge with existing 4M+ events")
        logger.info("   3. Launch Azure ML training for fast GPU training")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

