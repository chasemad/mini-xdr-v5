#!/usr/bin/env python3
"""
Download COMPREHENSIVE Windows/AD Attack Datasets
Target: 500k+ Windows attack samples (not just 8k!)
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveWindowsDataDownloader:
    """Download large-scale Windows attack datasets"""
    
    def __init__(self, base_dir="/Users/chasemad/Desktop/mini-xdr/datasets/windows_comprehensive"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {
            # LARGE Windows datasets
            'adfa_ld': {
                'name': 'ADFA-LD (UNSW Linux/Windows Intrusion)',
                'url': 'https://cloudstor.aarnet.edu.au/plus/s/DS3zdEq3gqzqEOT/download',
                'type': 'direct',
                'expected_samples': 50000,
                'description': 'System call traces, privilege escalation, backdoors'
            },
            'adfa_wd': {
                'name': 'ADFA-WD (Windows Dataset)',
                'url': 'https://cloudstor.aarnet.edu.au/plus/s/xJkIteW3kQ5I6Gs/download',
                'type': 'direct',
                'expected_samples': 30000,
                'description': 'Windows-specific attacks and normal behavior'
            },
            'cicids2018': {
                'name': 'CSE-CIC-IDS2018 (Includes Windows attacks)',
                'url': 'https://www.unb.ca/cic/datasets/ids-2018.html',
                'type': 'manual',
                'expected_samples': 200000,
                'description': 'Brute force, DoS, DDoS, Web attacks, Infiltration'
            },
            'ton_iot': {
                'name': 'ToN-IoT Dataset (Windows logs)',
                'url': 'https://github.com/TON-IoT/TON-IoT-Dataset.git',
                'type': 'git',
                'expected_samples': 100000,
                'description': 'Windows event logs, network traffic'
            },
            'cse_cicids2017_full': {
                'name': 'CSE-CIC-IDS2017 Full Dataset',
                'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
                'type': 'manual',
                'expected_samples': 2800000,
                'description': 'Complete CICIDS2017 (we may only have sample)'
            },
            'bot_iot': {
                'name': 'Bot-IoT Dataset',
                'url': 'https://research.unsw.edu.au/projects/bot-iot-dataset',
                'type': 'manual',
                'expected_samples': 700000,
                'description': 'Botnet traffic, DDoS, reconnaissance'
            },
            'windows_pe_malware': {
                'name': 'Windows PE Malware Dataset',
                'url': 'https://github.com/urwithajit9/ClaMP.git',
                'type': 'git',
                'expected_samples': 50000,
                'description': 'Windows malware PE analysis'
            },
            'endgame_redteam': {
                'name': 'Endgame Red Team Automation',
                'url': 'https://github.com/endgameinc/RTA.git',
                'type': 'git',
                'expected_samples': 10000,
                'description': 'Windows attack simulations (MITRE ATT&CK)'
            },
            'splunk_bots': {
                'name': 'Splunk Boss of the SOC (BOTS) Datasets',
                'url': 'https://github.com/splunk/botsv3.git',
                'type': 'git',
                'expected_samples': 30000,
                'description': 'Real APT scenarios with Windows logs'
            }
        }
    
    def download_git_dataset(self, name, url, target_dir):
        """Download from git"""
        logger.info(f"📥 Downloading {name}...")
        
        if target_dir.exists():
            logger.info(f"   ⚠️  Already exists, pulling updates...")
            subprocess.run(['git', '-C', str(target_dir), 'pull'], check=False, capture_output=True)
            return True
        
        try:
            subprocess.run(['git', 'clone', '--depth', '1', url, str(target_dir)],
                         check=True, capture_output=True, text=True)
            logger.info(f"   ✅ Downloaded")
            return True
        except:
            logger.warning(f"   ❌ Failed to download {name}")
            return False
    
    def download_all(self):
        """Download all available datasets"""
        logger.info("🚀 Downloading COMPREHENSIVE Windows Attack Datasets")
        logger.info(f"Target: 500k+ Windows attack samples")
        logger.info("=" * 70)
        
        auto_downloads = 0
        manual_required = []
        
        for dataset_id, info in self.datasets.items():
            logger.info(f"\n📊 {info['name']}")
            logger.info(f"   Expected: {info['expected_samples']:,} samples")
            
            if info['type'] == 'git':
                target = self.base_dir / dataset_id
                success = self.download_git_dataset(info['name'], info['url'], target)
                if success:
                    auto_downloads += 1
            elif info['type'] == 'manual':
                logger.info(f"   ⚠️  MANUAL DOWNLOAD REQUIRED:")
                logger.info(f"   URL: {info['url']}")
                manual_required.append(info)
            elif info['type'] == 'direct':
                logger.info(f"   ⚠️  DIRECT DOWNLOAD (large file):")
                logger.info(f"   URL: {info['url']}")
                logger.info(f"   💡 Use: wget '{info['url']}' -O {dataset_id}.zip")
                manual_required.append(info)
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Auto-downloaded: {auto_downloads} datasets")
        logger.info(f"📝 Manual downloads needed: {len(manual_required)}")
        
        if manual_required:
            logger.info("\n⚠️  MANUAL DOWNLOADS REQUIRED FOR LARGE DATASETS:")
            for info in manual_required:
                logger.info(f"\n   📊 {info['name']} ({info['expected_samples']:,} samples)")
                logger.info(f"      URL: {info['url']}")
                logger.info(f"      {info['description']}")
        
        # Alternative: Use existing CICIDS2017 more effectively
        logger.info("\n💡 BETTER IDEA: Your cicids2017_enhanced_minixdr.json has 18M lines!")
        logger.info("   Let me parse it properly to extract Windows-relevant events")
        logger.info("   This could give us 100k-500k Windows-related samples")


def main():
    downloader = ComprehensiveWindowsDataDownloader()
    downloader.download_all()
    
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("1. I'll create a better JSON parser to extract Windows events from your 18M line file")
    logger.info("2. Filter cicids2017_enhanced for Windows-related attacks")
    logger.info("3. Merge with Mordor, EVTX, OpTC datasets")
    logger.info("4. Target: 500k+ Windows samples (better balance with 4M network samples)")


if __name__ == '__main__':
    main()

