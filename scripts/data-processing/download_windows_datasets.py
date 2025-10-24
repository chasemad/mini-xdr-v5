#!/usr/bin/env python3
"""
Download Windows/AD Attack Datasets for Enterprise XDR Training
Downloads from public sources: ADFA-LD, OpTC, Mordor, EVTX samples
Target: 11,000+ samples of Windows and Active Directory attacks
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WindowsDatasetDownloader:
    """Downloads and extracts Windows/AD attack datasets"""
    
    def __init__(self, base_dir="/Users/chasemad/Desktop/mini-xdr/datasets/windows_ad_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset sources
        self.datasets = {
            'mordor': {
                'name': 'Mordor Security Datasets (MITRE ATT&CK)',
                'url': 'https://github.com/OTRF/Security-Datasets.git',
                'type': 'git',
                'target_dir': self.base_dir / 'mordor',
                'expected_samples': 2000,
                'description': 'Kerberos attacks, Golden Ticket, DCSync, Pass-the-hash'
            },
            'evtx_samples': {
                'name': 'Windows Event Log Attack Samples',
                'url': 'https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES.git',
                'type': 'git',
                'target_dir': self.base_dir / 'evtx_samples',
                'expected_samples': 1000,
                'description': 'Mimikatz, PSExec, PowerShell attacks'
            },
            'optc': {
                'name': 'OpTC Dataset (DARPA Operational Technology)',
                'url': 'https://github.com/FiveDirections/OpTC-data.git',
                'type': 'git',
                'target_dir': self.base_dir / 'optc',
                'expected_samples': 3000,
                'description': 'Lateral movement, C2, exfiltration'
            },
            # Additional valuable datasets
            'apt29': {
                'name': 'APT29 Evals Dataset',
                'url': 'https://github.com/OTRF/detection-hackathon-apt29.git',
                'type': 'git',
                'target_dir': self.base_dir / 'apt29',
                'expected_samples': 500,
                'description': 'Advanced Persistent Threat simulations'
            },
            'atomic_red_team': {
                'name': 'Atomic Red Team Test Data',
                'url': 'https://github.com/redcanaryco/atomic-red-team.git',
                'type': 'git',
                'target_dir': self.base_dir / 'atomic_red_team',
                'expected_samples': 1500,
                'description': 'MITRE ATT&CK technique testing'
            }
        }
    
    def download_git_dataset(self, name, url, target_dir):
        """Download dataset from git repository"""
        logger.info(f"📥 Downloading {name} from GitHub...")
        
        if target_dir.exists():
            logger.info(f"   ⚠️  Directory exists, pulling latest changes...")
            try:
                subprocess.run(['git', '-C', str(target_dir), 'pull'], 
                             check=True, capture_output=True)
                logger.info(f"   ✅ Updated {name}")
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(f"   ⚠️  Pull failed, will re-clone: {e}")
                import shutil
                shutil.rmtree(target_dir)
        
        try:
            # Clone with depth 1 for faster download
            logger.info(f"   🔄 Cloning repository...")
            subprocess.run(
                ['git', 'clone', '--depth', '1', url, str(target_dir)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"   ✅ Downloaded {name} successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ Failed to download {name}: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"   ❌ Error downloading {name}: {e}")
            return False
    
    def download_http_dataset(self, name, url, target_dir):
        """Download dataset from HTTP/HTTPS"""
        logger.info(f"📥 Downloading {name} from {url}...")
        
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(urlparse(url).path).name
            output_file = target_dir / filename
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            # Extract if archive
            if filename.endswith('.zip'):
                logger.info(f"   📦 Extracting zip archive...")
                with zipfile.ZipFile(output_file, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                output_file.unlink()  # Remove zip after extraction
            elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
                logger.info(f"   📦 Extracting tar.gz archive...")
                with tarfile.open(output_file, 'r:gz') as tar_ref:
                    tar_ref.extractall(target_dir)
                output_file.unlink()
            
            logger.info(f"   ✅ Downloaded {name} successfully")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to download {name}: {e}")
            return False
    
    def download_all(self):
        """Download all datasets"""
        logger.info("🚀 Starting Windows/AD Dataset Download")
        logger.info("=" * 70)
        
        results = {}
        total_expected = 0
        
        for dataset_id, dataset_info in self.datasets.items():
            logger.info(f"\n📊 Dataset: {dataset_info['name']}")
            logger.info(f"   Expected samples: {dataset_info['expected_samples']:,}")
            logger.info(f"   Description: {dataset_info['description']}")
            
            total_expected += dataset_info['expected_samples']
            
            if dataset_info['type'] == 'git':
                success = self.download_git_dataset(
                    dataset_info['name'],
                    dataset_info['url'],
                    dataset_info['target_dir']
                )
            else:
                success = self.download_http_dataset(
                    dataset_info['name'],
                    dataset_info['url'],
                    dataset_info['target_dir']
                )
            
            results[dataset_id] = {
                'success': success,
                'name': dataset_info['name'],
                'expected_samples': dataset_info['expected_samples']
            }
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 DOWNLOAD SUMMARY")
        logger.info("=" * 70)
        
        successful = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        for dataset_id, result in results.items():
            status = "✅" if result['success'] else "❌"
            logger.info(f"{status} {result['name']}: {result['expected_samples']:,} samples")
        
        logger.info(f"\n✅ Successfully downloaded: {successful}/{total} datasets")
        logger.info(f"📊 Total expected samples: {total_expected:,}")
        
        # Check actual files downloaded
        self.verify_downloads()
        
        return results
    
    def verify_downloads(self):
        """Verify downloaded files"""
        logger.info("\n🔍 Verifying downloaded files...")
        
        total_files = 0
        file_types = {}
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(('.json', '.csv', '.evtx', '.log', '.pcap')):
                    total_files += 1
                    ext = Path(file).suffix
                    file_types[ext] = file_types.get(ext, 0) + 1
        
        logger.info(f"   📁 Total relevant files: {total_files}")
        for ext, count in sorted(file_types.items()):
            logger.info(f"      {ext}: {count} files")
        
        if total_files > 0:
            logger.info("   ✅ Downloads verified - files present")
        else:
            logger.warning("   ⚠️  No data files found - may need manual download")
    
    def get_cicids2017_full(self):
        """Download full CICIDS2017 dataset (optional - large)"""
        logger.info("\n📊 CICIDS2017 Full Dataset (Optional)")
        logger.info("   Note: This is a LARGE dataset (~7GB)")
        logger.info("   URL: https://www.unb.ca/cic/datasets/ids-2017.html")
        logger.info("   Manual download recommended due to size")
        logger.info("   Current system already has CICIDS2017 samples")


def main():
    """Main execution"""
    logger.info("🎯 Windows/AD Attack Dataset Downloader")
    logger.info("Target: 11,000+ samples for enterprise XDR training\n")
    
    downloader = WindowsDatasetDownloader()
    
    try:
        results = downloader.download_all()
        
        logger.info("\n✅ Dataset download complete!")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Run: python3 scripts/data-processing/convert_windows_datasets.py")
        logger.info("   2. This will convert all datasets to Mini-XDR format")
        logger.info("   3. Then merge with existing 4M+ events for training")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Download interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

