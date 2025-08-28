#!/usr/bin/env python3
"""
Historical Data Importer for Adaptive Detection Training
Imports existing log files and honeypot data for ML training
"""
import json
import requests
import os
import gzip
import re
from datetime import datetime, timedelta
from typing import List, Dict, Iterator
import argparse

BASE_URL = "http://localhost:8000"
API_KEY = "test-api-key"

class HistoricalDataImporter:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        })
    
    def parse_cowrie_log(self, log_file: str) -> Iterator[Dict]:
        """Parse Cowrie honeypot log files"""
        print(f"📖 Parsing Cowrie log: {log_file}")
        
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        # Cowrie logs are typically JSON
                        log_entry = json.loads(line.strip())
                        
                        # Convert to our event format
                        event = {
                            "eventid": log_entry.get("eventid", "cowrie.unknown"),
                            "src_ip": log_entry.get("src_ip"),
                            "dst_port": log_entry.get("dst_port", 2222),
                            "message": log_entry.get("message", ""),
                            "timestamp": log_entry.get("timestamp"),
                            "raw": log_entry
                        }
                        
                        if event["src_ip"]:  # Only yield if we have a source IP
                            yield event
                            
                except (json.JSONDecodeError, KeyError) as e:
                    if line_num % 100 == 0:  # Log every 100th error to avoid spam
                        print(f"   ⚠️ Line {line_num}: Parse error - {e}")
                    continue
    
    def parse_generic_log(self, log_file: str, log_format: str = "auto") -> Iterator[Dict]:
        """Parse generic log files (Apache, Nginx, etc.)"""
        print(f"📖 Parsing generic log: {log_file}")
        
        # Common log patterns
        patterns = {
            "apache_common": r'(\S+) \S+ \S+ \[(.*?)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)',
            "nginx": r'(\S+) - - \[(.*?)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "(.*?)" "(.*?)"',
            "ssh": r'(\w+\s+\d+\s+\d+:\d+:\d+).*?sshd.*?Failed password for (\w+) from (\S+)',
        }
        
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Auto-detect format or use specified
                    if log_format == "auto":
                        for fmt_name, pattern in patterns.items():
                            match = re.search(pattern, line)
                            if match:
                                event = self._convert_log_match(match, fmt_name)
                                if event:
                                    yield event
                                break
                    else:
                        if log_format in patterns:
                            match = re.search(patterns[log_format], line)
                            if match:
                                event = self._convert_log_match(match, log_format)
                                if event:
                                    yield event
                
                except Exception as e:
                    if line_num % 100 == 0:
                        print(f"   ⚠️ Line {line_num}: Parse error - {e}")
                    continue
    
    def _convert_log_match(self, match, format_type: str) -> Dict:
        """Convert regex match to event format"""
        try:
            if format_type in ["apache_common", "nginx"]:
                ip, timestamp, method, path, protocol, status, size = match.groups()[:7]
                
                # Detect attack indicators
                attack_indicators = []
                path_lower = path.lower()
                
                if any(admin_path in path_lower for admin_path in ['/admin', '/wp-admin', '/phpmyadmin']):
                    attack_indicators.append("admin_scan")
                if any(sqli in path_lower for sqli in ["'", "union", "select", "drop"]):
                    attack_indicators.append("sql_injection")
                if any(path_trav in path for path_trav in ["../", "..\\", ".env"]):
                    attack_indicators.append("path_traversal")
                
                return {
                    "eventid": "webhoneypot.request",
                    "src_ip": ip,
                    "dst_port": 80,
                    "message": f"{method} {path}",
                    "timestamp": timestamp,
                    "raw": {
                        "method": method,
                        "path": path,
                        "status_code": int(status),
                        "response_size": int(size) if size.isdigit() else 0,
                        "attack_indicators": attack_indicators
                    }
                }
            
            elif format_type == "ssh":
                timestamp, username, ip = match.groups()
                return {
                    "eventid": "cowrie.login.failed",
                    "src_ip": ip,
                    "dst_port": 22,
                    "message": f"SSH login failed: {username}",
                    "timestamp": timestamp,
                    "raw": {
                        "username": username,
                        "password": "unknown"
                    }
                }
        
        except Exception:
            return None
        
        return None
    
    async def import_from_directory(self, directory: str, file_pattern: str = "*.log*") -> Dict:
        """Import all log files from a directory"""
        import glob
        
        print(f"📁 Scanning directory: {directory}")
        pattern = os.path.join(directory, file_pattern)
        log_files = glob.glob(pattern)
        
        if not log_files:
            print(f"   ⚠️ No files found matching pattern: {file_pattern}")
            return {"error": "No files found"}
        
        print(f"   📄 Found {len(log_files)} log files")
        
        total_events = 0
        total_incidents = 0
        results = []
        
        for log_file in log_files:
            print(f"\n🔄 Processing: {os.path.basename(log_file)}")
            result = await self.import_single_file(log_file)
            
            if "error" not in result:
                events = result.get("total_events", 0)
                incidents = result.get("incidents_detected", 0)
                total_events += events
                total_incidents += incidents
                
                print(f"   ✅ {events} events, {incidents} incidents")
            else:
                print(f"   ❌ {result['error']}")
            
            results.append(result)
        
        return {
            "files_processed": len(log_files),
            "total_events": total_events,
            "total_incidents": total_incidents,
            "results": results
        }
    
    async def import_single_file(self, log_file: str, batch_size: int = 50) -> Dict:
        """Import a single log file"""
        if not os.path.exists(log_file):
            return {"error": f"File not found: {log_file}"}
        
        # Determine parser based on file name/content
        if "cowrie" in log_file.lower() or log_file.endswith(".json"):
            events_iter = self.parse_cowrie_log(log_file)
            source_type = "cowrie"
        else:
            events_iter = self.parse_generic_log(log_file)
            source_type = "imported_logs"
        
        # Process in batches
        batch = []
        total_events = 0
        total_incidents = 0
        
        for event in events_iter:
            batch.append(event)
            total_events += 1
            
            if len(batch) >= batch_size:
                # Send batch
                result = await self._send_batch(batch, source_type)
                if "incidents_detected" in result:
                    total_incidents += result["incidents_detected"]
                
                batch = []
                
                # Progress indicator
                if total_events % 500 == 0:
                    print(f"   📊 Processed {total_events} events...")
        
        # Send remaining events
        if batch:
            result = await self._send_batch(batch, source_type)
            if "incidents_detected" in result:
                total_incidents += result["incidents_detected"]
        
        return {
            "file": log_file,
            "total_events": total_events,
            "incidents_detected": total_incidents
        }
    
    async def _send_batch(self, events: List[Dict], source_type: str) -> Dict:
        """Send a batch of events to the API"""
        payload = {
            "source_type": source_type,
            "hostname": "historical-import",
            "events": events
        }
        
        try:
            response = self.session.post(f"{self.base_url}/ingest/multi", json=payload)
            return response.json() if response.status_code == 200 else {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

async def main():
    parser = argparse.ArgumentParser(description="Import historical data for training")
    parser.add_argument("--source", required=True, help="Log file or directory to import")
    parser.add_argument("--type", choices=["auto", "cowrie", "apache", "nginx", "ssh"],
                       default="auto", help="Log format type")
    parser.add_argument("--pattern", default="*.log*", help="File pattern for directory import")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    
    args = parser.parse_args()
    
    importer = HistoricalDataImporter()
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not healthy. Please start Mini-XDR first.")
            return
    except Exception:
        print("❌ Cannot connect to backend. Please start Mini-XDR first.")
        return
    
    print("📚 Historical Data Importer for Adaptive Detection")
    print("=" * 50)
    
    if os.path.isdir(args.source):
        print(f"📁 Importing from directory: {args.source}")
        result = await importer.import_from_directory(args.source, args.pattern)
    else:
        print(f"📄 Importing file: {args.source}")
        result = await importer.import_single_file(args.source, args.batch_size)
    
    print("\n" + "=" * 50)
    if "error" in result:
        print(f"❌ Import failed: {result['error']}")
    else:
        print("✅ Import completed successfully!")
        print(f"   📊 Total events: {result.get('total_events', 0)}")
        print(f"   🚨 Incidents detected: {result.get('total_incidents', 0)}")
        
        if 'files_processed' in result:
            print(f"   📁 Files processed: {result['files_processed']}")
    
    # Trigger learning update
    print("\n🔄 Triggering learning pipeline update...")
    try:
        response = requests.post(f"{BASE_URL}/api/adaptive/force_learning")
        if response.status_code == 200:
            learning_result = response.json()
            print("✅ Learning pipeline updated!")
            print(f"   Results: {learning_result.get('results', {})}")
    except Exception as e:
        print(f"⚠️ Learning update failed: {e}")
    
    print("\n🎯 Next Steps:")
    print("   • Check status: python scripts/optimize-training.py --mode status")
    print("   • Generate more data: python scripts/generate-training-data.py")
    print("   • Test detection: ./scripts/test-adaptive-detection.sh")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
