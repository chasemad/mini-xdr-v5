"""
Attribution & Campaign Tracker Agent
AI-powered threat actor attribution and campaign tracking
"""
import asyncio
import aiohttp
import json
import hashlib
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import dns.resolver
import ipaddress
import re

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    ChatOpenAI = None

from ..models import Event, Incident, ThreatIntelSource
from ..external_intel import ThreatIntelligence
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ThreatActor:
    """Threat actor profile"""
    actor_id: str
    name: str
    aliases: List[str]
    ttps: List[str]  # Tactics, Techniques, Procedures
    infrastructure: Dict[str, List[str]]  # IPs, domains, ASNs
    malware_families: List[str]
    confidence_score: float
    first_seen: datetime
    last_seen: datetime
    campaign_count: int


@dataclass
class Campaign:
    """Attack campaign tracking"""
    campaign_id: str
    name: str
    attribution: Optional[str]  # Threat actor
    start_date: datetime
    end_date: Optional[datetime]
    targets: List[str]
    ttps: List[str]
    infrastructure: Dict[str, List[str]]
    indicators: Dict[str, List[str]]  # IOCs
    confidence_score: float
    incident_count: int
    severity: str


@dataclass
class InfrastructureCluster:
    """Related infrastructure cluster"""
    cluster_id: str
    ips: Set[str]
    domains: Set[str]
    asns: Set[str]
    shared_attributes: Dict[str, Any]
    confidence_score: float
    first_seen: datetime
    last_seen: datetime


class AttributionAgent:
    """AI Agent for threat actor attribution and campaign tracking"""
    
    def __init__(self, threat_intel=None, llm_client=None):
        self.threat_intel = threat_intel or ThreatIntelligence()
        self.llm_client = llm_client or self._init_llm_client()
        self.agent_id = "attribution_tracker_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Attribution data stores
        self.threat_actors: Dict[str, ThreatActor] = {}
        self.campaigns: Dict[str, Campaign] = {}
        self.infrastructure_clusters: Dict[str, InfrastructureCluster] = {}
        
        # MITRE ATT&CK TTP mappings
        self.ttp_patterns = {
            "T1110": {  # Brute Force
                "name": "Brute Force",
                "patterns": ["cowrie.login.failed", "multiple_failed_logins", "password_spray"]
            },
            "T1078": {  # Valid Accounts
                "name": "Valid Accounts",
                "patterns": ["cowrie.login.success", "credential_reuse"]
            },
            "T1105": {  # Ingress Tool Transfer
                "name": "Ingress Tool Transfer", 
                "patterns": ["cowrie.session.file_download", "wget", "curl"]
            },
            "T1059": {  # Command and Scripting Interpreter
                "name": "Command and Scripting Interpreter",
                "patterns": ["cowrie.command.input", "shell_commands"]
            },
            "T1570": {  # Lateral Tool Transfer
                "name": "Lateral Tool Transfer",
                "patterns": ["file_upload", "scp", "sftp"]
            }
        }
        
        # Known threat actor signatures
        self.actor_signatures = {
            "ssh_brute_force_group_1": {
                "name": "SSH Brute Force Group 1",
                "indicators": {
                    "password_patterns": ["123456", "password", "admin", "root"],
                    "username_patterns": ["root", "admin", "user", "test"],
                    "timing_patterns": {"rapid_succession": True, "rate_limit": False}
                },
                "confidence_threshold": 0.7
            },
            "credential_stuffing_group_1": {
                "name": "Credential Stuffing Group 1", 
                "indicators": {
                    "credential_diversity": {"high": True},
                    "geographic_spread": {"worldwide": True},
                    "persistence": {"multi_day": True}
                },
                "confidence_threshold": 0.8
            }
        }
    
    def _init_llm_client(self):
        """Initialize LLM client for attribution analysis"""
        try:
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                if ChatOpenAI:
                    return ChatOpenAI(
                        openai_api_key=settings.openai_api_key,
                        model_name=settings.openai_model,
                        temperature=0.2  # Conservative for attribution
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None
    
    async def analyze_attribution(
        self, 
        incidents: List[Incident], 
        events: List[Event],
        db_session=None
    ) -> Dict[str, Any]:
        """
        Comprehensive attribution analysis for incidents and events
        
        Returns:
            Dict with attribution results and campaign tracking
        """
        try:
            self.logger.info(f"Starting attribution analysis for {len(incidents)} incidents, {len(events)} events")
            
            # Infrastructure analysis
            infrastructure_analysis = await self._analyze_infrastructure(incidents, events)
            
            # TTP analysis
            ttp_analysis = await self._analyze_ttps(events)
            
            # Temporal analysis
            temporal_analysis = await self._analyze_temporal_patterns(incidents, events)
            
            # Campaign correlation
            campaign_analysis = await self._correlate_campaigns(incidents, ttp_analysis, infrastructure_analysis)
            
            # Threat actor attribution
            actor_attribution = await self._attribute_threat_actors(
                incidents, ttp_analysis, infrastructure_analysis, campaign_analysis
            )
            
            # AI-powered attribution enhancement
            ai_attribution = await self._ai_attribution_analysis(
                incidents, events, ttp_analysis, infrastructure_analysis
            )
            
            # Consolidate results
            attribution_results = {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "incidents_analyzed": len(incidents),
                "events_analyzed": len(events),
                "infrastructure_analysis": infrastructure_analysis,
                "ttp_analysis": ttp_analysis,
                "temporal_analysis": temporal_analysis,
                "campaign_analysis": campaign_analysis,
                "actor_attribution": actor_attribution,
                "ai_attribution": ai_attribution,
                "confidence_score": self._calculate_overall_confidence(
                    actor_attribution, campaign_analysis, ai_attribution
                )
            }
            
            # Update tracking data
            await self._update_attribution_tracking(attribution_results)
            
            return attribution_results
            
        except Exception as e:
            self.logger.error(f"Attribution analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "incidents_analyzed": len(incidents),
                "events_analyzed": len(events)
            }
    
    async def analyze_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Analyze IP reputation using multiple threat intelligence sources"""
        try:
            self.logger.info(f"Analyzing IP reputation for: {ip_address}")
            
            reputation_data = {
                "ip_address": ip_address,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "reputation_score": 0,  # 0-100, higher is more malicious
                "threat_categories": [],
                "sources": {}
            }
            
            # Simulate threat intelligence lookups
            # In a real implementation, this would query actual threat intel APIs
            
            # Mock VirusTotal-style response
            reputation_data["sources"]["virustotal"] = {
                "malicious_engines": 2,
                "total_engines": 70,
                "categories": ["malware", "suspicious"],
                "last_analysis": datetime.utcnow().isoformat()
            }
            
            # Mock AbuseIPDB-style response  
            reputation_data["sources"]["abuseipdb"] = {
                "abuse_confidence": 85,
                "country_code": "CN",
                "usage_type": "datacenter",
                "categories": ["ssh_brute_force", "web_attack"]
            }
            
            # Calculate overall reputation score
            if reputation_data["sources"]["virustotal"]["malicious_engines"] > 0:
                reputation_data["reputation_score"] += 40
            
            if reputation_data["sources"]["abuseipdb"]["abuse_confidence"] > 75:
                reputation_data["reputation_score"] += 35
                
            # Add threat categories
            reputation_data["threat_categories"] = list(set(
                reputation_data["sources"]["virustotal"]["categories"] +
                reputation_data["sources"]["abuseipdb"]["categories"]
            ))
            
            # Generate summary
            if reputation_data["reputation_score"] > 70:
                summary = f"High-risk IP with {reputation_data['reputation_score']}% malicious confidence"
            elif reputation_data["reputation_score"] > 40:
                summary = f"Suspicious IP with {reputation_data['reputation_score']}% malicious confidence"
            else:
                summary = f"Low-risk IP with {reputation_data['reputation_score']}% malicious confidence"
                
            reputation_data["summary"] = summary
            
            self.logger.info(f"IP reputation analysis complete: {summary}")
            return reputation_data
            
        except Exception as e:
            self.logger.error(f"IP reputation analysis failed: {e}")
            return {
                "error": str(e),
                "ip_address": ip_address,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_infrastructure(
        self, 
        incidents: List[Incident], 
        events: List[Event]
    ) -> Dict[str, Any]:
        """Analyze infrastructure relationships and patterns"""
        
        infrastructure_data = {
            "unique_ips": set(),
            "ip_asn_mapping": {},
            "ip_geolocation": {},
            "domain_relationships": {},
            "infrastructure_clusters": [],
            "shared_hosting": []
        }
        
        # Collect unique IPs
        for incident in incidents:
            infrastructure_data["unique_ips"].add(incident.src_ip)
        
        for event in events:
            if event.src_ip:
                infrastructure_data["unique_ips"].add(event.src_ip)
        
        # Analyze each IP
        for ip in infrastructure_data["unique_ips"]:
            try:
                # ASN lookup
                asn_info = await self._lookup_asn(ip)
                if asn_info:
                    infrastructure_data["ip_asn_mapping"][ip] = asn_info
                
                # Geolocation
                geo_info = await self._lookup_geolocation(ip)
                if geo_info:
                    infrastructure_data["ip_geolocation"][ip] = geo_info
                
                # DNS relationships
                dns_info = await self._analyze_dns_relationships(ip)
                if dns_info:
                    infrastructure_data["domain_relationships"][ip] = dns_info
                    
            except Exception as e:
                self.logger.warning(f"Infrastructure analysis failed for {ip}: {e}")
        
        # Identify infrastructure clusters
        clusters = await self._identify_infrastructure_clusters(infrastructure_data)
        infrastructure_data["infrastructure_clusters"] = clusters
        
        # Shared hosting analysis
        shared_hosting = await self._analyze_shared_hosting(infrastructure_data["ip_asn_mapping"])
        infrastructure_data["shared_hosting"] = shared_hosting
        
        return infrastructure_data
    
    async def _lookup_asn(self, ip: str) -> Optional[Dict[str, Any]]:
        """Lookup ASN information for an IP"""
        try:
            # Use a public ASN lookup service
            async with aiohttp.ClientSession() as session:
                url = f"https://api.hackertarget.com/aslookup/?q={ip}"
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        if text and not text.startswith("error"):
                            # Parse ASN response
                            parts = text.strip().split('"')
                            if len(parts) >= 3:
                                asn = parts[0].strip()
                                org = parts[1].strip()
                                return {
                                    "asn": asn,
                                    "organization": org,
                                    "raw_response": text
                                }
        except Exception as e:
            self.logger.debug(f"ASN lookup failed for {ip}: {e}")
        
        return None
    
    async def _lookup_geolocation(self, ip: str) -> Optional[Dict[str, Any]]:
        """Lookup geolocation for an IP"""
        try:
            # Use threat intelligence geolocation if available
            if self.threat_intel:
                intel_result = await self.threat_intel.lookup_ip(ip)
                if intel_result and hasattr(intel_result, 'geolocation'):
                    return intel_result.geolocation
        except Exception as e:
            self.logger.debug(f"Geolocation lookup failed for {ip}: {e}")
        
        return None
    
    async def _analyze_dns_relationships(self, ip: str) -> Optional[Dict[str, Any]]:
        """Analyze DNS relationships for an IP"""
        dns_info = {
            "reverse_dns": None,
            "related_domains": []
        }
        
        try:
            # Reverse DNS lookup
            try:
                reverse_dns = dns.resolver.resolve_address(ip)
                if reverse_dns:
                    dns_info["reverse_dns"] = str(reverse_dns[0])
            except:
                pass
            
            # Additional DNS analysis could be added here
            # (passive DNS, certificate transparency, etc.)
            
        except Exception as e:
            self.logger.debug(f"DNS analysis failed for {ip}: {e}")
        
        return dns_info if dns_info["reverse_dns"] or dns_info["related_domains"] else None
    
    async def _identify_infrastructure_clusters(self, infrastructure_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify clusters of related infrastructure"""
        clusters = []
        
        # Group by ASN
        asn_groups = defaultdict(list)
        for ip, asn_info in infrastructure_data["ip_asn_mapping"].items():
            asn = asn_info.get("asn", "unknown")
            asn_groups[asn].append(ip)
        
        # Create clusters for ASNs with multiple IPs
        for asn, ips in asn_groups.items():
            if len(ips) >= 2 and asn != "unknown":
                cluster = {
                    "cluster_type": "asn",
                    "cluster_id": f"asn_{asn}",
                    "ips": ips,
                    "shared_attributes": {
                        "asn": asn,
                        "organization": infrastructure_data["ip_asn_mapping"][ips[0]].get("organization", "")
                    },
                    "confidence_score": 0.8 if len(ips) >= 3 else 0.6
                }
                clusters.append(cluster)
        
        # Group by geographic location
        geo_groups = defaultdict(list)
        for ip, geo_info in infrastructure_data["ip_geolocation"].items():
            if geo_info:
                country = geo_info.get("country", "unknown")
                geo_groups[country].append(ip)
        
        # Add geographic clusters
        for country, ips in geo_groups.items():
            if len(ips) >= 3 and country != "unknown":
                cluster = {
                    "cluster_type": "geographic",
                    "cluster_id": f"geo_{country}",
                    "ips": ips,
                    "shared_attributes": {
                        "country": country
                    },
                    "confidence_score": 0.4  # Geographic clustering is less reliable
                }
                clusters.append(cluster)
        
        return clusters
    
    async def _analyze_shared_hosting(self, asn_mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze shared hosting providers"""
        shared_hosting = []
        
        # Known hosting providers and VPS services
        hosting_keywords = [
            "digitalocean", "amazon", "google", "microsoft", "vultr", 
            "linode", "ovh", "hetzner", "scaleway", "rackspace"
        ]
        
        for ip, asn_info in asn_mapping.items():
            org = asn_info.get("organization", "").lower()
            for keyword in hosting_keywords:
                if keyword in org:
                    shared_hosting.append({
                        "ip": ip,
                        "provider": keyword,
                        "organization": asn_info.get("organization", ""),
                        "asn": asn_info.get("asn", ""),
                        "hosting_type": "cloud_vps"
                    })
                    break
        
        return shared_hosting
    
    async def _analyze_ttps(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze Tactics, Techniques, and Procedures (TTPs)"""
        
        ttp_analysis = {
            "identified_ttps": {},
            "ttp_frequency": Counter(),
            "ttp_timeline": [],
            "mitre_mapping": {},
            "behavioral_patterns": {}
        }
        
        # Analyze events for TTP patterns
        for event in events:
            event_ttps = await self._extract_ttps_from_event(event)
            
            for ttp_id, confidence in event_ttps.items():
                if ttp_id not in ttp_analysis["identified_ttps"]:
                    ttp_analysis["identified_ttps"][ttp_id] = {
                        "name": self.ttp_patterns.get(ttp_id, {}).get("name", ttp_id),
                        "confidence": confidence,
                        "events": [],
                        "first_seen": event.ts,
                        "last_seen": event.ts
                    }
                else:
                    # Update confidence (take maximum)
                    ttp_analysis["identified_ttps"][ttp_id]["confidence"] = max(
                        ttp_analysis["identified_ttps"][ttp_id]["confidence"], confidence
                    )
                    ttp_analysis["identified_ttps"][ttp_id]["last_seen"] = max(
                        ttp_analysis["identified_ttps"][ttp_id]["last_seen"], event.ts
                    )
                
                ttp_analysis["identified_ttps"][ttp_id]["events"].append({
                    "event_id": event.id,
                    "timestamp": event.ts.isoformat(),
                    "src_ip": event.src_ip
                })
                
                ttp_analysis["ttp_frequency"][ttp_id] += 1
                
                # Timeline entry
                ttp_analysis["ttp_timeline"].append({
                    "timestamp": event.ts.isoformat(),
                    "ttp_id": ttp_id,
                    "src_ip": event.src_ip,
                    "confidence": confidence
                })
        
        # Sort timeline
        ttp_analysis["ttp_timeline"].sort(key=lambda x: x["timestamp"])
        
        # MITRE ATT&CK mapping
        for ttp_id in ttp_analysis["identified_ttps"]:
            if ttp_id in self.ttp_patterns:
                ttp_analysis["mitre_mapping"][ttp_id] = self.ttp_patterns[ttp_id]
        
        # Behavioral pattern analysis
        ttp_analysis["behavioral_patterns"] = await self._analyze_behavioral_patterns(
            ttp_analysis["identified_ttps"], ttp_analysis["ttp_timeline"]
        )
        
        return ttp_analysis
    
    async def _extract_ttps_from_event(self, event: Event) -> Dict[str, float]:
        """Extract TTPs from a single event"""
        ttps = {}
        
        # Map event types to TTPs
        if event.eventid == "cowrie.login.failed":
            ttps["T1110"] = 0.8  # Brute Force
        elif event.eventid == "cowrie.login.success":
            ttps["T1078"] = 0.7  # Valid Accounts
        elif event.eventid == "cowrie.session.file_download":
            ttps["T1105"] = 0.9  # Ingress Tool Transfer
        elif event.eventid == "cowrie.command.input":
            ttps["T1059"] = 0.8  # Command and Scripting Interpreter
            
            # Analyze command content for additional TTPs
            if event.raw and isinstance(event.raw, dict):
                command = event.raw.get("input", "").lower()
                
                # Look for lateral movement tools
                if any(tool in command for tool in ["scp", "sftp", "rsync", "ssh"]):
                    ttps["T1570"] = 0.7  # Lateral Tool Transfer
                
                # Look for download tools
                if any(tool in command for tool in ["wget", "curl", "nc", "netcat"]):
                    ttps["T1105"] = 0.8  # Ingress Tool Transfer
        
        return ttps
    
    async def _analyze_behavioral_patterns(
        self, 
        identified_ttps: Dict[str, Any], 
        ttp_timeline: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze behavioral patterns in TTP usage"""
        
        patterns = {
            "ttp_sequences": [],
            "common_combinations": [],
            "timing_patterns": {},
            "sophistication_score": 0.0
        }
        
        # Analyze TTP sequences
        if len(ttp_timeline) >= 2:
            sequences = []
            for i in range(len(ttp_timeline) - 1):
                current_ttp = ttp_timeline[i]["ttp_id"]
                next_ttp = ttp_timeline[i + 1]["ttp_id"]
                time_diff = (
                    datetime.fromisoformat(ttp_timeline[i + 1]["timestamp"]) - 
                    datetime.fromisoformat(ttp_timeline[i]["timestamp"])
                ).total_seconds()
                
                sequences.append({
                    "sequence": [current_ttp, next_ttp],
                    "time_gap": time_diff,
                    "src_ip": ttp_timeline[i]["src_ip"]
                })
            
            patterns["ttp_sequences"] = sequences
        
        # Find common TTP combinations
        ttp_counter = Counter(identified_ttps.keys())
        patterns["common_combinations"] = [
            {"ttp": ttp, "frequency": count} 
            for ttp, count in ttp_counter.most_common(5)
        ]
        
        # Calculate sophistication score
        sophistication_factors = {
            "T1105": 0.3,  # Tool transfer
            "T1570": 0.4,  # Lateral movement
            "T1059": 0.2,  # Command execution
            "T1110": 0.1,  # Brute force (basic)
            "T1078": 0.2   # Valid accounts
        }
        
        total_sophistication = 0.0
        for ttp_id in identified_ttps:
            total_sophistication += sophistication_factors.get(ttp_id, 0.1)
        
        patterns["sophistication_score"] = min(total_sophistication / len(identified_ttps) if identified_ttps else 0.0, 1.0)
        
        return patterns
    
    async def _analyze_temporal_patterns(
        self, 
        incidents: List[Incident], 
        events: List[Event]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in attack activity"""
        
        temporal_analysis = {
            "attack_windows": [],
            "timezone_analysis": {},
            "campaign_duration": None,
            "activity_intensity": {},
            "patterns": []
        }
        
        if not incidents and not events:
            return temporal_analysis
        
        # Combine timestamps
        all_timestamps = []
        for incident in incidents:
            all_timestamps.append(incident.created_at)
        for event in events:
            all_timestamps.append(event.ts)
        
        all_timestamps.sort()
        
        if len(all_timestamps) >= 2:
            # Campaign duration
            temporal_analysis["campaign_duration"] = {
                "start": all_timestamps[0].isoformat(),
                "end": all_timestamps[-1].isoformat(),
                "duration_hours": (all_timestamps[-1] - all_timestamps[0]).total_seconds() / 3600
            }
            
            # Activity intensity analysis
            hourly_activity = defaultdict(int)
            daily_activity = defaultdict(int)
            
            for ts in all_timestamps:
                hour_key = ts.strftime("%H")
                day_key = ts.strftime("%Y-%m-%d")
                
                hourly_activity[hour_key] += 1
                daily_activity[day_key] += 1
            
            temporal_analysis["activity_intensity"] = {
                "hourly": dict(hourly_activity),
                "daily": dict(daily_activity),
                "peak_hour": max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else None,
                "peak_day": max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else None
            }
            
            # Identify patterns
            patterns = []
            
            # Working hours pattern
            business_hours = sum(
                count for hour, count in hourly_activity.items() 
                if 9 <= int(hour) <= 17
            )
            total_activity = sum(hourly_activity.values())
            
            if business_hours / total_activity > 0.7:
                patterns.append({
                    "type": "business_hours",
                    "confidence": business_hours / total_activity,
                    "description": "Activity concentrated during business hours"
                })
            
            # Weekend activity
            weekend_activity = sum(
                count for day, count in daily_activity.items()
                if datetime.strptime(day, "%Y-%m-%d").weekday() >= 5
            )
            
            if weekend_activity / total_activity > 0.3:
                patterns.append({
                    "type": "weekend_activity", 
                    "confidence": weekend_activity / total_activity,
                    "description": "Significant weekend activity detected"
                })
            
            temporal_analysis["patterns"] = patterns
        
        return temporal_analysis
    
    async def _correlate_campaigns(
        self,
        incidents: List[Incident],
        ttp_analysis: Dict[str, Any],
        infrastructure_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Correlate incidents into potential campaigns"""
        
        campaign_analysis = {
            "potential_campaigns": [],
            "correlation_factors": {},
            "confidence_scores": {}
        }
        
        # Group incidents by various factors
        ip_groups = defaultdict(list)
        ttp_groups = defaultdict(list)
        time_groups = defaultdict(list)
        
        for incident in incidents:
            # Group by IP
            ip_groups[incident.src_ip].append(incident)
            
            # Group by time window (6-hour windows)
            time_key = incident.created_at.replace(
                hour=(incident.created_at.hour // 6) * 6, 
                minute=0, 
                second=0, 
                microsecond=0
            )
            time_groups[time_key].append(incident)
        
        # Identify potential campaigns
        campaign_id = 0
        
        # IP-based campaigns
        for src_ip, grouped_incidents in ip_groups.items():
            if len(grouped_incidents) >= 2:
                campaign_id += 1
                
                # Calculate confidence based on various factors
                confidence = 0.6  # Base confidence for same IP
                
                # Time span factor
                time_span = (grouped_incidents[-1].created_at - grouped_incidents[0].created_at).total_seconds() / 3600
                if time_span <= 24:  # Within 24 hours
                    confidence += 0.2
                elif time_span <= 168:  # Within a week
                    confidence += 0.1
                
                # Infrastructure cluster factor
                for cluster in infrastructure_analysis.get("infrastructure_clusters", []):
                    if src_ip in cluster.get("ips", []):
                        confidence += 0.1
                        break
                
                campaign = {
                    "campaign_id": f"campaign_{campaign_id}",
                    "type": "ip_based",
                    "incidents": [inc.id for inc in grouped_incidents],
                    "src_ip": src_ip,
                    "incident_count": len(grouped_incidents),
                    "time_span_hours": time_span,
                    "confidence": min(confidence, 1.0),
                    "correlation_factors": ["same_ip", "temporal_proximity"]
                }
                
                campaign_analysis["potential_campaigns"].append(campaign)
        
        # TTP-based correlation
        ttp_signatures = {}
        for incident in incidents:
            # Create TTP signature for each incident
            incident_ttps = set()
            
            # Find TTPs associated with this incident's IP
            for ttp_id, ttp_data in ttp_analysis.get("identified_ttps", {}).items():
                for event in ttp_data.get("events", []):
                    if event.get("src_ip") == incident.src_ip:
                        incident_ttps.add(ttp_id)
            
            if incident_ttps:
                ttp_signature = frozenset(incident_ttps)
                if ttp_signature not in ttp_signatures:
                    ttp_signatures[ttp_signature] = []
                ttp_signatures[ttp_signature].append(incident)
        
        # Create TTP-based campaigns
        for ttp_signature, grouped_incidents in ttp_signatures.items():
            if len(grouped_incidents) >= 2:
                campaign_id += 1
                
                confidence = 0.7  # Higher confidence for TTP correlation
                
                # Add factors for TTP sophistication
                sophistication = ttp_analysis.get("behavioral_patterns", {}).get("sophistication_score", 0.0)
                confidence += sophistication * 0.2
                
                campaign = {
                    "campaign_id": f"campaign_{campaign_id}",
                    "type": "ttp_based",
                    "incidents": [inc.id for inc in grouped_incidents],
                    "ttp_signature": list(ttp_signature),
                    "incident_count": len(grouped_incidents),
                    "confidence": min(confidence, 1.0),
                    "correlation_factors": ["shared_ttps", "behavioral_similarity"]
                }
                
                campaign_analysis["potential_campaigns"].append(campaign)
        
        return campaign_analysis
    
    async def _attribute_threat_actors(
        self,
        incidents: List[Incident],
        ttp_analysis: Dict[str, Any],
        infrastructure_analysis: Dict[str, Any],
        campaign_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attribute incidents to known or new threat actors"""
        
        attribution_results = {
            "attributed_actors": [],
            "new_actor_clusters": [],
            "attribution_confidence": {},
            "evidence_summary": {}
        }
        
        # Check against known actor signatures
        for signature_id, signature in self.actor_signatures.items():
            confidence = await self._match_actor_signature(
                signature, incidents, ttp_analysis, infrastructure_analysis
            )
            
            if confidence >= signature["confidence_threshold"]:
                attribution_results["attributed_actors"].append({
                    "actor_id": signature_id,
                    "name": signature["name"],
                    "confidence": confidence,
                    "matching_incidents": [inc.id for inc in incidents],
                    "evidence": await self._collect_attribution_evidence(
                        signature, incidents, ttp_analysis
                    )
                })
        
        # Identify potential new actor clusters
        new_clusters = await self._identify_new_actor_clusters(
            incidents, ttp_analysis, infrastructure_analysis, campaign_analysis
        )
        attribution_results["new_actor_clusters"] = new_clusters
        
        return attribution_results
    
    async def _match_actor_signature(
        self,
        signature: Dict[str, Any],
        incidents: List[Incident],
        ttp_analysis: Dict[str, Any],
        infrastructure_analysis: Dict[str, Any]
    ) -> float:
        """Match incidents against a threat actor signature"""
        
        total_score = 0.0
        max_score = 0.0
        
        indicators = signature.get("indicators", {})
        
        # Password pattern matching
        if "password_patterns" in indicators:
            password_score = await self._match_password_patterns(
                indicators["password_patterns"], incidents
            )
            total_score += password_score * 0.3
            max_score += 0.3
        
        # Username pattern matching
        if "username_patterns" in indicators:
            username_score = await self._match_username_patterns(
                indicators["username_patterns"], incidents
            )
            total_score += username_score * 0.2
            max_score += 0.2
        
        # Timing pattern matching
        if "timing_patterns" in indicators:
            timing_score = await self._match_timing_patterns(
                indicators["timing_patterns"], incidents
            )
            total_score += timing_score * 0.2
            max_score += 0.2
        
        # TTP matching
        ttp_score = await self._match_ttp_patterns(signature, ttp_analysis)
        total_score += ttp_score * 0.3
        max_score += 0.3
        
        return total_score / max_score if max_score > 0 else 0.0
    
    async def _match_password_patterns(self, patterns: List[str], incidents: List[Incident]) -> float:
        """Match password patterns against incidents"""
        # This would analyze actual password attempts from related events
        # For now, return a placeholder score
        return 0.5
    
    async def _match_username_patterns(self, patterns: List[str], incidents: List[Incident]) -> float:
        """Match username patterns against incidents"""
        # This would analyze actual username attempts from related events
        # For now, return a placeholder score
        return 0.5
    
    async def _match_timing_patterns(self, patterns: Dict[str, Any], incidents: List[Incident]) -> float:
        """Match timing patterns against incidents"""
        # This would analyze timing characteristics
        # For now, return a placeholder score
        return 0.5
    
    async def _match_ttp_patterns(self, signature: Dict[str, Any], ttp_analysis: Dict[str, Any]) -> float:
        """Match TTP patterns against signature"""
        # This would match TTPs against known actor patterns
        # For now, return a placeholder score
        return 0.5
    
    async def _collect_attribution_evidence(
        self,
        signature: Dict[str, Any],
        incidents: List[Incident],
        ttp_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect evidence supporting actor attribution"""
        
        evidence = {
            "indicators_matched": [],
            "ttp_overlap": [],
            "infrastructure_overlap": [],
            "temporal_correlation": {},
            "confidence_factors": []
        }
        
        # This would collect specific evidence
        # For now, return placeholder evidence
        
        return evidence
    
    async def _identify_new_actor_clusters(
        self,
        incidents: List[Incident],
        ttp_analysis: Dict[str, Any],
        infrastructure_analysis: Dict[str, Any],
        campaign_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential new threat actor clusters"""
        
        clusters = []
        
        # This would use clustering algorithms to identify new patterns
        # For now, return basic clustering based on campaigns
        
        for campaign in campaign_analysis.get("potential_campaigns", []):
            if campaign["confidence"] > 0.8 and campaign["incident_count"] >= 3:
                cluster = {
                    "cluster_id": f"new_actor_{campaign['campaign_id']}",
                    "incidents": campaign["incidents"],
                    "confidence": campaign["confidence"],
                    "characteristics": {
                        "campaign_type": campaign["type"],
                        "incident_count": campaign["incident_count"]
                    }
                }
                clusters.append(cluster)
        
        return clusters
    
    async def _ai_attribution_analysis(
        self,
        incidents: List[Incident],
        events: List[Event],
        ttp_analysis: Dict[str, Any],
        infrastructure_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use AI to enhance attribution analysis"""
        
        if not self.llm_client:
            return {"ai_enabled": False}
        
        # Prepare context for AI analysis
        context = {
            "incident_count": len(incidents),
            "event_count": len(events),
            "unique_ips": len(infrastructure_analysis.get("unique_ips", set())),
            "identified_ttps": list(ttp_analysis.get("identified_ttps", {}).keys()),
            "infrastructure_clusters": len(infrastructure_analysis.get("infrastructure_clusters", [])),
            "behavioral_sophistication": ttp_analysis.get("behavioral_patterns", {}).get("sophistication_score", 0.0)
        }
        
        prompt = f"""
        You are a threat intelligence analyst specializing in threat actor attribution.
        
        ANALYSIS DATA:
        {json.dumps(context, indent=2)}
        
        Based on this data, provide attribution insights:
        
        1. Likely threat actor type (nation-state, cybercriminal, hacktivist, script kiddie)
        2. Sophistication level assessment
        3. Potential motivations
        4. Geographic attribution indicators
        5. Campaign assessment
        
        Provide analysis in JSON format:
        {{
            "actor_type": "cybercriminal|nation-state|hacktivist|script-kiddie|unknown",
            "sophistication_level": "low|medium|high|advanced",
            "confidence": 0.75,
            "motivations": ["financial", "espionage", "disruption"],
            "geographic_indicators": ["country1", "country2"],
            "campaign_assessment": "single|multi-phase|ongoing|concluded",
            "reasoning": "detailed explanation of attribution",
            "recommendations": ["action1", "action2"]
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_client.invoke(prompt)
            )
            
            # Parse AI response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
                ai_result["ai_enabled"] = True
                return ai_result
                
        except Exception as e:
            self.logger.error(f"AI attribution analysis failed: {e}")
        
        return {"ai_enabled": False, "error": "AI analysis failed"}
    
    def _calculate_overall_confidence(
        self,
        actor_attribution: Dict[str, Any],
        campaign_analysis: Dict[str, Any],
        ai_attribution: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in attribution analysis"""
        
        confidence_factors = []
        
        # Actor attribution confidence
        if actor_attribution.get("attributed_actors"):
            max_actor_confidence = max(
                actor["confidence"] for actor in actor_attribution["attributed_actors"]
            )
            confidence_factors.append(max_actor_confidence * 0.4)
        
        # Campaign correlation confidence
        if campaign_analysis.get("potential_campaigns"):
            max_campaign_confidence = max(
                campaign["confidence"] for campaign in campaign_analysis["potential_campaigns"]
            )
            confidence_factors.append(max_campaign_confidence * 0.3)
        
        # AI attribution confidence
        if ai_attribution.get("ai_enabled") and "confidence" in ai_attribution:
            confidence_factors.append(ai_attribution["confidence"] * 0.3)
        
        return sum(confidence_factors) if confidence_factors else 0.0
    
    async def _update_attribution_tracking(self, attribution_results: Dict[str, Any]):
        """Update persistent attribution tracking data"""
        
        # Update threat actor tracking
        for actor in attribution_results.get("actor_attribution", {}).get("attributed_actors", []):
            actor_id = actor["actor_id"]
            
            if actor_id not in self.threat_actors:
                self.threat_actors[actor_id] = ThreatActor(
                    actor_id=actor_id,
                    name=actor["name"],
                    aliases=[],
                    ttps=[],
                    infrastructure={},
                    malware_families=[],
                    confidence_score=actor["confidence"],
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    campaign_count=1
                )
            else:
                # Update existing actor
                self.threat_actors[actor_id].last_seen = datetime.utcnow()
                self.threat_actors[actor_id].campaign_count += 1
                self.threat_actors[actor_id].confidence_score = max(
                    self.threat_actors[actor_id].confidence_score,
                    actor["confidence"]
                )
        
        # Update campaign tracking
        for campaign in attribution_results.get("campaign_analysis", {}).get("potential_campaigns", []):
            campaign_id = campaign["campaign_id"]
            
            if campaign_id not in self.campaigns:
                self.campaigns[campaign_id] = Campaign(
                    campaign_id=campaign_id,
                    name=f"Campaign {campaign_id}",
                    attribution=None,
                    start_date=datetime.utcnow(),
                    end_date=None,
                    targets=[],
                    ttps=[],
                    infrastructure={},
                    indicators={},
                    confidence_score=campaign["confidence"],
                    incident_count=campaign["incident_count"],
                    severity="medium"
                )
        
        self.logger.info(f"Updated attribution tracking: {len(self.threat_actors)} actors, {len(self.campaigns)} campaigns")
    
    async def get_attribution_summary(self) -> Dict[str, Any]:
        """Get summary of attribution tracking data"""
        
        summary = {
            "threat_actors": {
                "total_count": len(self.threat_actors),
                "active_actors": [
                    {
                        "actor_id": actor.actor_id,
                        "name": actor.name,
                        "confidence": actor.confidence_score,
                        "campaign_count": actor.campaign_count,
                        "last_seen": actor.last_seen.isoformat()
                    }
                    for actor in self.threat_actors.values()
                ]
            },
            "campaigns": {
                "total_count": len(self.campaigns),
                "active_campaigns": [
                    {
                        "campaign_id": campaign.campaign_id,
                        "name": campaign.name,
                        "confidence": campaign.confidence_score,
                        "incident_count": campaign.incident_count,
                        "severity": campaign.severity
                    }
                    for campaign in self.campaigns.values()
                ]
            },
            "infrastructure_clusters": {
                "total_count": len(self.infrastructure_clusters)
            }
        }
        
        return summary
