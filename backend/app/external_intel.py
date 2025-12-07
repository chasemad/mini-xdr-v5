"""
External Threat Intelligence Integration
Supports multiple threat intelligence sources with caching and rate limiting
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

from .config import settings
from .models import ThreatIntelSource  # This imports from the main models.py file

logger = logging.getLogger(__name__)


@dataclass
class ThreatIntelResult:
    """Result from threat intelligence lookup"""

    ip: str
    source: str
    risk_score: float  # 0.0 to 1.0
    category: str
    confidence: float
    last_seen: Optional[str] = None
    country: Optional[str] = None
    asn: Optional[str] = None
    is_malicious: bool = False
    is_tor: bool = False
    is_vpn: bool = False
    raw_data: Optional[Dict] = None
    cached: bool = False
    lookup_time: float = 0.0


class ThreatIntelligenceCache:
    """Simple in-memory cache for threat intel results"""

    def __init__(self, max_age_hours: int = 24, max_entries: int = 10000):
        self.cache = {}
        self.max_age = max_age_hours * 3600  # Convert to seconds
        self.max_entries = max_entries

    def get(self, key: str) -> Optional[ThreatIntelResult]:
        """Get cached result if not expired"""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.max_age:
                result.cached = True
                return result
            else:
                del self.cache[key]
        return None

    def set(self, key: str, result: ThreatIntelResult):
        """Cache a result"""
        # Implement simple LRU by removing oldest entries
        if len(self.cache) >= self.max_entries:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (result, time.time())

    def clear(self):
        """Clear the cache"""
        self.cache.clear()


class AbuseIPDBProvider:
    """AbuseIPDB threat intelligence provider"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.abuseipdb.com/api/v2"
        self.rate_limit = 1000  # requests per day
        self.requests_made = 0
        self.rate_reset_time = time.time() + 86400  # 24 hours

    async def lookup_ip(
        self, ip: str, session: aiohttp.ClientSession
    ) -> Optional[ThreatIntelResult]:
        """Look up IP in AbuseIPDB"""
        if not self._check_rate_limit():
            logger.warning("AbuseIPDB rate limit exceeded")
            return None

        try:
            headers = {"Key": self.api_key, "Accept": "application/json"}

            params = {"ipAddress": ip, "maxAgeInDays": 90, "verbose": ""}

            start_time = time.time()

            async with session.get(
                f"{self.base_url}/check",
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                self.requests_made += 1

                if response.status == 200:
                    data = await response.json()
                    return self._parse_abuseipdb_response(
                        ip, data, time.time() - start_time
                    )
                elif response.status == 429:
                    logger.warning("AbuseIPDB rate limited")
                    return None
                else:
                    logger.error(f"AbuseIPDB API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"AbuseIPDB lookup error for {ip}: {e}")
            return None

    def _parse_abuseipdb_response(
        self, ip: str, data: Dict, lookup_time: float
    ) -> ThreatIntelResult:
        """Parse AbuseIPDB API response"""
        abuse_confidence = data.get("abuseConfidencePercentage", 0)

        # Convert percentage to 0-1 score
        risk_score = abuse_confidence / 100.0

        # Determine category
        if abuse_confidence >= 75:
            category = "malicious"
        elif abuse_confidence >= 25:
            category = "suspicious"
        else:
            category = "benign"

        return ThreatIntelResult(
            ip=ip,
            source="abuseipdb",
            risk_score=risk_score,
            category=category,
            confidence=risk_score,
            last_seen=data.get("lastReportedAt"),
            country=data.get("countryCode"),
            is_malicious=abuse_confidence >= 50,
            is_tor=data.get("isTor", False),
            raw_data=data,
            lookup_time=lookup_time,
        )

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()

        # Reset counter if 24 hours have passed
        if current_time > self.rate_reset_time:
            self.requests_made = 0
            self.rate_reset_time = current_time + 86400

        return self.requests_made < self.rate_limit


class VirusTotalProvider:
    """VirusTotal threat intelligence provider"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.virustotal.com/api/v3"
        self.rate_limit = 500  # requests per day for free tier
        self.requests_made = 0
        self.rate_reset_time = time.time() + 86400

    async def lookup_ip(
        self, ip: str, session: aiohttp.ClientSession
    ) -> Optional[ThreatIntelResult]:
        """Look up IP in VirusTotal"""
        if not self._check_rate_limit():
            logger.warning("VirusTotal rate limit exceeded")
            return None

        try:
            headers = {"x-apikey": self.api_key}

            start_time = time.time()

            async with session.get(
                f"{self.base_url}/ip_addresses/{ip}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                self.requests_made += 1

                if response.status == 200:
                    data = await response.json()
                    return self._parse_virustotal_response(
                        ip, data, time.time() - start_time
                    )
                elif response.status == 429:
                    logger.warning("VirusTotal rate limited")
                    return None
                else:
                    logger.error(f"VirusTotal API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"VirusTotal lookup error for {ip}: {e}")
            return None

    def _parse_virustotal_response(
        self, ip: str, data: Dict, lookup_time: float
    ) -> ThreatIntelResult:
        """Parse VirusTotal API response"""
        attributes = data.get("data", {}).get("attributes", {})
        stats = attributes.get("last_analysis_stats", {})

        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        harmless = stats.get("harmless", 0)
        total = malicious + suspicious + harmless

        if total > 0:
            risk_score = (malicious + suspicious * 0.5) / total
        else:
            risk_score = 0.0

        # Determine category
        if malicious >= 3:
            category = "malicious"
        elif malicious >= 1 or suspicious >= 5:
            category = "suspicious"
        else:
            category = "benign"

        return ThreatIntelResult(
            ip=ip,
            source="virustotal",
            risk_score=risk_score,
            category=category,
            confidence=min(total / 10.0, 1.0),  # More engines = higher confidence
            country=attributes.get("country"),
            asn=str(attributes.get("asn", "")),
            is_malicious=malicious >= 2,
            raw_data=data,
            lookup_time=lookup_time,
        )

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()

        if current_time > self.rate_reset_time:
            self.requests_made = 0
            self.rate_reset_time = current_time + 86400

        return self.requests_made < self.rate_limit


class ThreatIntelligence:
    """Main threat intelligence aggregator"""

    def __init__(self):
        self.providers = {}
        self.cache = ThreatIntelligenceCache()
        self.session = None
        self.logger = logging.getLogger(__name__)

        # Initialize providers from environment
        self._init_providers()

    def _init_providers(self):
        """Initialize threat intelligence providers"""
        # AbuseIPDB
        abuseipdb_key = getattr(settings, "abuseipdb_api_key", None)
        if abuseipdb_key:
            self.providers["abuseipdb"] = AbuseIPDBProvider(abuseipdb_key)
            self.logger.info("AbuseIPDB provider initialized")

        # VirusTotal
        virustotal_key = getattr(settings, "virustotal_api_key", None)
        if virustotal_key:
            self.providers["virustotal"] = VirusTotalProvider(virustotal_key)
            self.logger.info("VirusTotal provider initialized")

        if not self.providers:
            self.logger.warning("No threat intelligence providers configured")

    def reinitialize_providers(self):
        """Reinitialize providers (useful after secrets are loaded)"""
        self.providers.clear()
        self._init_providers()

    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def lookup_ip(
        self, ip: str, sources: Optional[List[str]] = None
    ) -> ThreatIntelResult:
        """
        Look up IP in threat intelligence sources

        Args:
            ip: IP address to look up
            sources: List of specific sources to query (default: all)

        Returns:
            Aggregated threat intelligence result
        """
        # Check cache first
        cache_key = f"intel_{ip}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result

        await self._ensure_session()

        # Determine which sources to query
        if sources is None:
            sources = list(self.providers.keys())

        results = []

        # Query each provider
        for source in sources:
            if source in self.providers:
                try:
                    result = await self.providers[source].lookup_ip(ip, self.session)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error querying {source}: {e}")

        # Aggregate results
        if results:
            aggregated = self._aggregate_results(ip, results)
        else:
            # Return default result if no sources available
            aggregated = ThreatIntelResult(
                ip=ip, source="none", risk_score=0.0, category="unknown", confidence=0.0
            )

        # Cache the result
        self.cache.set(cache_key, aggregated)

        return aggregated

    def _aggregate_results(
        self, ip: str, results: List[ThreatIntelResult]
    ) -> ThreatIntelResult:
        """Aggregate results from multiple sources"""
        if len(results) == 1:
            return results[0]

        # Calculate weighted average risk score
        total_weight = 0
        weighted_risk = 0

        source_weights = {"virustotal": 0.6, "abuseipdb": 0.4}

        for result in results:
            weight = source_weights.get(result.source, 0.3)
            # Weight by confidence
            adjusted_weight = weight * result.confidence
            weighted_risk += result.risk_score * adjusted_weight
            total_weight += adjusted_weight

        avg_risk = weighted_risk / max(total_weight, 0.1)

        # Determine category based on risk score
        if avg_risk >= 0.7:
            category = "malicious"
        elif avg_risk >= 0.3:
            category = "suspicious"
        else:
            category = "benign"

        # Aggregate other fields
        sources = [r.source for r in results]
        is_malicious = any(r.is_malicious for r in results)
        is_tor = any(r.is_tor for r in results)

        # Use the result with highest confidence for other fields
        best_result = max(results, key=lambda r: r.confidence)

        return ThreatIntelResult(
            ip=ip,
            source=",".join(sources),
            risk_score=avg_risk,
            category=category,
            confidence=total_weight / len(results),
            last_seen=best_result.last_seen,
            country=best_result.country,
            asn=best_result.asn,
            is_malicious=is_malicious,
            is_tor=is_tor,
            is_vpn=best_result.is_vpn,
            raw_data={"aggregated": True, "sources": [asdict(r) for r in results]},
            lookup_time=sum(r.lookup_time for r in results),
        )

    async def bulk_lookup(
        self, ips: List[str], max_concurrent: int = 5
    ) -> Dict[str, ThreatIntelResult]:
        """Look up multiple IPs concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def lookup_with_semaphore(ip):
            async with semaphore:
                return ip, await self.lookup_ip(ip)

        tasks = [lookup_with_semaphore(ip) for ip in ips]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            ip: result for ip, result in results if not isinstance(result, Exception)
        }

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def clear_cache(self):
        """Clear the threat intelligence cache"""
        self.cache.clear()
        self.logger.info("Threat intelligence cache cleared")


# Global threat intelligence instance
threat_intel = ThreatIntelligence()


# Configuration helper for settings
def configure_threat_intel_sources():
    """Add threat intel API keys to settings if not present"""
    # This would be called during app startup to check for API keys
    if not hasattr(settings, "abuseipdb_api_key"):
        settings.abuseipdb_api_key = None
    if not hasattr(settings, "virustotal_api_key"):
        settings.virustotal_api_key = None
