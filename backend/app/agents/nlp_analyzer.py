"""
Natural Language Threat Analysis Agent for Mini-XDR
Enables analysts to query incidents using natural language with advanced LLM integration
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LANGCHAIN_AVAILABLE = True

from ..config import settings
from ..models import Action, Event, Incident

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of natural language queries supported"""

    INCIDENT_SEARCH = "incident_search"
    THREAT_ANALYSIS = "threat_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    TIMELINE_ANALYSIS = "timeline_analysis"
    IOC_EXTRACTION = "ioc_extraction"
    ATTRIBUTION_QUERY = "attribution_query"
    RECOMMENDATION = "recommendation"


@dataclass
class QueryContext:
    """Context for natural language queries"""

    query_type: QueryType
    confidence_score: float
    extracted_entities: Dict[str, List[str]]
    time_constraints: Optional[Dict[str, Any]]
    threat_categories: List[str]
    priority_level: str


@dataclass
class NLPResponse:
    """Response from NLP analysis"""

    query_understanding: str
    structured_query: Dict[str, Any]
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_score: float
    reasoning: str
    follow_up_questions: List[str]


class ThreatQueryParser:
    """Parser for natural language threat queries"""

    def __init__(self):
        # Threat-related keyword patterns
        self.threat_patterns = {
            "attack_types": [
                r"\b(?:brute\s*force|bruteforce)\b",
                r"\b(?:malware|virus|trojan|ransomware)\b",
                r"\b(?:phishing|spear\s*phishing)\b",
                r"\b(?:lateral\s*movement|privilege\s*escalation)\b",
                r"\b(?:data\s*exfil|exfiltration|data\s*theft)\b",
                r"\b(?:ddos|denial\s*of\s*service)\b",
                r"\b(?:reconnaissance|recon|scanning)\b",
                r"\b(?:c2|command\s*and\s*control)\b",
            ],
            "indicators": [
                r"\b(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",  # IP addresses
                r"\b(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b",  # Hashes
                r"\b(?:https?://[^\s]+)\b",  # URLs
                r"\b(?:[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",  # Domains
            ],
            "timeframes": [
                r"\b(?:last|past)\s+(\d+)\s+(hour|hours|day|days|week|weeks|month|months)\b",
                r"\b(?:today|yesterday|this\s+week|last\s+week)\b",
                r"\b(?:since|from|after)\s+([^\s]+)\b",
                r"\b(?:between)\s+([^\s]+)\s+(?:and)\s+([^\s]+)\b",
            ],
            "severity_levels": [
                r"\b(?:critical|high|medium|low)\s*(?:severity|priority)\b",
                r"\b(?:urgent|emergency|immediate)\b",
            ],
            "geolocation": [
                r"\bfrom\s+([A-Za-z\s]+)(?:\s+country|\s+region)?\b",
                r"\b(?:chinese|russian|iranian|north\s*korean|eastern\s*european)\b",
                r"\b(?:asia|europe|north\s*america|south\s*america|africa|oceania)\b",
            ],
        }

    def parse_query(self, query: str) -> QueryContext:
        """Parse natural language query into structured context"""
        query_lower = query.lower()

        # Determine query type
        query_type = self._classify_query_type(query_lower)

        # Extract entities
        entities = self._extract_entities(query)

        # Extract time constraints
        time_constraints = self._extract_time_constraints(query)

        # Extract threat categories
        threat_categories = self._extract_threat_categories(query_lower)

        # Determine priority
        priority_level = self._extract_priority(query_lower)

        # Calculate confidence based on extracted information
        confidence_score = self._calculate_confidence(
            entities, time_constraints, threat_categories
        )

        return QueryContext(
            query_type=query_type,
            confidence_score=confidence_score,
            extracted_entities=entities,
            time_constraints=time_constraints,
            threat_categories=threat_categories,
            priority_level=priority_level,
        )

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        if any(term in query for term in ["show", "list", "find", "get", "search"]):
            return QueryType.INCIDENT_SEARCH
        elif any(term in query for term in ["analyze", "analysis", "examine"]):
            return QueryType.THREAT_ANALYSIS
        elif any(term in query for term in ["pattern", "similar", "related"]):
            return QueryType.PATTERN_RECOGNITION
        elif any(term in query for term in ["timeline", "sequence", "chronological"]):
            return QueryType.TIMELINE_ANALYSIS
        elif any(term in query for term in ["indicator", "ioc", "hash", "domain"]):
            return QueryType.IOC_EXTRACTION
        elif any(term in query for term in ["who", "actor", "group", "attribution"]):
            return QueryType.ATTRIBUTION_QUERY
        elif any(term in query for term in ["recommend", "suggest", "should", "next"]):
            return QueryType.RECOMMENDATION
        else:
            return QueryType.INCIDENT_SEARCH  # Default

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract various entities from the query"""
        entities = {
            "ips": [],
            "domains": [],
            "hashes": [],
            "urls": [],
            "attack_types": [],
            "countries": [],
        }

        # Extract IPs
        ip_pattern = r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
        entities["ips"] = re.findall(ip_pattern, query)

        # Extract attack types
        for pattern in self.threat_patterns["attack_types"]:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["attack_types"].extend(matches)

        # Extract indicators
        for pattern in self.threat_patterns["indicators"]:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if "." in pattern and "http" not in pattern:
                entities["domains"].extend(matches)
            elif "http" in pattern:
                entities["urls"].extend(matches)
            elif any(c in pattern for c in ["[a-fA-F0-9]"]):
                entities["hashes"].extend(matches)

        return entities

    def _extract_time_constraints(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract time-related constraints"""
        time_info = {}

        # Look for relative time expressions
        relative_pattern = (
            r"\b(?:last|past)\s+(\d+)\s+(hour|hours|day|days|week|weeks|month|months)\b"
        )
        relative_match = re.search(relative_pattern, query, re.IGNORECASE)

        if relative_match:
            number = int(relative_match.group(1))
            unit = relative_match.group(2).lower()

            # Convert to hours for standardization
            multiplier = {
                "hour": 1,
                "hours": 1,
                "day": 24,
                "days": 24,
                "week": 168,
                "weeks": 168,
                "month": 720,
                "months": 720,
            }

            hours_back = number * multiplier.get(unit, 24)
            time_info = {
                "type": "relative",
                "hours_back": hours_back,
                "description": f"Last {number} {unit}",
            }

        # Look for absolute time expressions
        elif any(term in query.lower() for term in ["today", "yesterday", "this week"]):
            if "today" in query.lower():
                time_info = {
                    "type": "relative",
                    "hours_back": 24,
                    "description": "Today",
                }
            elif "yesterday" in query.lower():
                time_info = {
                    "type": "relative",
                    "hours_back": 48,
                    "description": "Yesterday",
                }
            elif "this week" in query.lower():
                time_info = {
                    "type": "relative",
                    "hours_back": 168,
                    "description": "This week",
                }

        return time_info if time_info else None

    def _extract_threat_categories(self, query: str) -> List[str]:
        """Extract threat categories from query"""
        categories = []

        if any(term in query for term in ["brute", "force", "login", "password"]):
            categories.append("brute_force")
        if any(term in query for term in ["malware", "virus", "trojan"]):
            categories.append("malware")
        if any(term in query for term in ["recon", "scan", "probe"]):
            categories.append("reconnaissance")
        if any(term in query for term in ["lateral", "movement", "escalation"]):
            categories.append("lateral_movement")
        if any(term in query for term in ["exfil", "theft", "steal"]):
            categories.append("data_exfiltration")

        return categories

    def _extract_priority(self, query: str) -> str:
        """Extract priority/severity level"""
        if any(term in query for term in ["critical", "urgent", "emergency"]):
            return "critical"
        elif any(term in query for term in ["high", "important", "severe"]):
            return "high"
        elif any(term in query for term in ["medium", "moderate"]):
            return "medium"
        else:
            return "medium"  # Default

    def _calculate_confidence(
        self,
        entities: Dict,
        time_constraints: Optional[Dict],
        threat_categories: List[str],
    ) -> float:
        """Calculate confidence score for parsed query"""
        score = 0.5  # Base score

        # Add score for extracted entities
        entity_count = sum(len(v) for v in entities.values())
        score += min(0.3, entity_count * 0.1)

        # Add score for time constraints
        if time_constraints:
            score += 0.1

        # Add score for threat categories
        score += min(0.1, len(threat_categories) * 0.05)

        return min(score, 1.0)


class NaturalLanguageThreatAnalyzer:
    """
    Natural Language Threat Analysis Agent
    Enables natural language queries over incident data with AI-powered understanding
    """

    def __init__(self):
        self.agent_id = "nlp_analyzer_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.query_parser = ThreatQueryParser()

        # LLM integration
        self.llm = None
        self.embeddings = None
        self.vector_store = None

        # Configuration
        self.max_context_length = 4000
        self.response_temperature = 0.3
        self.max_findings = 10

        # Statistics
        self.stats = {
            "queries_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_response_time": 0.0,
            "last_activity": datetime.utcnow(),
        }

        # Initialize LLM if available
        if LANGCHAIN_AVAILABLE:
            self._initialize_llm()
        else:
            self.logger.warning(
                "LangChain not available - using fallback NLP processing"
            )

    def _initialize_llm(self):
        """Initialize LLM components"""
        try:
            # Check for OpenAI API key
            openai_key = settings.openai_api_key or settings.xai_api_key

            if openai_key:
                # Initialize ChatOpenAI with appropriate model
                self.llm = ChatOpenAI(
                    temperature=self.response_temperature,
                    model_name="gpt-4"
                    if "sk-" in openai_key
                    else "grok-beta",  # Use Grok for XAI keys
                    max_tokens=1000,
                    openai_api_key=openai_key,
                )

                # Initialize embeddings for semantic search
                self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

                self.logger.info("NLP Analyzer initialized with LLM support")
            else:
                self.logger.warning(
                    "No LLM API key found - using pattern-based analysis only"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")

    async def analyze_natural_language_query(
        self,
        query: str,
        incidents: List[Incident],
        events: List[Event],
        context: Optional[Dict[str, Any]] = None,
    ) -> NLPResponse:
        """
        Analyze natural language query and return structured findings

        Args:
            query: Natural language query from analyst
            incidents: Available incidents to search/analyze
            events: Available events for analysis
            context: Additional context for the query

        Returns:
            Structured NLP response with findings and recommendations
        """
        start_time = datetime.utcnow()

        try:
            self.logger.info(f"Processing NL query: {query[:100]}...")

            # Parse the natural language query
            query_context = self.query_parser.parse_query(query)

            # Generate structured query
            structured_query = await self._generate_structured_query(
                query, query_context
            )

            # Execute analysis based on query type
            findings = await self._execute_analysis(
                query_context, structured_query, incidents, events
            )

            # Generate AI-powered insights if LLM is available
            if self.llm and LANGCHAIN_AVAILABLE:
                ai_insights = await self._generate_ai_insights(
                    query, query_context, findings, incidents, events
                )
                findings.extend(ai_insights.get("findings", []))

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                query, query_context, findings
            )

            # Generate follow-up questions
            follow_up_questions = await self._generate_follow_up_questions(
                query, query_context, findings
            )

            # Update statistics
            self.stats["queries_processed"] += 1
            self.stats["successful_analyses"] += 1
            self.stats["last_activity"] = datetime.utcnow()

            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats["avg_response_time"] = (
                self.stats["avg_response_time"] * (self.stats["queries_processed"] - 1)
                + execution_time
            ) / self.stats["queries_processed"]

            return NLPResponse(
                query_understanding=f"Interpreted as {query_context.query_type.value} with {query_context.confidence_score:.1%} confidence",
                structured_query=structured_query,
                findings=findings[: self.max_findings],
                recommendations=recommendations,
                confidence_score=query_context.confidence_score,
                reasoning=f"Query classified as {query_context.query_type.value} based on keywords and context",
                follow_up_questions=follow_up_questions,
            )

        except Exception as e:
            self.logger.error(f"NL query analysis failed: {e}")
            self.stats["failed_analyses"] += 1

            return NLPResponse(
                query_understanding="Failed to process query",
                structured_query={},
                findings=[{"type": "error", "message": f"Analysis failed: {str(e)}"}],
                recommendations=["Please try rephrasing your query or contact support"],
                confidence_score=0.0,
                reasoning=f"Error during processing: {str(e)}",
                follow_up_questions=[],
            )

    async def _generate_structured_query(
        self, query: str, context: QueryContext
    ) -> Dict[str, Any]:
        """Generate structured query from natural language"""

        structured = {
            "query_type": context.query_type.value,
            "entities": context.extracted_entities,
            "time_constraints": context.time_constraints,
            "threat_categories": context.threat_categories,
            "priority": context.priority_level,
            "filters": {},
        }

        # Add filters based on extracted entities
        if context.extracted_entities.get("ips"):
            structured["filters"]["src_ips"] = context.extracted_entities["ips"]

        if context.threat_categories:
            structured["filters"]["threat_categories"] = context.threat_categories

        if context.time_constraints:
            structured["filters"]["time_range"] = context.time_constraints

        return structured

    async def _execute_analysis(
        self,
        context: QueryContext,
        structured_query: Dict[str, Any],
        incidents: List[Incident],
        events: List[Event],
    ) -> List[Dict[str, Any]]:
        """Execute analysis based on query type"""

        findings = []

        if context.query_type == QueryType.INCIDENT_SEARCH:
            findings = await self._search_incidents(structured_query, incidents)

        elif context.query_type == QueryType.THREAT_ANALYSIS:
            findings = await self._analyze_threats(structured_query, incidents, events)

        elif context.query_type == QueryType.PATTERN_RECOGNITION:
            findings = await self._recognize_patterns(
                structured_query, incidents, events
            )

        elif context.query_type == QueryType.TIMELINE_ANALYSIS:
            findings = await self._analyze_timeline(structured_query, incidents, events)

        elif context.query_type == QueryType.IOC_EXTRACTION:
            findings = await self._extract_iocs(structured_query, incidents, events)

        elif context.query_type == QueryType.ATTRIBUTION_QUERY:
            findings = await self._analyze_attribution(
                structured_query, incidents, events
            )

        elif context.query_type == QueryType.RECOMMENDATION:
            findings = await self._generate_action_recommendations(
                structured_query, incidents, events
            )

        return findings

    async def _search_incidents(
        self, query: Dict[str, Any], incidents: List[Incident]
    ) -> List[Dict[str, Any]]:
        """Search incidents based on structured query"""
        findings = []

        # Apply filters
        filtered_incidents = incidents

        if query.get("filters", {}).get("src_ips"):
            target_ips = query["filters"]["src_ips"]
            filtered_incidents = [
                inc for inc in filtered_incidents if inc.src_ip in target_ips
            ]

        if query.get("filters", {}).get("time_range"):
            time_range = query["filters"]["time_range"]
            if time_range.get("hours_back"):
                cutoff_time = datetime.utcnow() - timedelta(
                    hours=time_range["hours_back"]
                )
                filtered_incidents = [
                    inc for inc in filtered_incidents if inc.created_at >= cutoff_time
                ]

        # Generate findings
        for incident in filtered_incidents[:10]:  # Limit results
            findings.append(
                {
                    "type": "incident",
                    "incident_id": incident.id,
                    "src_ip": incident.src_ip,
                    "status": incident.status,
                    "reason": incident.reason,
                    "created_at": incident.created_at.isoformat(),
                    "risk_score": getattr(incident, "risk_score", 0.0),
                    "severity": getattr(incident, "escalation_level", "medium"),
                    "relevance_score": 0.9,  # High relevance for direct matches
                }
            )

        return findings

    async def _analyze_threats(
        self, query: Dict[str, Any], incidents: List[Incident], events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Analyze threats based on query"""
        findings = []

        # Group incidents by threat characteristics
        threat_groups = {}

        for incident in incidents:
            threat_type = incident.reason or "unknown"
            if threat_type not in threat_groups:
                threat_groups[threat_type] = []
            threat_groups[threat_type].append(incident)

        # Analyze each threat group
        for threat_type, threat_incidents in threat_groups.items():
            if (
                len(threat_incidents) >= 2
            ):  # Only include patterns with multiple incidents
                findings.append(
                    {
                        "type": "threat_analysis",
                        "threat_type": threat_type,
                        "incident_count": len(threat_incidents),
                        "unique_ips": len(set(inc.src_ip for inc in threat_incidents)),
                        "time_span_hours": self._calculate_time_span(threat_incidents),
                        "severity_distribution": self._calculate_severity_distribution(
                            threat_incidents
                        ),
                        "relevance_score": 0.8,
                    }
                )

        return findings

    async def _recognize_patterns(
        self, query: Dict[str, Any], incidents: List[Incident], events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Recognize patterns in incidents and events"""
        findings = []

        # Pattern 1: IP clustering
        ip_counts = {}
        for incident in incidents:
            ip_counts[incident.src_ip] = ip_counts.get(incident.src_ip, 0) + 1

        # Find IPs with multiple incidents
        repeat_attackers = {ip: count for ip, count in ip_counts.items() if count > 1}

        if repeat_attackers:
            findings.append(
                {
                    "type": "pattern_repeat_attackers",
                    "description": "IPs with multiple incidents detected",
                    "repeat_ips": list(repeat_attackers.keys()),
                    "max_incidents_per_ip": max(repeat_attackers.values()),
                    "total_repeat_incidents": sum(repeat_attackers.values()),
                    "relevance_score": 0.9,
                }
            )

        # Pattern 2: Time-based clustering
        time_clusters = self._find_time_clusters(incidents)
        if time_clusters:
            findings.append(
                {
                    "type": "pattern_time_clustering",
                    "description": "Temporal clustering of incidents detected",
                    "clusters": time_clusters,
                    "relevance_score": 0.7,
                }
            )

        return findings

    async def _analyze_timeline(
        self, query: Dict[str, Any], incidents: List[Incident], events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Analyze timeline of incidents and events"""
        findings = []

        # Sort incidents by time
        sorted_incidents = sorted(incidents, key=lambda x: x.created_at)

        # Create timeline analysis
        timeline_events = []
        for incident in sorted_incidents[-20:]:  # Last 20 incidents
            timeline_events.append(
                {
                    "timestamp": incident.created_at.isoformat(),
                    "type": "incident",
                    "src_ip": incident.src_ip,
                    "reason": incident.reason,
                    "status": incident.status,
                }
            )

        findings.append(
            {
                "type": "timeline_analysis",
                "description": "Chronological sequence of recent incidents",
                "timeline": timeline_events,
                "total_incidents": len(sorted_incidents),
                "time_span": self._calculate_time_span(sorted_incidents),
                "relevance_score": 0.8,
            }
        )

        return findings

    async def _extract_iocs(
        self, query: Dict[str, Any], incidents: List[Incident], events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Extract Indicators of Compromise (IOCs)"""
        findings = []

        # Extract IPs
        unique_ips = list(set(incident.src_ip for incident in incidents))

        # Extract other IOCs from events (if available)
        domains = set()
        urls = set()
        hashes = set()

        for event in events:
            if hasattr(event, "raw") and event.raw:
                raw_data = (
                    json.dumps(event.raw)
                    if isinstance(event.raw, dict)
                    else str(event.raw)
                )

                # Extract domains (simple pattern)
                import re

                domain_pattern = r"\b(?:[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
                domains.update(re.findall(domain_pattern, raw_data))

                # Extract URLs
                url_pattern = r"https?://[^\s]+"
                urls.update(re.findall(url_pattern, raw_data))

        findings.append(
            {
                "type": "ioc_extraction",
                "description": "Extracted Indicators of Compromise",
                "iocs": {
                    "ip_addresses": unique_ips[:50],  # Limit results
                    "domains": list(domains)[:20],
                    "urls": list(urls)[:10],
                    "hashes": list(hashes)[:20],
                },
                "total_unique_ips": len(unique_ips),
                "total_domains": len(domains),
                "relevance_score": 0.9,
            }
        )

        return findings

    async def _analyze_attribution(
        self, query: Dict[str, Any], incidents: List[Incident], events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Analyze threat attribution"""
        findings = []

        # Simple geographical clustering based on IP patterns
        geo_clusters = {}

        for incident in incidents:
            # Simple IP-based geographical inference (in reality, would use GeoIP)
            ip_prefix = ".".join(incident.src_ip.split(".")[:2])
            if ip_prefix not in geo_clusters:
                geo_clusters[ip_prefix] = []
            geo_clusters[ip_prefix].append(incident)

        # Generate attribution findings
        for ip_prefix, prefix_incidents in geo_clusters.items():
            if len(prefix_incidents) >= 3:  # Minimum for attribution
                findings.append(
                    {
                        "type": "attribution_cluster",
                        "description": f"Activity cluster from IP range {ip_prefix}.*.*",
                        "ip_prefix": ip_prefix,
                        "incident_count": len(prefix_incidents),
                        "unique_ips": len(set(inc.src_ip for inc in prefix_incidents)),
                        "attack_types": list(
                            set(inc.reason for inc in prefix_incidents if inc.reason)
                        ),
                        "confidence_score": min(len(prefix_incidents) * 0.1, 0.8),
                        "relevance_score": 0.7,
                    }
                )

        return findings

    async def _generate_action_recommendations(
        self, query: Dict[str, Any], incidents: List[Incident], events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        findings = []

        # Analyze incident statuses
        status_counts = {}
        for incident in incidents:
            status_counts[incident.status] = status_counts.get(incident.status, 0) + 1

        recommendations = []

        # Recommend containment for open incidents
        open_incidents = status_counts.get("open", 0)
        if open_incidents > 0:
            recommendations.append(
                f"Consider containing {open_incidents} open incidents"
            )

        # Recommend investigation for new incidents
        new_incidents = status_counts.get("new", 0)
        if new_incidents > 0:
            recommendations.append(f"Investigate {new_incidents} new incidents")

        # Recommend pattern analysis if many incidents
        total_incidents = len(incidents)
        if total_incidents > 10:
            recommendations.append("Perform pattern analysis on recent incident surge")

        findings.append(
            {
                "type": "recommendations",
                "description": "Actionable recommendations based on current incident landscape",
                "recommendations": recommendations,
                "incident_summary": status_counts,
                "total_incidents": total_incidents,
                "relevance_score": 0.9,
            }
        )

        return findings

    async def _generate_ai_insights(
        self,
        query: str,
        context: QueryContext,
        findings: List[Dict[str, Any]],
        incidents: List[Incident],
        events: List[Event],
    ) -> Dict[str, Any]:
        """Generate AI-powered insights using LLM"""

        if not self.llm:
            return {"findings": []}

        try:
            # Prepare context for LLM
            incident_summary = self._prepare_incident_summary(
                incidents[:10]
            )  # Limit context
            findings_summary = self._prepare_findings_summary(findings)

            # Create prompt for AI analysis
            system_prompt = """You are a cybersecurity analyst AI assistant. Analyze the provided incident data and findings to generate additional insights.

Focus on:
1. Threat patterns and correlations
2. Risk assessments
3. Strategic recommendations
4. Potential missed indicators

Provide structured, actionable insights."""

            user_prompt = f"""
Original Query: {query}

Incident Summary:
{incident_summary}

Current Findings:
{findings_summary}

Please provide additional insights and analysis that might not be immediately obvious from the data.
"""

            # Get AI response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            ai_response = await self.llm.agenerate([messages])
            ai_content = ai_response.generations[0][0].text

            # Parse AI response into structured findings
            ai_findings = [
                {
                    "type": "ai_insights",
                    "description": "AI-powered analysis and insights",
                    "content": ai_content,
                    "confidence_score": 0.8,
                    "relevance_score": 0.9,
                    "source": "llm_analysis",
                }
            ]

            return {"findings": ai_findings}

        except Exception as e:
            self.logger.error(f"AI insights generation failed: {e}")
            return {"findings": []}

    async def _generate_recommendations(
        self, query: str, context: QueryContext, findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on findings"""

        recommendations = []

        # Generic recommendations based on query type
        if context.query_type == QueryType.INCIDENT_SEARCH:
            recommendations.append(
                "Review identified incidents for containment opportunities"
            )
            recommendations.append("Check for related incidents with similar patterns")

        elif context.query_type == QueryType.THREAT_ANALYSIS:
            recommendations.append("Monitor identified threat patterns for escalation")
            recommendations.append(
                "Update detection rules based on threat characteristics"
            )

        elif context.query_type == QueryType.PATTERN_RECOGNITION:
            recommendations.append("Investigate root causes of identified patterns")
            recommendations.append("Consider proactive blocking of repeat offenders")

        # Add specific recommendations based on findings
        for finding in findings:
            if finding.get("type") == "pattern_repeat_attackers":
                recommendations.append("Consider blocking repeat attacker IPs")
            elif finding.get("type") == "attribution_cluster":
                recommendations.append("Investigate potential coordinated campaign")

        # Add time-sensitive recommendations
        if (
            context.time_constraints
            and context.time_constraints.get("hours_back", 0) <= 1
        ):
            recommendations.append(
                "Consider immediate containment for recent incidents"
            )

        return list(set(recommendations))  # Remove duplicates

    async def _generate_follow_up_questions(
        self, query: str, context: QueryContext, findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate relevant follow-up questions"""

        follow_ups = []

        # Based on query type
        if context.query_type == QueryType.INCIDENT_SEARCH:
            follow_ups.extend(
                [
                    "Would you like to see more details about any specific incident?",
                    "Should we analyze patterns in these incidents?",
                    "Are there any specific IPs you'd like to investigate further?",
                ]
            )

        elif context.query_type == QueryType.THREAT_ANALYSIS:
            follow_ups.extend(
                [
                    "Would you like attribution analysis for these threats?",
                    "Should we look for similar threats in a different time period?",
                    "Do you want recommendations for containment actions?",
                ]
            )

        # Based on findings
        if any(f.get("type") == "pattern_repeat_attackers" for f in findings):
            follow_ups.append(
                "Would you like detailed analysis of the repeat attackers?"
            )

        if any(f.get("type") == "attribution_cluster" for f in findings):
            follow_ups.append("Should we investigate the potential threat campaign?")

        # Generic follow-ups
        follow_ups.extend(
            [
                "Would you like to refine your search criteria?",
                "Should we expand the time range for analysis?",
                "Do you need specific IOCs extracted from these findings?",
            ]
        )

        return follow_ups[:5]  # Limit to top 5

    # Helper methods
    def _calculate_time_span(self, incidents: List[Incident]) -> float:
        """Calculate time span in hours between first and last incident"""
        if len(incidents) < 2:
            return 0.0

        times = [inc.created_at for inc in incidents]
        time_span = (max(times) - min(times)).total_seconds() / 3600
        return round(time_span, 2)

    def _calculate_severity_distribution(
        self, incidents: List[Incident]
    ) -> Dict[str, int]:
        """Calculate distribution of severity levels"""
        distribution = {}
        for incident in incidents:
            severity = getattr(incident, "escalation_level", "medium")
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution

    def _find_time_clusters(self, incidents: List[Incident]) -> List[Dict[str, Any]]:
        """Find temporal clusters in incidents"""
        if len(incidents) < 3:
            return []

        # Sort by time
        sorted_incidents = sorted(incidents, key=lambda x: x.created_at)

        clusters = []
        current_cluster = [sorted_incidents[0]]

        for i in range(1, len(sorted_incidents)):
            time_diff = (
                sorted_incidents[i].created_at - current_cluster[-1].created_at
            ).total_seconds()

            # If within 1 hour, add to current cluster
            if time_diff <= 3600:
                current_cluster.append(sorted_incidents[i])
            else:
                # Save current cluster if it has multiple incidents
                if len(current_cluster) >= 2:
                    clusters.append(
                        {
                            "start_time": current_cluster[0].created_at.isoformat(),
                            "end_time": current_cluster[-1].created_at.isoformat(),
                            "incident_count": len(current_cluster),
                            "duration_minutes": (
                                current_cluster[-1].created_at
                                - current_cluster[0].created_at
                            ).total_seconds()
                            / 60,
                        }
                    )

                # Start new cluster
                current_cluster = [sorted_incidents[i]]

        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            clusters.append(
                {
                    "start_time": current_cluster[0].created_at.isoformat(),
                    "end_time": current_cluster[-1].created_at.isoformat(),
                    "incident_count": len(current_cluster),
                    "duration_minutes": (
                        current_cluster[-1].created_at - current_cluster[0].created_at
                    ).total_seconds()
                    / 60,
                }
            )

        return clusters

    def _prepare_incident_summary(self, incidents: List[Incident]) -> str:
        """Prepare incident summary for LLM context"""
        if not incidents:
            return "No incidents available"

        summary_lines = []
        for incident in incidents:
            summary_lines.append(
                f"ID:{incident.id} IP:{incident.src_ip} Status:{incident.status} "
                f"Reason:{incident.reason} Time:{incident.created_at.strftime('%Y-%m-%d %H:%M')}"
            )

        return "\n".join(summary_lines)

    def _prepare_findings_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Prepare findings summary for LLM context"""
        if not findings:
            return "No findings available"

        summary_lines = []
        for finding in findings:
            summary_lines.append(
                f"Type: {finding.get('type', 'unknown')} - {finding.get('description', 'No description')}"
            )

        return "\n".join(summary_lines)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "langchain_available": LANGCHAIN_AVAILABLE,
            "llm_initialized": self.llm is not None,
            "embeddings_available": self.embeddings is not None,
            "statistics": self.stats.copy(),
            "configuration": {
                "max_context_length": self.max_context_length,
                "response_temperature": self.response_temperature,
                "max_findings": self.max_findings,
            },
        }

    async def semantic_search_incidents(
        self, query: str, incidents: List[Incident], top_k: int = 5
    ) -> List[Tuple[Incident, float]]:
        """
        Perform semantic search over incidents using embeddings

        Returns list of (incident, similarity_score) tuples
        """
        if not self.embeddings or not incidents:
            return []

        try:
            # Prepare incident documents
            documents = []
            for incident in incidents:
                doc_text = f"IP: {incident.src_ip} Reason: {incident.reason} Status: {incident.status}"
                documents.append(
                    Document(
                        page_content=doc_text,
                        metadata={"incident_id": incident.id, "incident": incident},
                    )
                )

            # Create temporary vector store
            vector_store = FAISS.from_documents(documents, self.embeddings)

            # Search for similar documents
            results = vector_store.similarity_search_with_score(query, k=top_k)

            # Extract incidents with scores
            incident_results = []
            for doc, score in results:
                incident = doc.metadata["incident"]
                # Convert distance to similarity (FAISS returns distance, lower is better)
                similarity = max(0, 1 - score)
                incident_results.append((incident, similarity))

            return incident_results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []


# Global NLP analyzer instance
nlp_analyzer = NaturalLanguageThreatAnalyzer()


async def get_nlp_analyzer() -> NaturalLanguageThreatAnalyzer:
    """Get the global NLP analyzer instance"""
    return nlp_analyzer
