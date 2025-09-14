"""
Predictive Threat Hunting Agent for Mini-XDR
Proactively identifies emerging threats using time-series analysis and behavioral predictions
"""
import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import math

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..models import Incident, Event, Action
from ..config import settings

logger = logging.getLogger(__name__)


class ThreatPredictionModel(Enum):
    """Available prediction models"""
    TIME_SERIES_ARIMA = "time_series_arima"
    LSTM_PREDICTOR = "lstm_predictor"
    ISOLATION_FOREST = "isolation_forest"
    BEHAVIORAL_CLUSTERING = "behavioral_clustering"
    ENSEMBLE_PREDICTOR = "ensemble_predictor"


class ThreatCategory(Enum):
    """Categories of threats for prediction"""
    BRUTE_FORCE = "brute_force"
    MALWARE = "malware"
    RECONNAISSANCE = "reconnaissance"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    DENIAL_OF_SERVICE = "denial_of_service"
    COMMAND_AND_CONTROL = "command_and_control"
    UNKNOWN = "unknown"


class PredictionHorizon(Enum):
    """Time horizons for predictions"""
    SHORT_TERM = timedelta(hours=1)     # Next 1 hour
    MEDIUM_TERM = timedelta(hours=24)   # Next 24 hours
    LONG_TERM = timedelta(days=7)       # Next 7 days


@dataclass
class ThreatPrediction:
    """A threat prediction with confidence and details"""
    prediction_id: str
    threat_category: ThreatCategory
    predicted_probability: float
    confidence_score: float
    prediction_horizon: PredictionHorizon
    contributing_factors: List[str]
    predicted_targets: List[str]
    early_warning_indicators: List[Dict[str, Any]]
    mitigation_recommendations: List[str]
    prediction_timestamp: datetime
    model_used: ThreatPredictionModel
    supporting_evidence: Dict[str, Any]


@dataclass
class HuntingHypothesis:
    """A hypothesis for threat hunting"""
    hypothesis_id: str
    hypothesis_text: str
    threat_categories: List[ThreatCategory]
    confidence: float
    supporting_indicators: List[str]
    hunting_queries: List[str]
    expected_evidence: List[str]
    priority_level: str
    created_timestamp: datetime
    validation_criteria: Dict[str, Any]


@dataclass
class BehavioralBaseline:
    """Baseline behavioral patterns for anomaly detection"""
    baseline_id: str
    entity_type: str  # 'ip', 'user', 'system'
    entity_id: str
    time_window: timedelta
    baseline_metrics: Dict[str, float]
    statistical_bounds: Dict[str, Tuple[float, float]]
    last_updated: datetime
    confidence_level: float


class TimeSeriesPredictor:
    """Time-series analysis for threat prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.predictions_cache = {}
        self.model_accuracy = defaultdict(float)
    
    def prepare_time_series_data(
        self, 
        events: List[Event], 
        time_window: timedelta = timedelta(hours=1)
    ) -> pd.DataFrame:
        """Prepare event data for time-series analysis"""
        
        if not events:
            return pd.DataFrame()
        
        # Create time-series DataFrame
        event_data = []
        for event in events:
            event_data.append({
                'timestamp': event.ts,
                'src_ip': event.src_ip,
                'eventid': event.eventid,
                'severity': getattr(event, 'severity', 'medium'),
                'hour': event.ts.hour,
                'day_of_week': event.ts.weekday(),
                'is_weekend': event.ts.weekday() >= 5
            })
        
        df = pd.DataFrame(event_data)
        
        if df.empty:
            return df
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create time-based aggregations
        time_series = df.resample(f'{int(time_window.total_seconds()//60)}T').agg({
            'src_ip': 'nunique',
            'eventid': 'count', 
            'hour': 'mean',
            'day_of_week': 'mean',
            'is_weekend': 'max'
        }).fillna(0)
        
        # Add rolling statistics
        time_series['event_count_ma_3'] = time_series['eventid'].rolling(window=3, center=True).mean().fillna(0)
        time_series['event_count_ma_6'] = time_series['eventid'].rolling(window=6, center=True).mean().fillna(0)
        time_series['unique_ips_ma_3'] = time_series['src_ip'].rolling(window=3, center=True).mean().fillna(0)
        
        return time_series
    
    async def predict_threat_probability(
        self,
        historical_data: pd.DataFrame,
        prediction_horizon: PredictionHorizon,
        threat_category: ThreatCategory
    ) -> float:
        """Predict probability of threat occurrence"""
        
        if historical_data.empty or len(historical_data) < 10:
            return 0.0
        
        try:
            # Simple trend-based prediction
            recent_events = historical_data['eventid'].tail(6)
            if len(recent_events) < 2:
                return 0.0
            
            # Calculate trend
            x = np.arange(len(recent_events))
            y = recent_events.values
            
            # Linear regression for trend
            if len(x) >= 2:
                slope = np.polyfit(x, y, 1)[0]
                
                # Normalize slope to probability
                base_probability = min(max(slope / 10.0, 0.0), 1.0)
                
                # Adjust based on threat category patterns
                category_multipliers = {
                    ThreatCategory.BRUTE_FORCE: 1.2,
                    ThreatCategory.RECONNAISSANCE: 0.8,
                    ThreatCategory.MALWARE: 1.0,
                    ThreatCategory.LATERAL_MOVEMENT: 1.1,
                    ThreatCategory.DATA_EXFILTRATION: 0.9
                }
                
                multiplier = category_multipliers.get(threat_category, 1.0)
                probability = min(base_probability * multiplier, 1.0)
                
                # Factor in recent activity levels
                recent_activity = recent_events.mean()
                historical_activity = historical_data['eventid'].mean()
                
                if recent_activity > historical_activity * 1.5:
                    probability *= 1.3
                elif recent_activity < historical_activity * 0.5:
                    probability *= 0.7
                
                return min(probability, 1.0)
        
        except Exception as e:
            logger.error(f"Threat probability prediction failed: {e}")
        
        return 0.0


class BehavioralAnalyzer:
    """Analyzes behavioral patterns to detect anomalies"""
    
    def __init__(self):
        self.baselines: Dict[str, BehavioralBaseline] = {}
        self.anomaly_detector = IsolationForest(contamination=0.1) if SKLEARN_AVAILABLE else None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    async def create_behavioral_baseline(
        self,
        entity_type: str,
        entity_id: str,
        historical_events: List[Event],
        time_window: timedelta = timedelta(days=7)
    ) -> BehavioralBaseline:
        """Create behavioral baseline for an entity"""
        
        baseline_id = f"{entity_type}_{entity_id}_{int(datetime.utcnow().timestamp())}"
        
        # Extract behavioral metrics
        metrics = self._extract_behavioral_metrics(historical_events)
        
        # Calculate statistical bounds (mean ± 2 standard deviations)
        statistical_bounds = {}
        for metric, values in metrics.items():
            if values and len(values) > 1:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                statistical_bounds[metric] = (
                    max(0, mean_val - 2 * std_val),
                    mean_val + 2 * std_val
                )
            else:
                statistical_bounds[metric] = (0.0, 0.0)
        
        # Calculate baseline summary metrics
        baseline_metrics = {}
        for metric, values in metrics.items():
            if values:
                baseline_metrics[f"{metric}_mean"] = statistics.mean(values)
                baseline_metrics[f"{metric}_median"] = statistics.median(values)
                baseline_metrics[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0
            else:
                baseline_metrics[f"{metric}_mean"] = 0
                baseline_metrics[f"{metric}_median"] = 0
                baseline_metrics[f"{metric}_std"] = 0
        
        # Calculate confidence level based on data quality
        data_points = len(historical_events)
        time_coverage = (max(e.ts for e in historical_events) - min(e.ts for e in historical_events)).days if historical_events else 0
        confidence_level = min(0.95, (data_points / 100) * (time_coverage / 7))
        
        baseline = BehavioralBaseline(
            baseline_id=baseline_id,
            entity_type=entity_type,
            entity_id=entity_id,
            time_window=time_window,
            baseline_metrics=baseline_metrics,
            statistical_bounds=statistical_bounds,
            last_updated=datetime.utcnow(),
            confidence_level=confidence_level
        )
        
        # Store baseline
        key = f"{entity_type}_{entity_id}"
        self.baselines[key] = baseline
        
        logger.info(f"Created behavioral baseline for {entity_type} {entity_id} with {confidence_level:.2f} confidence")
        
        return baseline
    
    def _extract_behavioral_metrics(self, events: List[Event]) -> Dict[str, List[float]]:
        """Extract behavioral metrics from events"""
        
        metrics = defaultdict(list)
        
        if not events:
            return metrics
        
        # Group events by time periods
        events_by_hour = defaultdict(list)
        events_by_day = defaultdict(list)
        
        for event in events:
            hour_key = event.ts.replace(minute=0, second=0, microsecond=0)
            day_key = event.ts.replace(hour=0, minute=0, second=0, microsecond=0)
            
            events_by_hour[hour_key].append(event)
            events_by_day[day_key].append(event)
        
        # Calculate hourly metrics
        for hour, hour_events in events_by_hour.items():
            metrics['events_per_hour'].append(len(hour_events))
            
            unique_event_types = len(set(e.eventid for e in hour_events))
            metrics['unique_event_types_per_hour'].append(unique_event_types)
            
            # Time-based patterns
            metrics['hour_of_day'].append(hour.hour)
            metrics['is_business_hours'].append(1.0 if 9 <= hour.hour <= 17 else 0.0)
        
        # Calculate daily metrics
        for day, day_events in events_by_day.items():
            metrics['events_per_day'].append(len(day_events))
            
            unique_sources = len(set(e.src_ip for e in day_events))
            metrics['unique_sources_per_day'].append(unique_sources)
            
            # Day patterns
            metrics['day_of_week'].append(day.weekday())
            metrics['is_weekend'].append(1.0 if day.weekday() >= 5 else 0.0)
        
        # Calculate sequence-based metrics
        if len(events) > 1:
            time_intervals = []
            for i in range(1, len(events)):
                interval = (events[i].ts - events[i-1].ts).total_seconds() / 60  # minutes
                time_intervals.append(interval)
            
            if time_intervals:
                metrics['avg_time_between_events'] = [statistics.mean(time_intervals)]
                metrics['median_time_between_events'] = [statistics.median(time_intervals)]
        
        return dict(metrics)
    
    async def detect_anomalies(
        self,
        entity_type: str,
        entity_id: str,
        recent_events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies against baseline"""
        
        key = f"{entity_type}_{entity_id}"
        baseline = self.baselines.get(key)
        
        if not baseline:
            logger.warning(f"No baseline found for {entity_type} {entity_id}")
            return []
        
        # Extract current metrics
        current_metrics = self._extract_behavioral_metrics(recent_events)
        
        anomalies = []
        
        # Check each metric against baseline bounds
        for metric_group, values in current_metrics.items():
            if not values:
                continue
                
            current_value = statistics.mean(values)
            baseline_key = f"{metric_group}_mean"
            
            if baseline_key in baseline.baseline_metrics:
                baseline_mean = baseline.baseline_metrics[baseline_key]
                baseline_bounds = baseline.statistical_bounds.get(metric_group, (0, float('inf')))
                
                # Check if current value is outside bounds
                lower_bound, upper_bound = baseline_bounds
                
                if current_value < lower_bound or current_value > upper_bound:
                    deviation = abs(current_value - baseline_mean) / max(baseline_mean, 0.1)
                    
                    anomalies.append({
                        'metric': metric_group,
                        'current_value': current_value,
                        'baseline_mean': baseline_mean,
                        'baseline_bounds': baseline_bounds,
                        'deviation_magnitude': deviation,
                        'anomaly_type': 'above_baseline' if current_value > upper_bound else 'below_baseline',
                        'severity': 'high' if deviation > 2.0 else 'medium' if deviation > 1.0 else 'low'
                    })
        
        return anomalies


class HypothesisGenerator:
    """Generates threat hunting hypotheses"""
    
    def __init__(self):
        self.hypothesis_templates = self._load_hypothesis_templates()
        self.generated_hypotheses: Dict[str, HuntingHypothesis] = {}
    
    def _load_hypothesis_templates(self) -> List[Dict[str, Any]]:
        """Load predefined hypothesis templates"""
        
        return [
            {
                'template': "Multiple failed login attempts from {ip} may indicate brute force attack",
                'threat_categories': [ThreatCategory.BRUTE_FORCE],
                'trigger_conditions': ['failed_login_count > 10'],
                'hunting_queries': [
                    "eventid:cowrie.login.failed src_ip:{ip}",
                    "eventid:cowrie.session.connect src_ip:{ip}"
                ],
                'expected_evidence': ['failed_login_events', 'session_attempts', 'credential_patterns']
            },
            {
                'template': "Unusual port scanning activity from {ip} suggests reconnaissance",
                'threat_categories': [ThreatCategory.RECONNAISSANCE],
                'trigger_conditions': ['unique_ports > 5', 'scan_rate > 1_per_minute'],
                'hunting_queries': [
                    "dst_port:* src_ip:{ip}",
                    "eventid:cowrie.client.version src_ip:{ip}"
                ],
                'expected_evidence': ['port_scan_patterns', 'service_enumeration', 'banner_grabbing']
            },
            {
                'template': "File download activity from {ip} could indicate malware deployment",
                'threat_categories': [ThreatCategory.MALWARE],
                'trigger_conditions': ['download_count > 0'],
                'hunting_queries': [
                    "eventid:cowrie.session.file_download src_ip:{ip}",
                    "eventid:cowrie.command.input input:*wget* src_ip:{ip}",
                    "eventid:cowrie.command.input input:*curl* src_ip:{ip}"
                ],
                'expected_evidence': ['downloaded_files', 'command_execution', 'file_hashes']
            },
            {
                'template': "Command execution patterns from {ip} suggest lateral movement attempts",
                'threat_categories': [ThreatCategory.LATERAL_MOVEMENT],
                'trigger_conditions': ['command_diversity > 5', 'network_commands > 0'],
                'hunting_queries': [
                    "eventid:cowrie.command.input input:*ssh* src_ip:{ip}",
                    "eventid:cowrie.command.input input:*nc* src_ip:{ip}",
                    "eventid:cowrie.command.input input:*nmap* src_ip:{ip}"
                ],
                'expected_evidence': ['lateral_movement_commands', 'network_discovery', 'credential_usage']
            },
            {
                'template': "Data exfiltration indicators detected from {ip}",
                'threat_categories': [ThreatCategory.DATA_EXFILTRATION],
                'trigger_conditions': ['upload_count > 0', 'large_data_transfers'],
                'hunting_queries': [
                    "eventid:cowrie.session.file_upload src_ip:{ip}",
                    "eventid:cowrie.command.input input:*scp* src_ip:{ip}",
                    "eventid:cowrie.command.input input:*rsync* src_ip:{ip}"
                ],
                'expected_evidence': ['uploaded_files', 'data_transfer_volumes', 'compression_activities']
            }
        ]
    
    async def generate_hypotheses(
        self,
        recent_incidents: List[Incident],
        recent_events: List[Event],
        anomalies: List[Dict[str, Any]]
    ) -> List[HuntingHypothesis]:
        """Generate hunting hypotheses based on recent activity"""
        
        hypotheses = []
        
        # Group events by source IP for analysis
        events_by_ip = defaultdict(list)
        for event in recent_events:
            events_by_ip[event.src_ip].append(event)
        
        # Generate hypotheses for each IP with significant activity
        for src_ip, ip_events in events_by_ip.items():
            if len(ip_events) < 5:  # Skip low-activity IPs
                continue
            
            ip_hypotheses = await self._generate_ip_hypotheses(src_ip, ip_events)
            hypotheses.extend(ip_hypotheses)
        
        # Generate hypotheses based on anomalies
        if anomalies:
            anomaly_hypotheses = await self._generate_anomaly_hypotheses(anomalies)
            hypotheses.extend(anomaly_hypotheses)
        
        # Generate pattern-based hypotheses
        pattern_hypotheses = await self._generate_pattern_hypotheses(recent_incidents, recent_events)
        hypotheses.extend(pattern_hypotheses)
        
        # Store generated hypotheses
        for hypothesis in hypotheses:
            self.generated_hypotheses[hypothesis.hypothesis_id] = hypothesis
        
        # Sort by confidence and priority
        hypotheses.sort(key=lambda h: (h.confidence, h.priority_level == 'high'), reverse=True)
        
        return hypotheses
    
    async def _generate_ip_hypotheses(self, src_ip: str, events: List[Event]) -> List[HuntingHypothesis]:
        """Generate hypotheses for a specific IP"""
        
        hypotheses = []
        
        # Analyze event patterns
        event_types = defaultdict(int)
        failed_logins = 0
        unique_ports = set()
        downloads = 0
        uploads = 0
        commands = []
        
        for event in events:
            event_types[event.eventid] += 1
            
            if event.eventid == "cowrie.login.failed":
                failed_logins += 1
            elif hasattr(event, 'dst_port') and event.dst_port:
                unique_ports.add(event.dst_port)
            elif event.eventid == "cowrie.session.file_download":
                downloads += 1
            elif event.eventid == "cowrie.session.file_upload":
                uploads += 1
            elif event.eventid == "cowrie.command.input" and hasattr(event, 'raw'):
                if isinstance(event.raw, dict) and 'input' in event.raw:
                    commands.append(event.raw['input'])
        
        # Check each template against the patterns
        for template in self.hypothesis_templates:
            conditions_met = []
            
            # Check trigger conditions
            for condition in template['trigger_conditions']:
                if 'failed_login_count' in condition and failed_logins > 10:
                    conditions_met.append(f"Failed logins: {failed_logins}")
                elif 'unique_ports' in condition and len(unique_ports) > 5:
                    conditions_met.append(f"Unique ports accessed: {len(unique_ports)}")
                elif 'download_count' in condition and downloads > 0:
                    conditions_met.append(f"Downloads: {downloads}")
                elif 'upload_count' in condition and uploads > 0:
                    conditions_met.append(f"Uploads: {uploads}")
                elif 'command_diversity' in condition and len(set(commands)) > 5:
                    conditions_met.append(f"Unique commands: {len(set(commands))}")
            
            # If conditions are met, generate hypothesis
            if conditions_met:
                hypothesis_text = template['template'].format(ip=src_ip)
                confidence = min(0.9, 0.3 + (len(conditions_met) * 0.2))
                
                hunting_queries = [
                    query.format(ip=src_ip) for query in template['hunting_queries']
                ]
                
                hypothesis = HuntingHypothesis(
                    hypothesis_id=f"ip_{src_ip}_{template['threat_categories'][0].value}_{int(datetime.utcnow().timestamp())}",
                    hypothesis_text=hypothesis_text,
                    threat_categories=template['threat_categories'],
                    confidence=confidence,
                    supporting_indicators=conditions_met,
                    hunting_queries=hunting_queries,
                    expected_evidence=template['expected_evidence'],
                    priority_level='high' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'low',
                    created_timestamp=datetime.utcnow(),
                    validation_criteria={
                        'minimum_events': 5,
                        'time_window_hours': 24,
                        'confidence_threshold': 0.6
                    }
                )
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_anomaly_hypotheses(self, anomalies: List[Dict[str, Any]]) -> List[HuntingHypothesis]:
        """Generate hypotheses based on detected anomalies"""
        
        hypotheses = []
        
        for anomaly in anomalies:
            if anomaly['severity'] in ['high', 'medium']:
                hypothesis_text = f"Behavioral anomaly detected: {anomaly['metric']} is {anomaly['anomaly_type']} (current: {anomaly['current_value']:.2f}, baseline: {anomaly['baseline_mean']:.2f})"
                
                confidence = 0.8 if anomaly['severity'] == 'high' else 0.6
                
                hypothesis = HuntingHypothesis(
                    hypothesis_id=f"anomaly_{anomaly['metric']}_{int(datetime.utcnow().timestamp())}",
                    hypothesis_text=hypothesis_text,
                    threat_categories=[ThreatCategory.UNKNOWN],
                    confidence=confidence,
                    supporting_indicators=[f"Deviation: {anomaly['deviation_magnitude']:.2f}x"],
                    hunting_queries=[
                        f"metric:{anomaly['metric']}",
                        "eventid:* | stats by src_ip"
                    ],
                    expected_evidence=['statistical_deviation', 'behavioral_change'],
                    priority_level='high' if anomaly['severity'] == 'high' else 'medium',
                    created_timestamp=datetime.utcnow(),
                    validation_criteria={
                        'minimum_deviation': 1.5,
                        'baseline_confidence': 0.7
                    }
                )
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_pattern_hypotheses(
        self, 
        incidents: List[Incident], 
        events: List[Event]
    ) -> List[HuntingHypothesis]:
        """Generate hypotheses based on incident patterns"""
        
        hypotheses = []
        
        if len(incidents) < 2:
            return hypotheses
        
        # Look for patterns in recent incidents
        incident_patterns = defaultdict(list)
        
        for incident in incidents:
            # Group by reason
            incident_patterns[incident.reason].append(incident)
        
        # Generate hypotheses for patterns with multiple incidents
        for reason, reason_incidents in incident_patterns.items():
            if len(reason_incidents) >= 2:
                unique_ips = set(inc.src_ip for inc in reason_incidents)
                time_span = (max(inc.created_at for inc in reason_incidents) - 
                           min(inc.created_at for inc in reason_incidents)).total_seconds() / 3600
                
                if len(unique_ips) > 1 and time_span < 48:  # Multiple IPs in short time
                    hypothesis_text = f"Coordinated {reason} campaign detected: {len(unique_ips)} unique IPs in {time_span:.1f} hours"
                    
                    hypothesis = HuntingHypothesis(
                        hypothesis_id=f"campaign_{reason}_{int(datetime.utcnow().timestamp())}",
                        hypothesis_text=hypothesis_text,
                        threat_categories=[ThreatCategory.UNKNOWN],
                        confidence=0.7,
                        supporting_indicators=[
                            f"Incidents: {len(reason_incidents)}",
                            f"Unique IPs: {len(unique_ips)}",
                            f"Time window: {time_span:.1f} hours"
                        ],
                        hunting_queries=[
                            f"reason:{reason}",
                            f"src_ip:({' OR '.join(unique_ips)})"
                        ],
                        expected_evidence=['coordinated_timing', 'similar_tactics', 'infrastructure_overlap'],
                        priority_level='high',
                        created_timestamp=datetime.utcnow(),
                        validation_criteria={
                            'minimum_incidents': 2,
                            'maximum_time_window_hours': 48
                        }
                    )
                    
                    hypotheses.append(hypothesis)
        
        return hypotheses


class PredictiveThreatHunter:
    """
    Predictive Threat Hunting Agent
    Proactively identifies emerging threats using time-series analysis and behavioral predictions
    """
    
    def __init__(self):
        self.agent_id = "predictive_hunter_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core components
        self.time_series_predictor = TimeSeriesPredictor()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.hypothesis_generator = HypothesisGenerator()
        
        # Prediction cache and history
        self.active_predictions: Dict[str, ThreatPrediction] = {}
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.stats = {
            "predictions_generated": 0,
            "hypotheses_generated": 0,
            "successful_predictions": 0,
            "false_positives": 0,
            "baselines_created": 0,
            "anomalies_detected": 0,
            "last_activity": datetime.utcnow()
        }
        
        # Configuration
        self.prediction_thresholds = {
            'minimum_confidence': 0.6,
            'minimum_probability': 0.3,
            'maximum_predictions_per_run': 50
        }
    
    async def hunt_emerging_threats(
        self,
        incidents: List[Incident],
        events: List[Event],
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """
        Main threat hunting method - analyzes data to predict emerging threats
        
        Args:
            incidents: Recent incidents for analysis
            events: Recent events for analysis
            time_window: Time window for analysis
            
        Returns:
            Comprehensive threat hunting results with predictions and hypotheses
        """
        
        hunt_start = datetime.utcnow()
        hunt_id = f"hunt_{int(hunt_start.timestamp())}"
        
        try:
            self.logger.info(f"Starting predictive threat hunt {hunt_id}")
            
            # 1. Time-series analysis and prediction
            time_series_predictions = await self._perform_time_series_analysis(events, time_window)
            
            # 2. Behavioral analysis
            behavioral_analysis = await self._perform_behavioral_analysis(events, time_window)
            
            # 3. Generate hunting hypotheses
            hunting_hypotheses = await self.hypothesis_generator.generate_hypotheses(
                incidents, events, behavioral_analysis.get('anomalies', [])
            )
            
            # 4. Early warning system check
            early_warnings = await self._generate_early_warnings(
                time_series_predictions, behavioral_analysis, hunting_hypotheses
            )
            
            # 5. Predictive risk assessment
            risk_assessment = await self._calculate_predictive_risk_score(
                time_series_predictions, behavioral_analysis, incidents
            )
            
            # 6. Generate recommendations
            recommendations = await self._generate_hunting_recommendations(
                time_series_predictions, behavioral_analysis, hunting_hypotheses, risk_assessment
            )
            
            # Update statistics
            self.stats["predictions_generated"] += len(time_series_predictions)
            self.stats["hypotheses_generated"] += len(hunting_hypotheses)
            self.stats["anomalies_detected"] += len(behavioral_analysis.get('anomalies', []))
            self.stats["last_activity"] = datetime.utcnow()
            
            hunt_duration = (datetime.utcnow() - hunt_start).total_seconds()
            
            results = {
                "success": True,
                "hunt_id": hunt_id,
                "hunt_duration": hunt_duration,
                "analysis_summary": {
                    "events_analyzed": len(events),
                    "incidents_analyzed": len(incidents),
                    "time_window_hours": time_window.total_seconds() / 3600,
                    "predictions_generated": len(time_series_predictions),
                    "hypotheses_generated": len(hunting_hypotheses),
                    "anomalies_detected": len(behavioral_analysis.get('anomalies', []))
                },
                "threat_predictions": time_series_predictions,
                "behavioral_analysis": behavioral_analysis,
                "hunting_hypotheses": hunting_hypotheses,
                "early_warnings": early_warnings,
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "agent_performance": self.get_agent_status()
            }
            
            self.logger.info(f"Completed threat hunt {hunt_id} in {hunt_duration:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Threat hunt {hunt_id} failed: {e}")
            return {
                "success": False,
                "hunt_id": hunt_id,
                "error": str(e),
                "partial_results": {}
            }
    
    async def _perform_time_series_analysis(
        self, 
        events: List[Event], 
        time_window: timedelta
    ) -> List[Dict[str, Any]]:
        """Perform time-series analysis to predict future threats"""
        
        predictions = []
        
        try:
            # Prepare time-series data
            ts_data = self.time_series_predictor.prepare_time_series_data(events, timedelta(hours=1))
            
            if ts_data.empty:
                return predictions
            
            # Generate predictions for each threat category
            for threat_category in ThreatCategory:
                if threat_category == ThreatCategory.UNKNOWN:
                    continue
                
                for horizon in PredictionHorizon:
                    probability = await self.time_series_predictor.predict_threat_probability(
                        ts_data, horizon, threat_category
                    )
                    
                    if probability >= self.prediction_thresholds['minimum_probability']:
                        # Generate prediction details
                        prediction = await self._create_threat_prediction(
                            threat_category, probability, horizon, ts_data
                        )
                        
                        if prediction['confidence_score'] >= self.prediction_thresholds['minimum_confidence']:
                            predictions.append(prediction)
            
            # Sort by probability and confidence
            predictions.sort(key=lambda p: p['predicted_probability'] * p['confidence_score'], reverse=True)
            
            # Limit number of predictions
            return predictions[:self.prediction_thresholds['maximum_predictions_per_run']]
            
        except Exception as e:
            self.logger.error(f"Time-series analysis failed: {e}")
            return []
    
    async def _create_threat_prediction(
        self,
        threat_category: ThreatCategory,
        probability: float,
        horizon: PredictionHorizon,
        time_series_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create a detailed threat prediction"""
        
        prediction_id = f"pred_{threat_category.value}_{horizon.name}_{int(datetime.utcnow().timestamp())}"
        
        # Analyze contributing factors
        contributing_factors = []
        if not time_series_data.empty:
            recent_trend = time_series_data['eventid'].tail(5).mean()
            historical_avg = time_series_data['eventid'].mean()
            
            if recent_trend > historical_avg * 1.2:
                contributing_factors.append("Increasing event frequency")
            
            unique_ip_trend = time_series_data['src_ip'].tail(5).mean()
            if unique_ip_trend > time_series_data['src_ip'].mean() * 1.1:
                contributing_factors.append("Growing number of unique source IPs")
        
        # Generate early warning indicators
        early_warnings = []
        if probability > 0.7:
            early_warnings.append({
                "indicator": "High probability threat prediction",
                "threshold": 0.7,
                "current_value": probability,
                "action_required": "Immediate monitoring"
            })
        
        # Generate mitigation recommendations
        mitigations = {
            ThreatCategory.BRUTE_FORCE: [
                "Implement rate limiting on authentication endpoints",
                "Deploy fail2ban or similar IP blocking",
                "Monitor for credential stuffing patterns"
            ],
            ThreatCategory.MALWARE: [
                "Enhance file scanning and sandboxing",
                "Monitor for suspicious download patterns",
                "Update threat detection signatures"
            ],
            ThreatCategory.RECONNAISSANCE: [
                "Deploy network deception technologies",
                "Monitor for port scanning activities",
                "Implement network segmentation"
            ]
        }.get(threat_category, ["Monitor threat indicators", "Review security controls"])
        
        # Calculate confidence based on data quality and model performance
        data_quality_score = min(1.0, len(time_series_data) / 100)
        model_accuracy = self.time_series_predictor.model_accuracy.get(threat_category.value, 0.7)
        confidence_score = (probability * 0.4 + data_quality_score * 0.3 + model_accuracy * 0.3)
        
        return {
            "prediction_id": prediction_id,
            "threat_category": threat_category.value,
            "predicted_probability": probability,
            "confidence_score": confidence_score,
            "prediction_horizon": {
                "name": horizon.name,
                "duration_hours": horizon.value.total_seconds() / 3600
            },
            "contributing_factors": contributing_factors,
            "early_warning_indicators": early_warnings,
            "mitigation_recommendations": mitigations,
            "model_used": ThreatPredictionModel.TIME_SERIES_ARIMA.value,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "supporting_evidence": {
                "data_points_analyzed": len(time_series_data),
                "time_series_trend": "increasing" if not time_series_data.empty and 
                                   time_series_data['eventid'].tail(5).mean() > time_series_data['eventid'].mean() 
                                   else "stable",
                "confidence_factors": {
                    "data_quality": data_quality_score,
                    "model_accuracy": model_accuracy,
                    "prediction_strength": probability
                }
            }
        }
    
    async def _perform_behavioral_analysis(
        self, 
        events: List[Event], 
        time_window: timedelta
    ) -> Dict[str, Any]:
        """Perform behavioral analysis to detect anomalies"""
        
        analysis_results = {
            "baselines_analyzed": 0,
            "anomalies": [],
            "behavioral_insights": [],
            "baseline_health": {}
        }
        
        try:
            # Group events by entity (IP addresses)
            events_by_ip = defaultdict(list)
            for event in events:
                events_by_ip[event.src_ip].append(event)
            
            # Analyze each IP's behavior
            for src_ip, ip_events in events_by_ip.items():
                if len(ip_events) < 10:  # Skip low-activity IPs
                    continue
                
                # Create or update baseline
                baseline = await self.behavioral_analyzer.create_behavioral_baseline(
                    entity_type="ip",
                    entity_id=src_ip,
                    historical_events=ip_events,
                    time_window=time_window
                )
                
                analysis_results["baselines_analyzed"] += 1
                self.stats["baselines_created"] += 1
                
                # Detect anomalies
                recent_events = [e for e in ip_events if (datetime.utcnow() - e.ts) <= timedelta(hours=4)]
                if recent_events:
                    anomalies = await self.behavioral_analyzer.detect_anomalies(
                        entity_type="ip",
                        entity_id=src_ip,
                        recent_events=recent_events
                    )
                    
                    for anomaly in anomalies:
                        anomaly['entity_type'] = 'ip'
                        anomaly['entity_id'] = src_ip
                        anomaly['baseline_confidence'] = baseline.confidence_level
                        analysis_results["anomalies"].append(anomaly)
                
                # Generate behavioral insights
                if baseline.confidence_level > 0.7:
                    insights = self._generate_behavioral_insights(baseline, ip_events)
                    analysis_results["behavioral_insights"].extend(insights)
                
                # Track baseline health
                analysis_results["baseline_health"][src_ip] = {
                    "confidence": baseline.confidence_level,
                    "data_points": len(ip_events),
                    "last_updated": baseline.last_updated.isoformat()
                }
            
            # Sort anomalies by severity and deviation
            analysis_results["anomalies"].sort(
                key=lambda a: (a['severity'] == 'high', a['deviation_magnitude']), 
                reverse=True
            )
            
        except Exception as e:
            self.logger.error(f"Behavioral analysis failed: {e}")
        
        return analysis_results
    
    def _generate_behavioral_insights(
        self, 
        baseline: BehavioralBaseline, 
        events: List[Event]
    ) -> List[Dict[str, Any]]:
        """Generate insights from behavioral analysis"""
        
        insights = []
        
        # Activity pattern insights
        if baseline.baseline_metrics.get('events_per_hour_mean', 0) > 10:
            insights.append({
                "type": "high_activity",
                "entity_id": baseline.entity_id,
                "description": f"High activity entity: {baseline.baseline_metrics['events_per_hour_mean']:.1f} events/hour on average",
                "risk_level": "medium",
                "recommendation": "Monitor for potential malicious activity"
            })
        
        # Time pattern insights
        business_hours_ratio = baseline.baseline_metrics.get('is_business_hours_mean', 0)
        if business_hours_ratio < 0.2:  # Mostly active outside business hours
            insights.append({
                "type": "off_hours_activity",
                "entity_id": baseline.entity_id,
                "description": f"Primarily active outside business hours ({(1-business_hours_ratio)*100:.1f}% of activity)",
                "risk_level": "medium",
                "recommendation": "Investigate legitimacy of off-hours access"
            })
        
        return insights
    
    async def _generate_early_warnings(
        self,
        predictions: List[Dict[str, Any]],
        behavioral_analysis: Dict[str, Any],
        hypotheses: List[HuntingHypothesis]
    ) -> List[Dict[str, Any]]:
        """Generate early warning alerts"""
        
        warnings = []
        
        # High-probability predictions trigger warnings
        for prediction in predictions:
            if prediction['predicted_probability'] > 0.8:
                warnings.append({
                    "type": "high_probability_threat",
                    "threat_category": prediction['threat_category'],
                    "probability": prediction['predicted_probability'],
                    "time_horizon": prediction['prediction_horizon']['duration_hours'],
                    "severity": "critical",
                    "message": f"High probability {prediction['threat_category']} threat predicted in next {prediction['prediction_horizon']['duration_hours']:.1f} hours",
                    "recommended_actions": prediction['mitigation_recommendations'][:3]
                })
        
        # High-severity anomalies trigger warnings
        high_severity_anomalies = [a for a in behavioral_analysis.get('anomalies', []) if a['severity'] == 'high']
        if len(high_severity_anomalies) > 3:
            warnings.append({
                "type": "multiple_behavioral_anomalies",
                "anomaly_count": len(high_severity_anomalies),
                "severity": "high",
                "message": f"Multiple high-severity behavioral anomalies detected ({len(high_severity_anomalies)} anomalies)",
                "recommended_actions": [
                    "Investigate anomalous entities",
                    "Review recent security events",
                    "Consider increasing monitoring levels"
                ]
            })
        
        # High-confidence hypotheses trigger warnings
        high_confidence_hypotheses = [h for h in hypotheses if h.confidence > 0.8]
        if high_confidence_hypotheses:
            warnings.append({
                "type": "high_confidence_hypotheses",
                "hypothesis_count": len(high_confidence_hypotheses),
                "severity": "medium",
                "message": f"High-confidence threat hunting hypotheses generated ({len(high_confidence_hypotheses)} hypotheses)",
                "recommended_actions": [
                    "Execute hunting queries",
                    "Validate hypotheses with threat intelligence",
                    "Deploy additional monitoring"
                ]
            })
        
        return warnings
    
    async def _calculate_predictive_risk_score(
        self,
        predictions: List[Dict[str, Any]],
        behavioral_analysis: Dict[str, Any],
        incidents: List[Incident]
    ) -> Dict[str, Any]:
        """Calculate overall predictive risk score"""
        
        risk_components = {
            "prediction_risk": 0.0,
            "behavioral_risk": 0.0,
            "incident_trend_risk": 0.0,
            "overall_risk": 0.0
        }
        
        # Prediction-based risk
        if predictions:
            high_prob_predictions = [p for p in predictions if p['predicted_probability'] > 0.7]
            prediction_risk = min(1.0, len(high_prob_predictions) * 0.2 + 
                                 sum(p['predicted_probability'] for p in predictions[:5]) / 5)
            risk_components["prediction_risk"] = prediction_risk
        
        # Behavioral anomaly risk
        anomalies = behavioral_analysis.get('anomalies', [])
        if anomalies:
            high_severity_count = sum(1 for a in anomalies if a['severity'] == 'high')
            behavioral_risk = min(1.0, high_severity_count * 0.1 + len(anomalies) * 0.05)
            risk_components["behavioral_risk"] = behavioral_risk
        
        # Incident trend risk
        if incidents:
            recent_incidents = [i for i in incidents if (datetime.utcnow() - i.created_at).days <= 7]
            if len(incidents) > 0:
                trend_risk = min(1.0, len(recent_incidents) / len(incidents))
                risk_components["incident_trend_risk"] = trend_risk
        
        # Calculate overall risk (weighted average)
        weights = {"prediction": 0.4, "behavioral": 0.35, "incident_trend": 0.25}
        overall_risk = (
            risk_components["prediction_risk"] * weights["prediction"] +
            risk_components["behavioral_risk"] * weights["behavioral"] +
            risk_components["incident_trend_risk"] * weights["incident_trend"]
        )
        
        risk_components["overall_risk"] = overall_risk
        
        # Determine risk level
        if overall_risk >= 0.8:
            risk_level = "critical"
        elif overall_risk >= 0.6:
            risk_level = "high"
        elif overall_risk >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "overall_risk_score": overall_risk,
            "risk_level": risk_level,
            "risk_components": risk_components,
            "risk_factors": [
                f"Prediction risk: {risk_components['prediction_risk']:.2f}",
                f"Behavioral risk: {risk_components['behavioral_risk']:.2f}",
                f"Incident trend risk: {risk_components['incident_trend_risk']:.2f}"
            ],
            "confidence": 0.8  # Confidence in risk assessment
        }
    
    async def _generate_hunting_recommendations(
        self,
        predictions: List[Dict[str, Any]],
        behavioral_analysis: Dict[str, Any],
        hypotheses: List[HuntingHypothesis],
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable threat hunting recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        risk_level = risk_assessment.get("risk_level", "low")
        
        if risk_level == "critical":
            recommendations.extend([
                "Implement immediate threat monitoring and alerting",
                "Activate incident response procedures",
                "Deploy additional security controls proactively"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Increase monitoring frequency and coverage",
                "Review and validate high-confidence predictions",
                "Prepare incident response teams"
            ])
        
        # Prediction-based recommendations
        high_prob_predictions = [p for p in predictions if p['predicted_probability'] > 0.6]
        if high_prob_predictions:
            top_threat = max(high_prob_predictions, key=lambda x: x['predicted_probability'])
            recommendations.append(
                f"Focus monitoring on {top_threat['threat_category']} threats - highest predicted probability ({top_threat['predicted_probability']:.2f})"
            )
        
        # Hypothesis-based recommendations
        if len(hypotheses) >= 5:
            recommendations.append(f"Execute {min(len(hypotheses), 10)} threat hunting hypotheses, starting with highest confidence")
        
        # Behavioral analysis recommendations
        anomaly_count = len(behavioral_analysis.get('anomalies', []))
        if anomaly_count > 5:
            recommendations.append(f"Investigate {anomaly_count} behavioral anomalies, prioritize high-severity findings")
        
        # Data quality recommendations
        baseline_health = behavioral_analysis.get('baseline_health', {})
        low_confidence_baselines = [k for k, v in baseline_health.items() if v['confidence'] < 0.6]
        
        if len(low_confidence_baselines) > len(baseline_health) * 0.3:
            recommendations.append("Improve behavioral baseline data quality - extend monitoring periods for better accuracy")
        
        # Generic recommendations
        if not recommendations:
            recommendations.extend([
                "Continue proactive threat monitoring",
                "Review and update threat hunting procedures",
                "Maintain current security posture"
            ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "sklearn_available": SKLEARN_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "statistics": self.stats.copy(),
            "active_predictions": len(self.active_predictions),
            "generated_hypotheses": len(self.hypothesis_generator.generated_hypotheses),
            "behavioral_baselines": len(self.behavioral_analyzer.baselines),
            "configuration": self.prediction_thresholds.copy(),
            "model_performance": dict(self.time_series_predictor.model_accuracy),
            "capabilities": [
                "time_series_prediction",
                "behavioral_analysis", 
                "hypothesis_generation",
                "anomaly_detection",
                "early_warning_system",
                "predictive_risk_assessment"
            ]
        }


# Global predictive hunter instance
predictive_hunter = PredictiveThreatHunter()


async def get_predictive_hunter() -> PredictiveThreatHunter:
    """Get the global predictive hunter instance"""
    return predictive_hunter
