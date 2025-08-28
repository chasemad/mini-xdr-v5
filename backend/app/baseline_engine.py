"""
Statistical Baseline Engine for Learning Normal Behavior Patterns
Continuously learns and maintains statistical baselines for adaptive detection
"""
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import numpy as np
# from scipy import stats  # Disabled for macOS compatibility
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Event, Incident
from .config import settings

logger = logging.getLogger(__name__)


class BaselineEngine:
    """Learns and maintains statistical baselines for normal behavior"""
    
    def __init__(self):
        self.baselines = {
            'per_ip': defaultdict(dict),     # IP-specific baselines
            'global': {},                    # System-wide baselines
            'temporal': defaultdict(dict)    # Time-based patterns
        }
        
        # Learning parameters
        self.learning_window_days = 7
        self.clean_data_threshold = 0.95  # Only learn from 95% clean data
        self.min_samples_for_baseline = 10
        self.baseline_confidence_threshold = 0.7
        
        # Deviation thresholds for anomaly detection
        self.deviation_thresholds = {
            'minor': 2.0,    # 2 standard deviations
            'moderate': 3.0, # 3 standard deviations  
            'severe': 4.0    # 4 standard deviations
        }
        
        # Track last deviations for reporting
        self.last_deviations = {}
        
    async def update_baselines(self, db: AsyncSession):
        """Continuously update behavioral baselines from historical data"""
        logger.info("Starting baseline update process")
        
        try:
            # Learn normal request patterns
            await self._learn_request_patterns(db)
            
            # Learn normal timing patterns
            await self._learn_temporal_patterns(db)
            
            # Learn normal error rates
            await self._learn_error_patterns(db)
            
            # Learn normal parameter usage
            await self._learn_parameter_patterns(db)
            
            # Learn global system patterns
            await self._learn_global_patterns(db)
            
            logger.info("Baseline update completed successfully")
            
        except Exception as e:
            logger.error(f"Baseline update failed: {e}")
    
    async def _get_clean_data_period(self, db: AsyncSession, days: int = 7) -> Dict[str, List[Event]]:
        """Get events from a period with minimal incidents (clean data)"""
        
        # Find periods with few incidents
        window_start = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get incident count by day
        daily_incidents = await self._get_daily_incident_counts(db, window_start)
        
        # Find cleanest days (lowest incident count)
        clean_days = sorted(daily_incidents.items(), key=lambda x: x[1])[:max(1, days // 2)]
        clean_day_dates = [day for day, count in clean_days]
        
        # Get events from clean days
        clean_events = {}
        for clean_date in clean_day_dates:
            day_start = clean_date
            day_end = day_start + timedelta(days=1)
            
            query = select(Event).where(
                and_(
                    Event.ts >= day_start,
                    Event.ts < day_end
                )
            ).order_by(Event.ts)
            
            result = await db.execute(query)
            events = result.scalars().all()
            
            # Group by IP
            for event in events:
                if event.src_ip not in clean_events:
                    clean_events[event.src_ip] = []
                clean_events[event.src_ip].append(event)
        
        # Filter out IPs with incidents during this period
        incident_ips = await self._get_incident_ips_in_period(db, window_start)
        clean_events = {ip: events for ip, events in clean_events.items() 
                       if ip not in incident_ips and len(events) >= self.min_samples_for_baseline}
        
        logger.info(f"Found clean data for {len(clean_events)} IPs from {len(clean_day_dates)} clean days")
        return clean_events
    
    async def _get_daily_incident_counts(self, db: AsyncSession, start_date: datetime) -> Dict[datetime, int]:
        """Get incident count by day"""
        query = select(
            func.date(Incident.created_at).label('incident_date'),
            func.count(Incident.id).label('incident_count')
        ).where(
            Incident.created_at >= start_date
        ).group_by(
            func.date(Incident.created_at)
        )
        
        result = await db.execute(query)
        return {row.incident_date: row.incident_count for row in result}
    
    async def _get_incident_ips_in_period(self, db: AsyncSession, start_date: datetime) -> set:
        """Get IPs that had incidents in the given period"""
        query = select(Incident.src_ip.distinct()).where(
            Incident.created_at >= start_date
        )
        
        result = await db.execute(query)
        return {row[0] for row in result}
    
    async def _learn_request_patterns(self, db: AsyncSession):
        """Learn what constitutes normal request behavior"""
        logger.info("Learning request patterns")
        
        clean_period = await self._get_clean_data_period(db, self.learning_window_days)
        
        for ip, events in clean_period.items():
            if len(events) < self.min_samples_for_baseline:
                continue
                
            # Calculate baseline metrics for this IP
            baseline_metrics = await self._calculate_request_baselines(events)
            self.baselines['per_ip'][ip].update(baseline_metrics)
        
        logger.info(f"Learned request patterns for {len(clean_period)} IPs")
    
    async def _calculate_request_baselines(self, events: List[Event]) -> Dict[str, Any]:
        """Calculate baseline request metrics for an IP"""
        baselines = {}
        
        # Request rate analysis
        if len(events) > 1:
            time_span = (events[-1].ts - events[0].ts).total_seconds()
            requests_per_hour = len(events) / max(time_span / 3600, 1)
            baselines['avg_requests_per_hour'] = requests_per_hour
            baselines['requests_per_hour_std'] = 0  # Will be updated with more data
        
        # Extract request details
        paths = []
        user_agents = []
        status_codes = []
        parameters = []
        
        for event in events:
            try:
                raw_data = event.raw if isinstance(event.raw, dict) else json.loads(event.raw) if event.raw else {}
                
                if 'path' in raw_data:
                    paths.append(raw_data['path'])
                if 'user_agent' in raw_data:
                    user_agents.append(raw_data['user_agent'])
                if 'status_code' in raw_data:
                    status_codes.append(raw_data['status_code'])
                if 'parameters' in raw_data:
                    parameters.extend(raw_data['parameters'])
                    
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Path patterns
        baselines['typical_paths'] = list(set(paths))
        baselines['path_diversity'] = len(set(paths)) / max(len(events), 1)
        
        # User agent patterns
        baselines['normal_user_agents'] = list(set(user_agents))
        baselines['user_agent_diversity'] = len(set(user_agents))
        
        # Error rate
        error_responses = sum(1 for code in status_codes if code >= 400)
        baselines['typical_error_rate'] = error_responses / max(len(events), 1)
        
        # Parameter complexity
        if parameters:
            param_lengths = [len(str(p)) for p in parameters]
            baselines['parameter_complexity_avg'] = np.mean(param_lengths)
            baselines['parameter_complexity_std'] = np.std(param_lengths)
        else:
            baselines['parameter_complexity_avg'] = 0
            baselines['parameter_complexity_std'] = 0
        
        return baselines
    
    async def _learn_temporal_patterns(self, db: AsyncSession):
        """Learn normal timing patterns"""
        logger.info("Learning temporal patterns")
        
        clean_period = await self._get_clean_data_period(db, self.learning_window_days)
        
        temporal_data = defaultdict(list)
        
        for ip, events in clean_period.items():
            for event in events:
                hour = event.ts.hour
                day_of_week = event.ts.weekday()
                
                temporal_data[f"hour_{hour}"].append(1)
                temporal_data[f"dow_{day_of_week}"].append(1)
                temporal_data[f"ip_{ip}_hour_{hour}"].append(1)
        
        # Calculate temporal baselines
        for time_key, occurrences in temporal_data.items():
            self.baselines['temporal'][time_key] = {
                'avg_activity': np.mean(occurrences),
                'std_activity': np.std(occurrences),
                'total_samples': len(occurrences)
            }
        
        logger.info(f"Learned temporal patterns for {len(temporal_data)} time periods")
    
    async def _learn_error_patterns(self, db: AsyncSession):
        """Learn normal error rates and patterns"""
        logger.info("Learning error patterns")
        
        clean_period = await self._get_clean_data_period(db, self.learning_window_days)
        
        error_metrics = []
        
        for ip, events in clean_period.items():
            error_count = 0
            total_requests = 0
            
            for event in events:
                try:
                    raw_data = event.raw if isinstance(event.raw, dict) else json.loads(event.raw) if event.raw else {}
                    if 'status_code' in raw_data:
                        total_requests += 1
                        if raw_data['status_code'] >= 400:
                            error_count += 1
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if total_requests > 0:
                error_rate = error_count / total_requests
                error_metrics.append(error_rate)
        
        if error_metrics:
            self.baselines['global']['normal_error_rate_mean'] = np.mean(error_metrics)
            self.baselines['global']['normal_error_rate_std'] = np.std(error_metrics)
        
        logger.info(f"Learned error patterns from {len(error_metrics)} IP samples")
    
    async def _learn_parameter_patterns(self, db: AsyncSession):
        """Learn normal parameter usage patterns"""
        logger.info("Learning parameter patterns")
        
        clean_period = await self._get_clean_data_period(db, self.learning_window_days)
        
        param_metrics = []
        
        for ip, events in clean_period.items():
            parameters = []
            
            for event in events:
                try:
                    raw_data = event.raw if isinstance(event.raw, dict) else json.loads(event.raw) if event.raw else {}
                    if 'parameters' in raw_data:
                        parameters.extend(raw_data['parameters'])
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if parameters:
                unique_params = len(set(parameters))
                param_diversity = unique_params / max(len(parameters), 1)
                param_metrics.append({
                    'unique_params': unique_params,
                    'param_diversity': param_diversity,
                    'avg_param_length': np.mean([len(str(p)) for p in parameters])
                })
        
        if param_metrics:
            self.baselines['global']['normal_param_diversity_mean'] = np.mean([m['param_diversity'] for m in param_metrics])
            self.baselines['global']['normal_param_diversity_std'] = np.std([m['param_diversity'] for m in param_metrics])
            self.baselines['global']['normal_unique_params_mean'] = np.mean([m['unique_params'] for m in param_metrics])
            self.baselines['global']['normal_unique_params_std'] = np.std([m['unique_params'] for m in param_metrics])
        
        logger.info(f"Learned parameter patterns from {len(param_metrics)} IP samples")
    
    async def _learn_global_patterns(self, db: AsyncSession):
        """Learn system-wide baseline patterns"""
        logger.info("Learning global patterns")
        
        # Global request rate
        window_start = datetime.now(timezone.utc) - timedelta(days=self.learning_window_days)
        
        query = select(func.count(Event.id)).where(Event.ts >= window_start)
        result = await db.execute(query)
        total_events = result.scalar()
        
        hours_in_window = self.learning_window_days * 24
        global_requests_per_hour = total_events / max(hours_in_window, 1)
        
        self.baselines['global']['avg_global_requests_per_hour'] = global_requests_per_hour
        
        logger.info(f"Global baseline: {global_requests_per_hour:.1f} requests/hour")
    
    async def calculate_deviation(self, db: AsyncSession, src_ip: str) -> float:
        """Calculate deviation from learned baselines for an IP"""
        
        if src_ip not in self.baselines['per_ip']:
            return 0.0  # No baseline available
        
        # Get recent activity for this IP
        recent_events = await self._get_recent_events(db, src_ip, 60)  # Last hour
        
        if not recent_events:
            return 0.0
        
        deviation_score = 0.0
        deviation_count = 0
        
        self.last_deviations = {}
        
        # Calculate current metrics
        current_metrics = await self._calculate_request_baselines(recent_events)
        baseline_metrics = self.baselines['per_ip'][src_ip]
        
        # Compare current behavior to baseline
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                
                # Skip non-numeric metrics
                if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                    continue
                
                # Calculate z-score if we have standard deviation
                std_key = f"{metric}_std"
                if std_key in baseline_metrics and baseline_metrics[std_key] > 0:
                    z_score = abs(current_value - baseline_value) / baseline_metrics[std_key]
                    
                    # Determine deviation level
                    if z_score > self.deviation_thresholds['severe']:
                        deviation_score += 0.5
                        self.last_deviations[metric] = 'severe'
                    elif z_score > self.deviation_thresholds['moderate']:
                        deviation_score += 0.3
                        self.last_deviations[metric] = 'moderate'
                    elif z_score > self.deviation_thresholds['minor']:
                        deviation_score += 0.1
                        self.last_deviations[metric] = 'minor'
                    
                    deviation_count += 1
                else:
                    # Simple percentage deviation
                    if baseline_value > 0:
                        percent_deviation = abs(current_value - baseline_value) / baseline_value
                        if percent_deviation > 2.0:  # 200% deviation
                            deviation_score += 0.3
                            self.last_deviations[metric] = 'high'
                        elif percent_deviation > 1.0:  # 100% deviation
                            deviation_score += 0.2
                            self.last_deviations[metric] = 'moderate'
                        elif percent_deviation > 0.5:  # 50% deviation
                            deviation_score += 0.1
                            self.last_deviations[metric] = 'minor'
                        
                        deviation_count += 1
        
        # Normalize by number of metrics compared
        if deviation_count > 0:
            normalized_deviation = deviation_score / deviation_count
            return min(normalized_deviation, 1.0)
        
        return 0.0
    
    async def _get_recent_events(self, db: AsyncSession, src_ip: str, minutes: int) -> List[Event]:
        """Get recent events for an IP"""
        window_start = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        query = select(Event).where(
            and_(
                Event.src_ip == src_ip,
                Event.ts >= window_start
            )
        ).order_by(Event.ts.desc())
        
        result = await db.execute(query)
        return result.scalars().all()
    
    def get_baseline_status(self) -> Dict[str, Any]:
        """Get current baseline status and statistics"""
        return {
            'per_ip_baselines': len(self.baselines['per_ip']),
            'global_baselines': len(self.baselines['global']),
            'temporal_baselines': len(self.baselines['temporal']),
            'learning_window_days': self.learning_window_days,
            'last_deviations': self.last_deviations,
            'baseline_sample': {
                'sample_ip_baseline': list(self.baselines['per_ip'].keys())[:3] if self.baselines['per_ip'] else [],
                'global_metrics': list(self.baselines['global'].keys()),
                'temporal_periods': list(self.baselines['temporal'].keys())[:5]
            }
        }
    
    def adjust_sensitivity(self, sensitivity_level: str):
        """Adjust detection sensitivity by modifying thresholds"""
        if sensitivity_level == "high":
            self.deviation_thresholds = {
                'minor': 1.5,
                'moderate': 2.0,
                'severe': 2.5
            }
            self.baseline_confidence_threshold = 0.6
        elif sensitivity_level == "low":
            self.deviation_thresholds = {
                'minor': 3.0,
                'moderate': 4.0,
                'severe': 5.0
            }
            self.baseline_confidence_threshold = 0.8
        else:  # medium (default)
            self.deviation_thresholds = {
                'minor': 2.0,
                'moderate': 3.0,
                'severe': 4.0
            }
            self.baseline_confidence_threshold = 0.7
        
        logger.info(f"Adjusted sensitivity to {sensitivity_level}")


# Global baseline engine instance
baseline_engine = BaselineEngine()
