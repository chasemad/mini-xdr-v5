#!/usr/bin/env python3
"""
📊 SAGEMAKER TRAINING MONITOR
Real-time monitoring of training job for error detection and progress tracking
"""

import boto3
import time
import json
from datetime import datetime, timezone
import sys

class TrainingMonitor:
    def __init__(self, training_job_name):
        self.training_job_name = training_job_name
        self.sagemaker = boto3.client('sagemaker')
        self.logs = boto3.client('logs')
        self.log_group = "/aws/sagemaker/TrainingJobs"
        self.log_stream = None
        self.last_timestamp = 0

    def get_job_status(self):
        """Get current training job status"""
        try:
            response = self.sagemaker.describe_training_job(
                TrainingJobName=self.training_job_name
            )
            return {
                'status': response['TrainingJobStatus'],
                'secondary_status': response.get('SecondaryStatus', ''),
                'failure_reason': response.get('FailureReason', ''),
                'training_time': response.get('TrainingTimeInSeconds', 0),
                'billable_time': response.get('BillableTimeInSeconds', 0)
            }
        except Exception as e:
            return {'error': str(e)}

    def find_log_stream(self):
        """Find the log stream for this training job"""
        if self.log_stream:
            return self.log_stream

        try:
            response = self.logs.describe_log_streams(
                logGroupName=self.log_group,
                orderBy='LastEventTime',
                descending=True,
                limit=10
            )

            for stream in response['logStreams']:
                if self.training_job_name in stream['logStreamName']:
                    self.log_stream = stream['logStreamName']
                    return self.log_stream

        except Exception as e:
            print(f"Error finding log stream: {e}")

        return None

    def get_recent_logs(self, limit=20):
        """Get recent log entries"""
        log_stream = self.find_log_stream()
        if not log_stream:
            return []

        try:
            kwargs = {
                'logGroupName': self.log_group,
                'logStreamName': log_stream,
                'limit': limit
            }

            if self.last_timestamp > 0:
                kwargs['startTime'] = self.last_timestamp + 1

            response = self.logs.get_log_events(**kwargs)
            events = response.get('events', [])

            if events:
                self.last_timestamp = events[-1]['timestamp']

            return events

        except Exception as e:
            print(f"Error getting logs: {e}")
            return []

    def analyze_logs_for_issues(self, events):
        """Analyze log events for potential issues"""
        issues = []
        warnings = []

        for event in events:
            message = event['message'].lower()

            # Critical errors
            if any(error in message for error in ['error', 'exception', 'failed', 'cuda error']):
                issues.append({
                    'type': 'ERROR',
                    'timestamp': datetime.fromtimestamp(event['timestamp']/1000).strftime('%H:%M:%S'),
                    'message': event['message']
                })

            # Warnings
            elif any(warn in message for warn in ['warning', 'deprecated', 'memory']):
                warnings.append({
                    'type': 'WARNING',
                    'timestamp': datetime.fromtimestamp(event['timestamp']/1000).strftime('%H:%M:%S'),
                    'message': event['message']
                })

        return issues, warnings

    def extract_training_metrics(self, events):
        """Extract training metrics from logs"""
        metrics = {}

        for event in events:
            message = event['message']

            # Look for epoch progress
            if 'Epoch' in message and '%' in message:
                try:
                    if '|' in message:
                        parts = message.split('|')
                        if len(parts) >= 2:
                            percent_part = parts[1].strip()
                            if '%' in percent_part:
                                epoch_progress = percent_part.split('%')[0].strip()
                                metrics['current_epoch_progress'] = f"{epoch_progress}%"
                except:
                    pass

            # Look for training metrics
            if 'Train Loss:' in message:
                try:
                    loss = message.split('Train Loss:')[1].strip().split()[0]
                    metrics['train_loss'] = float(loss)
                except:
                    pass

            if 'Train Acc:' in message:
                try:
                    acc = message.split('Train Acc:')[1].strip().split()[0]
                    metrics['train_accuracy'] = float(acc)
                except:
                    pass

        return metrics

    def monitor_once(self):
        """Single monitoring check"""
        print(f"🔍 Monitoring Training Job: {self.training_job_name}")
        print("=" * 60)

        # Get job status
        status = self.get_job_status()
        if 'error' in status:
            print(f"❌ Error getting job status: {status['error']}")
            return False

        print(f"📊 Status: {status['status']}")
        print(f"🔄 Secondary: {status['secondary_status']}")
        print(f"⏱️ Training Time: {status['training_time']}s ({status['training_time']/60:.1f} min)")

        if status['failure_reason']:
            print(f"❌ Failure Reason: {status['failure_reason']}")
            return False

        # Get recent logs
        events = self.get_recent_logs()
        if events:
            print(f"\n📝 Recent Logs ({len(events)} entries):")

            # Analyze for issues
            issues, warnings = self.analyze_logs_for_issues(events)

            if issues:
                print(f"\n🚨 CRITICAL ISSUES DETECTED ({len(issues)}):")
                for issue in issues[-3:]:  # Show last 3 issues
                    print(f"  [{issue['timestamp']}] {issue['message']}")

            if warnings:
                print(f"\n⚠️ WARNINGS ({len(warnings)}):")
                for warning in warnings[-2:]:  # Show last 2 warnings
                    print(f"  [{warning['timestamp']}] {warning['message']}")

            # Extract metrics
            metrics = self.extract_training_metrics(events)
            if metrics:
                print(f"\n📊 Training Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")

            # Show recent progress
            print(f"\n📈 Recent Progress:")
            for event in events[-3:]:  # Last 3 log entries
                timestamp = datetime.fromtimestamp(event['timestamp']/1000).strftime('%H:%M:%S')
                message = event['message'][:100] + "..." if len(event['message']) > 100 else event['message']
                print(f"  [{timestamp}] {message}")

        print("\n" + "=" * 60)

        return status['status'] == 'InProgress'

    def continuous_monitor(self, check_interval=60):
        """Continuously monitor training job"""
        print(f"🚀 Starting continuous monitoring (checking every {check_interval}s)")
        print("Press Ctrl+C to stop monitoring")

        try:
            while True:
                still_running = self.monitor_once()

                if not still_running:
                    print("🏁 Training job completed or failed. Stopping monitor.")
                    break

                print(f"⏳ Waiting {check_interval}s for next check...\n")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")

def main():
    training_job_name = "enhanced-xdr-20250929-063500"

    if len(sys.argv) > 1:
        training_job_name = sys.argv[1]

    monitor = TrainingMonitor(training_job_name)

    # Single check or continuous monitoring
    if len(sys.argv) > 2 and sys.argv[2] == "--continuous":
        monitor.continuous_monitor()
    else:
        monitor.monitor_once()

if __name__ == '__main__':
    main()