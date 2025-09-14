"""
Apache Kafka Manager for Distributed MCP
Handles topic management, message routing, and high-performance distributed messaging
for the Mini-XDR distributed architecture.
"""

import asyncio
import logging
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Kafka imports
try:
    import aiokafka
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer, AIOKafkaAdminClient
    from aiokafka.admin import ConfigResource, ConfigResourceType, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError, KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    aiokafka = None
    logging.warning("AioKafka not available - distributed messaging disabled")

from ..config import settings

logger = logging.getLogger(__name__)


class TopicType(str, Enum):
    """Kafka topic types for MCP messaging"""
    COMMANDS = "commands"         # Tool execution commands
    RESPONSES = "responses"       # Tool execution responses
    HEARTBEATS = "heartbeats"     # Node health monitoring
    DISCOVERY = "discovery"       # Node discovery and registration
    BROADCASTS = "broadcasts"     # System-wide announcements
    COORDINATION = "coordination" # Distributed coordination
    LOGS = "logs"                # Distributed logging
    METRICS = "metrics"          # Performance metrics


class MessagePriority(int, Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class TopicConfig:
    """Kafka topic configuration"""
    name: str
    partitions: int = 3
    replication_factor: int = 1  # Increase in production
    retention_ms: int = 86400000  # 24 hours
    compression_type: str = "snappy"
    cleanup_policy: str = "delete"
    max_message_bytes: int = 1048576  # 1MB


@dataclass
class MessageMetrics:
    """Message processing metrics"""
    total_sent: int = 0
    total_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    last_activity: float = 0.0


class KafkaManager:
    """
    Advanced Kafka Manager for Distributed MCP
    Handles high-performance messaging, topic management, and message routing
    """
    
    def __init__(self, node_id: str, region: str = "us-west-1"):
        self.node_id = node_id
        self.region = region
        
        # Kafka configuration
        self.kafka_config = {
            'bootstrap_servers': getattr(settings, 'kafka_servers', 'localhost:9092').split(','),
            'security_protocol': 'PLAINTEXT',  # Use SSL/SASL in production
            'client_id': f'mcp-kafka-{node_id}',
            'api_version': 'auto',
            'connections_max_idle_ms': 540000,
            'request_timeout_ms': 40000,
            'retry_backoff_ms': 100
        }
        
        # Kafka clients
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.admin_client: Optional[AIOKafkaAdminClient] = None
        
        # Topic management
        self.topics: Dict[TopicType, TopicConfig] = {}
        self.consumer_groups: Set[str] = set()
        
        # Message routing and handlers
        self.message_handlers: Dict[str, Callable] = {}
        self.message_filters: Dict[str, Callable] = {}
        
        # Performance metrics
        self.metrics: Dict[str, MessageMetrics] = {}
        
        # Consumer management
        self.running = False
        self.consumer_tasks: List[asyncio.Task] = []
        
        # Initialize default topics
        self._setup_default_topics()
        
        logger.info(f"Kafka Manager initialized for node: {node_id}")
    
    def _setup_default_topics(self):
        """Setup default MCP topic configurations"""
        self.topics = {
            TopicType.COMMANDS: TopicConfig(
                name=f"mcp-commands-{self.region}",
                partitions=6,  # Higher parallelism for commands
                retention_ms=3600000,  # 1 hour retention
                compression_type="lz4"
            ),
            TopicType.RESPONSES: TopicConfig(
                name=f"mcp-responses-{self.region}",
                partitions=6,
                retention_ms=3600000,  # 1 hour retention
                compression_type="lz4"
            ),
            TopicType.HEARTBEATS: TopicConfig(
                name=f"mcp-heartbeats-{self.region}",
                partitions=1,  # Single partition for ordering
                retention_ms=300000,  # 5 minutes retention
                cleanup_policy="compact"
            ),
            TopicType.DISCOVERY: TopicConfig(
                name="mcp-discovery-global",  # Global discovery
                partitions=1,
                retention_ms=1800000,  # 30 minutes retention
                cleanup_policy="compact"
            ),
            TopicType.BROADCASTS: TopicConfig(
                name=f"mcp-broadcasts-{self.region}",
                partitions=3,
                retention_ms=86400000,  # 24 hours retention
                compression_type="snappy"
            ),
            TopicType.COORDINATION: TopicConfig(
                name="mcp-coordination-global",  # Global coordination
                partitions=1,  # Single partition for ordering
                retention_ms=3600000,  # 1 hour retention
                cleanup_policy="compact"
            ),
            TopicType.LOGS: TopicConfig(
                name=f"mcp-logs-{self.region}",
                partitions=12,  # High parallelism for logs
                retention_ms=604800000,  # 7 days retention
                compression_type="gzip"
            ),
            TopicType.METRICS: TopicConfig(
                name=f"mcp-metrics-{self.region}",
                partitions=6,
                retention_ms=2592000000,  # 30 days retention
                compression_type="snappy"
            )
        }
    
    async def start(self):
        """Start Kafka manager and initialize connections"""
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available - cannot start Kafka manager")
            raise Exception("Kafka dependencies not available")
        
        logger.info("Starting Kafka Manager...")
        
        try:
            # Initialize admin client
            await self._init_admin_client()
            
            # Create topics
            await self._create_topics()
            
            # Initialize producer
            await self._init_producer()
            
            # Initialize metrics
            self._init_metrics()
            
            self.running = True
            logger.info("Kafka Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka Manager: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop Kafka manager and close all connections"""
        logger.info("Stopping Kafka Manager...")
        
        self.running = False
        
        # Stop consumer tasks
        for task in self.consumer_tasks:
            if not task.done():
                task.cancel()
        
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        # Close consumers
        for consumer in self.consumers.values():
            await consumer.stop()
        self.consumers.clear()
        
        # Close producer
        if self.producer:
            await self.producer.stop()
        
        # Close admin client
        if self.admin_client:
            await self.admin_client.close()
        
        logger.info("Kafka Manager stopped")
    
    async def _init_admin_client(self):
        """Initialize Kafka admin client for topic management"""
        self.admin_client = AIOKafkaAdminClient(**self.kafka_config)
        await self.admin_client.start()
        logger.info("Kafka admin client initialized")
    
    async def _init_producer(self):
        """Initialize Kafka producer for sending messages"""
        producer_config = {
            **self.kafka_config,
            'value_serializer': self._serialize_message,
            'compression_type': 'snappy',
            'batch_size': 16384,
            'linger_ms': 10,  # Small delay for batching
            'max_request_size': 1048576,  # 1MB
            'acks': 'all',  # Wait for all replicas in production
            'retries': 3,
            'enable_idempotence': True
        }
        
        self.producer = AIOKafkaProducer(**producer_config)
        await self.producer.start()
        logger.info("Kafka producer initialized")
    
    def _serialize_message(self, message: Dict[str, Any]) -> bytes:
        """Serialize message to bytes with compression"""
        try:
            # Add metadata
            message['_kafka_timestamp'] = time.time()
            message['_kafka_node_id'] = self.node_id
            
            # Serialize to JSON
            json_str = json.dumps(message, separators=(',', ':'))
            return json_str.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Message serialization failed: {e}")
            raise
    
    def _deserialize_message(self, data: bytes) -> Dict[str, Any]:
        """Deserialize message from bytes"""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Message deserialization failed: {e}")
            raise
    
    async def _create_topics(self):
        """Create Kafka topics if they don't exist"""
        logger.info("Creating Kafka topics...")
        
        # Get existing topics
        try:
            metadata = await self.admin_client.describe_cluster()
            existing_topics = set()
            
            for topic_config in self.topics.values():
                topic_name = topic_config.name
                
                if topic_name not in existing_topics:
                    try:
                        new_topic = NewTopic(
                            name=topic_name,
                            num_partitions=topic_config.partitions,
                            replication_factor=topic_config.replication_factor,
                            config={
                                'retention.ms': str(topic_config.retention_ms),
                                'compression.type': topic_config.compression_type,
                                'cleanup.policy': topic_config.cleanup_policy,
                                'max.message.bytes': str(topic_config.max_message_bytes)
                            }
                        )
                        
                        await self.admin_client.create_topics([new_topic])
                        logger.info(f"Created topic: {topic_name}")
                        
                    except TopicAlreadyExistsError:
                        logger.debug(f"Topic already exists: {topic_name}")
                    except Exception as e:
                        logger.error(f"Failed to create topic {topic_name}: {e}")
        
        except Exception as e:
            logger.error(f"Topic creation failed: {e}")
            # Continue without failing - topics might be created externally
    
    def _init_metrics(self):
        """Initialize metrics tracking for each topic"""
        for topic_type in self.topics.keys():
            self.metrics[topic_type.value] = MessageMetrics()
    
    async def send_message(self, topic_type: TopicType, message: Dict[str, Any], 
                          key: Optional[str] = None, priority: MessagePriority = MessagePriority.NORMAL,
                          headers: Optional[Dict[str, str]] = None) -> bool:
        """Send message to Kafka topic"""
        if not self.producer or not self.running:
            logger.error("Kafka producer not available")
            return False
        
        try:
            topic_config = self.topics.get(topic_type)
            if not topic_config:
                logger.error(f"Unknown topic type: {topic_type}")
                return False
            
            # Add message metadata
            enriched_message = {
                **message,
                '_priority': priority,
                '_timestamp': time.time(),
                '_source_node': self.node_id,
                '_topic_type': topic_type.value
            }
            
            # Prepare headers
            kafka_headers = []
            if headers:
                kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]
            
            kafka_headers.append(('priority', str(priority).encode('utf-8')))
            kafka_headers.append(('node_id', self.node_id.encode('utf-8')))
            
            # Send to Kafka
            partition = None
            if key:
                # Use consistent hashing for key-based partitioning
                partition = self._get_partition_for_key(key, topic_config.partitions)
            
            await self.producer.send(
                topic_config.name,
                value=enriched_message,
                key=key.encode('utf-8') if key else None,
                partition=partition,
                headers=kafka_headers
            )
            
            # Update metrics
            metrics = self.metrics[topic_type.value]
            metrics.total_sent += 1
            metrics.bytes_sent += len(json.dumps(enriched_message).encode('utf-8'))
            metrics.last_activity = time.time()
            
            logger.debug(f"Message sent to {topic_config.name}: {message.get('message_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {topic_type}: {e}")
            self.metrics[topic_type.value].errors += 1
            return False
    
    def _get_partition_for_key(self, key: str, num_partitions: int) -> int:
        """Get partition number for key using consistent hashing"""
        hash_value = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
        return hash_value % num_partitions
    
    async def start_consumer(self, topic_type: TopicType, group_id: str, 
                           handler: Callable[[Dict[str, Any]], None],
                           message_filter: Optional[Callable[[Dict[str, Any]], bool]] = None) -> str:
        """Start a consumer for a specific topic"""
        if not KAFKA_AVAILABLE or not self.running:
            logger.error("Cannot start consumer - Kafka not available or manager not running")
            return None
        
        topic_config = self.topics.get(topic_type)
        if not topic_config:
            logger.error(f"Unknown topic type: {topic_type}")
            return None
        
        consumer_id = f"{group_id}-{topic_type.value}-{uuid.uuid4().hex[:8]}"
        
        try:
            # Consumer configuration
            consumer_config = {
                **self.kafka_config,
                'group_id': group_id,
                'auto_offset_reset': 'latest',  # Start from latest messages
                'enable_auto_commit': True,
                'auto_commit_interval_ms': 5000,
                'max_poll_records': 100,
                'session_timeout_ms': 30000,
                'heartbeat_interval_ms': 10000,
                'value_deserializer': self._deserialize_message
            }
            
            consumer = AIOKafkaConsumer(
                topic_config.name,
                **consumer_config
            )
            
            await consumer.start()
            self.consumers[consumer_id] = consumer
            self.consumer_groups.add(group_id)
            
            # Start consumer task
            task = asyncio.create_task(
                self._consumer_loop(consumer_id, consumer, handler, message_filter, topic_type)
            )
            self.consumer_tasks.append(task)
            
            logger.info(f"Started consumer {consumer_id} for topic {topic_config.name}")
            return consumer_id
            
        except Exception as e:
            logger.error(f"Failed to start consumer for {topic_type}: {e}")
            return None
    
    async def _consumer_loop(self, consumer_id: str, consumer: AIOKafkaConsumer,
                           handler: Callable, message_filter: Optional[Callable],
                           topic_type: TopicType):
        """Consumer message processing loop"""
        logger.info(f"Starting consumer loop: {consumer_id}")
        
        try:
            async for message in consumer:
                if not self.running:
                    break
                
                try:
                    # Deserialize message
                    msg_data = message.value
                    
                    # Skip messages from self (if not needed)
                    if msg_data.get('_source_node') == self.node_id:
                        continue
                    
                    # Apply filter if provided
                    if message_filter and not message_filter(msg_data):
                        continue
                    
                    # Update metrics
                    metrics = self.metrics[topic_type.value]
                    metrics.total_received += 1
                    metrics.bytes_received += len(json.dumps(msg_data).encode('utf-8'))
                    metrics.last_activity = time.time()
                    
                    # Calculate latency
                    msg_timestamp = msg_data.get('_kafka_timestamp', time.time())
                    latency = (time.time() - msg_timestamp) * 1000  # ms
                    metrics.avg_latency_ms = (metrics.avg_latency_ms + latency) / 2
                    
                    # Call handler
                    await handler(msg_data)
                    
                except Exception as e:
                    logger.error(f"Consumer {consumer_id} message processing error: {e}")
                    self.metrics[topic_type.value].errors += 1
                    continue
        
        except Exception as e:
            logger.error(f"Consumer {consumer_id} loop error: {e}")
        finally:
            logger.info(f"Consumer loop ended: {consumer_id}")
    
    async def stop_consumer(self, consumer_id: str):
        """Stop a specific consumer"""
        if consumer_id in self.consumers:
            consumer = self.consumers.pop(consumer_id)
            await consumer.stop()
            logger.info(f"Stopped consumer: {consumer_id}")
    
    async def send_priority_message(self, topic_type: TopicType, message: Dict[str, Any],
                                  priority: MessagePriority = MessagePriority.HIGH) -> bool:
        """Send high-priority message with expedited processing"""
        # Add priority routing key
        key = f"priority-{priority}-{uuid.uuid4().hex[:8]}"
        
        return await self.send_message(
            topic_type=topic_type,
            message=message,
            key=key,
            priority=priority,
            headers={'priority': str(priority)}
        )
    
    async def broadcast_message(self, message: Dict[str, Any]) -> bool:
        """Broadcast message to all nodes in region"""
        return await self.send_message(
            topic_type=TopicType.BROADCASTS,
            message=message,
            priority=MessagePriority.HIGH
        )
    
    async def send_heartbeat(self, node_info: Dict[str, Any]) -> bool:
        """Send node heartbeat message"""
        return await self.send_message(
            topic_type=TopicType.HEARTBEATS,
            message={
                'node_id': self.node_id,
                'heartbeat_timestamp': time.time(),
                'node_info': node_info
            },
            key=self.node_id,  # Ensure ordering per node
            priority=MessagePriority.NORMAL
        )
    
    async def send_command(self, target_node: str, command: str, parameters: Dict[str, Any],
                          correlation_id: Optional[str] = None) -> bool:
        """Send command to specific target node"""
        return await self.send_message(
            topic_type=TopicType.COMMANDS,
            message={
                'command': command,
                'parameters': parameters,
                'target_node': target_node,
                'correlation_id': correlation_id or str(uuid.uuid4())
            },
            key=target_node,  # Route to target node's partition
            priority=MessagePriority.HIGH
        )
    
    async def send_response(self, original_message: Dict[str, Any], response_data: Dict[str, Any]) -> bool:
        """Send response to command"""
        correlation_id = original_message.get('correlation_id')
        source_node = original_message.get('_source_node')
        
        return await self.send_message(
            topic_type=TopicType.RESPONSES,
            message={
                'response_data': response_data,
                'correlation_id': correlation_id,
                'original_command': original_message.get('command')
            },
            key=source_node,  # Route back to original sender
            priority=MessagePriority.HIGH
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Kafka metrics"""
        total_metrics = MessageMetrics()
        
        # Aggregate metrics across all topics
        for metrics in self.metrics.values():
            total_metrics.total_sent += metrics.total_sent
            total_metrics.total_received += metrics.total_received
            total_metrics.bytes_sent += metrics.bytes_sent
            total_metrics.bytes_received += metrics.bytes_received
            total_metrics.errors += metrics.errors
        
        return {
            'node_id': self.node_id,
            'running': self.running,
            'kafka_available': KAFKA_AVAILABLE,
            'producer_connected': self.producer is not None,
            'active_consumers': len(self.consumers),
            'consumer_groups': list(self.consumer_groups),
            'topics_configured': len(self.topics),
            'total_metrics': asdict(total_metrics),
            'per_topic_metrics': {k: asdict(v) for k, v in self.metrics.items()},
            'kafka_config': {
                'bootstrap_servers': self.kafka_config['bootstrap_servers'],
                'region': self.region
            }
        }
    
    def get_topic_info(self) -> Dict[str, Any]:
        """Get information about configured topics"""
        return {
            topic_type.value: {
                'name': config.name,
                'partitions': config.partitions,
                'retention_ms': config.retention_ms,
                'compression': config.compression_type
            }
            for topic_type, config in self.topics.items()
        }
    
    async def health_check(self) -> bool:
        """Check Kafka connectivity and health"""
        if not KAFKA_AVAILABLE or not self.producer:
            return False
        
        try:
            # Try to send a test message
            test_msg = {
                'test': True,
                'timestamp': time.time(),
                'node_id': self.node_id
            }
            
            await self.send_message(TopicType.HEARTBEATS, test_msg)
            return True
            
        except Exception as e:
            logger.error(f"Kafka health check failed: {e}")
            return False


# Module-level functions
async def create_kafka_manager(node_id: str, region: str = "us-west-1") -> KafkaManager:
    """Create and start a Kafka manager instance"""
    manager = KafkaManager(node_id, region)
    await manager.start()
    return manager
