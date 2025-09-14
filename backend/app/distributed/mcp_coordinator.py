"""
Distributed MCP Coordinator
Manages distributed Model Context Protocol operations across multiple Mini-XDR instances
with Apache Kafka messaging and Redis state management.
"""

import asyncio
import logging
import json
import uuid
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Distributed messaging
try:
    import aiokafka
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    aiokafka = None
    logging.warning("AioKafka not available - distributed messaging disabled")

# Distributed state management
import redis
from redis.asyncio import Redis as AsyncRedis

# Authentication and encryption
from jose import jwt, JWTError
import secrets
import hashlib

# Configuration
from ..config import settings

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of distributed MCP messages"""
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    LOAD_BALANCE = "load_balance"
    COORDINATION = "coordination"


class NodeRole(str, Enum):
    """Distributed MCP node roles"""
    COORDINATOR = "coordinator"  # Primary coordination node
    PARTICIPANT = "participant"  # Standard processing node
    OBSERVER = "observer"       # Read-only monitoring node
    BACKUP = "backup"          # Backup coordinator


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategies for distributed operations"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    GEOGRAPHIC = "geographic"
    CAPABILITY_BASED = "capability_based"
    CONSISTENT_HASHING = "consistent_hashing"


@dataclass
class MCPMessage:
    """Distributed MCP message structure"""
    message_id: str
    message_type: MessageType
    source_node: str
    target_node: Optional[str] = None  # None for broadcasts
    payload: Dict[str, Any] = None
    timestamp: float = None
    expires_at: Optional[float] = None
    priority: int = 5  # 1-10, higher = more priority
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.payload is None:
            self.payload = {}


@dataclass 
class MCPNode:
    """Distributed MCP node information"""
    node_id: str
    role: NodeRole
    host: str
    port: int
    capabilities: List[str]
    load_metrics: Dict[str, float]
    last_heartbeat: float
    region: Optional[str] = None
    version: str = "1.0.0"
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy based on heartbeat"""
        return (time.time() - self.last_heartbeat) < 30  # 30 second timeout


class DistributedMCPCoordinator:
    """
    Distributed MCP Coordinator
    Orchestrates MCP operations across multiple Mini-XDR instances
    """
    
    def __init__(self, node_id: str = None, role: NodeRole = NodeRole.COORDINATOR):
        self.node_id = node_id or self._generate_node_id()
        self.role = role
        self.region = getattr(settings, 'mcp_region', 'us-west-1')
        
        # Kafka configuration
        self.kafka_config = {
            'bootstrap_servers': getattr(settings, 'kafka_servers', 'localhost:9092'),
            'security_protocol': 'PLAINTEXT',  # Use SSL in production
            'client_id': f'mcp-{self.node_id}'
        }
        
        # Redis configuration
        self.redis_config = {
            'host': getattr(settings, 'redis_host', 'localhost'),
            'port': getattr(settings, 'redis_port', 6379),
            'db': getattr(settings, 'redis_db', 0),
            'decode_responses': True
        }
        
        # Initialize components
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None
        self.redis_client: Optional[AsyncRedis] = None
        
        # Node registry and load balancing
        self.known_nodes: Dict[str, MCPNode] = {}
        self.load_balancer = LoadBalancer()
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Coordinator state
        self.running = False
        self.coordinator_tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'tools_executed': 0,
            'load_balance_decisions': 0,
            'coordination_events': 0,
            'last_heartbeat_sent': 0
        }
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        logger.info(f"Distributed MCP Coordinator initialized: {self.node_id} ({role})")
    
    def _generate_node_id(self) -> str:
        """Generate unique node identifier"""
        hostname = getattr(settings, 'hostname', 'unknown')
        return f"mcp-{hostname}-{uuid.uuid4().hex[:8]}"
    
    def _setup_message_handlers(self):
        """Setup message type handlers"""
        self.message_handlers = {
            MessageType.TOOL_REQUEST: self._handle_tool_request,
            MessageType.TOOL_RESPONSE: self._handle_tool_response,
            MessageType.BROADCAST: self._handle_broadcast,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.DISCOVERY: self._handle_discovery,
            MessageType.LOAD_BALANCE: self._handle_load_balance,
            MessageType.COORDINATION: self._handle_coordination
        }
    
    async def start(self):
        """Start the distributed MCP coordinator"""
        logger.info(f"Starting Distributed MCP Coordinator: {self.node_id}")
        
        try:
            # Initialize Redis connection
            await self._init_redis()
            
            # Initialize Kafka if available
            if KAFKA_AVAILABLE:
                await self._init_kafka()
            else:
                logger.warning("Kafka not available - using Redis-only messaging")
            
            # Register this node
            await self._register_node()
            
            # Start coordinator tasks
            self.running = True
            await self._start_coordinator_tasks()
            
            logger.info(f"Distributed MCP Coordinator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP coordinator: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the distributed MCP coordinator"""
        logger.info(f"Stopping Distributed MCP Coordinator: {self.node_id}")
        
        self.running = False
        
        # Cancel coordinator tasks
        for task in self.coordinator_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.coordinator_tasks:
            await asyncio.gather(*self.coordinator_tasks, return_exceptions=True)
        
        # Close Kafka connections
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.aclose()
        
        logger.info("Distributed MCP Coordinator stopped")
    
    async def _init_redis(self):
        """Initialize Redis connection for distributed state"""
        try:
            self.redis_client = AsyncRedis(**self.redis_config)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established for distributed state")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _init_kafka(self):
        """Initialize Kafka producer and consumer"""
        try:
            # Initialize producer for sending messages
            self.kafka_producer = AIOKafkaProducer(
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                **self.kafka_config
            )
            await self.kafka_producer.start()
            
            # Initialize consumer for receiving messages
            self.kafka_consumer = AIOKafkaConsumer(
                f'mcp-{self.region}',  # Topic per region
                f'mcp-global',        # Global coordination topic
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id=f'mcp-group-{self.node_id}',
                **self.kafka_config
            )
            await self.kafka_consumer.start()
            
            logger.info("Kafka messaging initialized for distributed MCP")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            KAFKA_AVAILABLE = False
            raise
    
    async def _register_node(self):
        """Register this node in the distributed registry"""
        node_info = MCPNode(
            node_id=self.node_id,
            role=self.role,
            host=getattr(settings, 'mcp_host', 'localhost'),
            port=getattr(settings, 'mcp_port', 3001),
            capabilities=self._get_node_capabilities(),
            load_metrics=await self._get_load_metrics(),
            last_heartbeat=time.time(),
            region=self.region
        )
        
        # Store in Redis
        node_key = f"mcp:nodes:{self.node_id}"
        await self.redis_client.setex(
            node_key, 
            60,  # 60 second TTL
            json.dumps(asdict(node_info))
        )
        
        # Add to local registry
        self.known_nodes[self.node_id] = node_info
        
        # Announce discovery
        await self._send_discovery_message()
        
        logger.info(f"Node registered: {self.node_id} ({self.role})")
    
    def _get_node_capabilities(self) -> List[str]:
        """Get capabilities of this node"""
        return [
            "threat_detection",
            "incident_management", 
            "forensics_analysis",
            "containment_actions",
            "ml_analysis",
            "federated_learning",
            "explainable_ai"
        ]
    
    async def _get_load_metrics(self) -> Dict[str, float]:
        """Get current load metrics for this node"""
        # In a real implementation, get actual system metrics
        return {
            'cpu_usage': 0.25,
            'memory_usage': 0.40,
            'active_connections': 10,
            'queue_depth': 0,
            'response_time_avg': 150.0  # ms
        }
    
    async def _start_coordinator_tasks(self):
        """Start background coordinator tasks"""
        self.coordinator_tasks = [
            asyncio.create_task(self._heartbeat_task()),
            asyncio.create_task(self._node_discovery_task()),
            asyncio.create_task(self._load_balancing_task()),
            asyncio.create_task(self._message_processing_task()),
            asyncio.create_task(self._cleanup_task())
        ]
    
    async def _heartbeat_task(self):
        """Send periodic heartbeats to maintain node presence"""
        while self.running:
            try:
                # Update node info
                await self._register_node()
                
                # Send heartbeat message
                heartbeat_msg = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    source_node=self.node_id,
                    payload={
                        'load_metrics': await self._get_load_metrics(),
                        'capabilities': self._get_node_capabilities()
                    }
                )
                
                await self._send_message(heartbeat_msg, topic='mcp-heartbeats')
                self.metrics['last_heartbeat_sent'] = time.time()
                
                await asyncio.sleep(15)  # Send heartbeat every 15 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat task error: {e}")
                await asyncio.sleep(5)
    
    async def _node_discovery_task(self):
        """Discover and monitor other nodes"""
        while self.running:
            try:
                # Get all nodes from Redis
                node_keys = await self.redis_client.keys("mcp:nodes:*")
                
                for node_key in node_keys:
                    node_data = await self.redis_client.get(node_key)
                    if node_data:
                        node_info = MCPNode(**json.loads(node_data))
                        if node_info.node_id != self.node_id:
                            self.known_nodes[node_info.node_id] = node_info
                
                # Clean up dead nodes
                dead_nodes = [
                    node_id for node_id, node in self.known_nodes.items()
                    if not node.is_healthy
                ]
                
                for node_id in dead_nodes:
                    del self.known_nodes[node_id]
                    logger.info(f"Removed dead node: {node_id}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Node discovery task error: {e}")
                await asyncio.sleep(10)
    
    async def _load_balancing_task(self):
        """Periodic load balancing optimization"""
        while self.running:
            try:
                if len(self.known_nodes) > 1:
                    await self.load_balancer.optimize_distribution(
                        list(self.known_nodes.values())
                    )
                    self.metrics['load_balance_decisions'] += 1
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                logger.error(f"Load balancing task error: {e}")
                await asyncio.sleep(30)
    
    async def _message_processing_task(self):
        """Process incoming Kafka messages"""
        if not KAFKA_AVAILABLE or not self.kafka_consumer:
            return
            
        while self.running:
            try:
                async for message in self.kafka_consumer:
                    mcp_message = MCPMessage(**message.value)
                    
                    # Skip messages from self
                    if mcp_message.source_node == self.node_id:
                        continue
                    
                    # Handle message
                    await self._process_message(mcp_message)
                    self.metrics['messages_received'] += 1
                    
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_task(self):
        """Periodic cleanup of expired data"""
        while self.running:
            try:
                # Clean up expired messages, sessions, etc.
                current_time = time.time()
                
                # Clean up Redis expired keys (automatic with TTL)
                # Clean up local caches, etc.
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(60)
    
    async def _send_message(self, message: MCPMessage, topic: str = None):
        """Send message via Kafka or Redis fallback"""
        try:
            if KAFKA_AVAILABLE and self.kafka_producer:
                topic = topic or f'mcp-{self.region}'
                await self.kafka_producer.send(topic, asdict(message))
                self.metrics['messages_sent'] += 1
            else:
                # Fallback to Redis pub/sub
                channel = f"mcp:messages:{topic or 'default'}"
                await self.redis_client.publish(channel, json.dumps(asdict(message)))
                self.metrics['messages_sent'] += 1
                
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
    
    async def _send_discovery_message(self):
        """Send node discovery message"""
        discovery_msg = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DISCOVERY,
            source_node=self.node_id,
            payload={
                'role': self.role,
                'capabilities': self._get_node_capabilities(),
                'region': self.region
            }
        )
        
        await self._send_message(discovery_msg, topic='mcp-discovery')
    
    async def _process_message(self, message: MCPMessage):
        """Process incoming distributed MCP message"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Message handler error for {message.message_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
    
    # Message Handlers
    async def _handle_tool_request(self, message: MCPMessage):
        """Handle distributed tool execution request"""
        logger.info(f"Handling tool request: {message.message_id}")
        # Implementation for executing tools across distributed nodes
        pass
    
    async def _handle_tool_response(self, message: MCPMessage):
        """Handle tool execution response"""
        logger.info(f"Handling tool response: {message.message_id}")
        # Implementation for processing tool execution results
        pass
    
    async def _handle_broadcast(self, message: MCPMessage):
        """Handle broadcast message"""
        logger.info(f"Handling broadcast: {message.message_id}")
        # Implementation for broadcast message processing
        pass
    
    async def _handle_heartbeat(self, message: MCPMessage):
        """Handle node heartbeat"""
        # Update node information
        source_node = message.source_node
        if source_node in self.known_nodes:
            self.known_nodes[source_node].last_heartbeat = message.timestamp
            if 'load_metrics' in message.payload:
                self.known_nodes[source_node].load_metrics = message.payload['load_metrics']
    
    async def _handle_discovery(self, message: MCPMessage):
        """Handle node discovery message"""
        logger.info(f"Node discovery: {message.source_node}")
        # Respond with our node information
        await self._send_discovery_message()
    
    async def _handle_load_balance(self, message: MCPMessage):
        """Handle load balancing coordination"""
        logger.info(f"Load balancing event: {message.message_id}")
        # Implementation for load balancing coordination
        pass
    
    async def _handle_coordination(self, message: MCPMessage):
        """Handle coordination message"""
        logger.info(f"Coordination event: {message.message_id}")
        self.metrics['coordination_events'] += 1
        # Implementation for distributed coordination
        pass
    
    # Public API methods
    async def execute_distributed_tool(self, tool_name: str, parameters: Dict[str, Any], 
                                     strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED) -> Dict[str, Any]:
        """Execute a tool across the distributed MCP network"""
        
        # Select best node for execution
        target_node = await self.load_balancer.select_node(
            list(self.known_nodes.values()), 
            tool_name, 
            strategy
        )
        
        if not target_node:
            raise Exception("No suitable nodes available for tool execution")
        
        # Create tool request message
        request_msg = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TOOL_REQUEST,
            source_node=self.node_id,
            target_node=target_node.node_id,
            payload={
                'tool_name': tool_name,
                'parameters': parameters
            },
            correlation_id=str(uuid.uuid4())
        )
        
        # Send request
        await self._send_message(request_msg)
        
        # Wait for response (simplified - in production use proper async response handling)
        # This is a placeholder for the response handling logic
        return {"status": "executed", "target_node": target_node.node_id}
    
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """Broadcast message to all nodes"""
        broadcast_msg = MCPMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            source_node=self.node_id,
            payload={
                'broadcast_type': message_type,
                'data': payload
            }
        )
        
        await self._send_message(broadcast_msg, topic='mcp-broadcasts')
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get current coordinator status"""
        return {
            'node_id': self.node_id,
            'role': self.role,
            'running': self.running,
            'known_nodes': len(self.known_nodes),
            'healthy_nodes': len([n for n in self.known_nodes.values() if n.is_healthy]),
            'region': self.region,
            'metrics': self.metrics.copy(),
            'kafka_available': KAFKA_AVAILABLE,
            'redis_connected': self.redis_client is not None
        }


class LoadBalancer:
    """Load balancing for distributed MCP operations"""
    
    def __init__(self):
        self.round_robin_index = 0
    
    async def select_node(self, nodes: List[MCPNode], tool_name: str, 
                         strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED) -> Optional[MCPNode]:
        """Select best node for tool execution based on strategy"""
        
        # Filter healthy nodes
        healthy_nodes = [n for n in nodes if n.is_healthy]
        if not healthy_nodes:
            return None
        
        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            selected = healthy_nodes[self.round_robin_index % len(healthy_nodes)]
            self.round_robin_index += 1
            return selected
            
        elif strategy == LoadBalanceStrategy.LEAST_LOADED:
            return min(healthy_nodes, key=lambda n: n.load_metrics.get('cpu_usage', 1.0))
            
        elif strategy == LoadBalanceStrategy.CAPABILITY_BASED:
            # Select nodes that have the required capability for the tool
            capable_nodes = [n for n in healthy_nodes if self._has_tool_capability(n, tool_name)]
            if capable_nodes:
                return min(capable_nodes, key=lambda n: n.load_metrics.get('cpu_usage', 1.0))
            return healthy_nodes[0]  # Fallback
            
        else:
            # Default to first healthy node
            return healthy_nodes[0]
    
    def _has_tool_capability(self, node: MCPNode, tool_name: str) -> bool:
        """Check if node has capability for specific tool"""
        # Mapping of tools to capabilities
        tool_capabilities = {
            'get_incidents': 'incident_management',
            'contain_incident': 'containment_actions', 
            'analyze_forensics': 'forensics_analysis',
            'detect_threats': 'threat_detection',
            'explain_prediction': 'explainable_ai'
        }
        
        required_capability = tool_capabilities.get(tool_name)
        return required_capability in node.capabilities if required_capability else True
    
    async def optimize_distribution(self, nodes: List[MCPNode]):
        """Optimize load distribution across nodes"""
        # Advanced load balancing optimization
        # This is a placeholder for more sophisticated algorithms
        pass


# Global coordinator instance (will be initialized by main app)
distributed_coordinator: Optional[DistributedMCPCoordinator] = None


async def get_distributed_coordinator() -> DistributedMCPCoordinator:
    """Get or create distributed coordinator instance"""
    global distributed_coordinator
    
    if distributed_coordinator is None:
        distributed_coordinator = DistributedMCPCoordinator()
        await distributed_coordinator.start()
    
    return distributed_coordinator


async def initialize_distributed_mcp(node_role: NodeRole = NodeRole.COORDINATOR) -> DistributedMCPCoordinator:
    """Initialize distributed MCP system"""
    global distributed_coordinator
    
    if distributed_coordinator is not None:
        await distributed_coordinator.stop()
    
    distributed_coordinator = DistributedMCPCoordinator(role=node_role)
    await distributed_coordinator.start()
    
    return distributed_coordinator
