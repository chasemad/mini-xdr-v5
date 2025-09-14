"""
Distributed MCP Architecture - Phase 3
Advanced distributed coordination, messaging, and state management for Mini-XDR

This module provides enterprise-grade distributed capabilities:
- Apache Kafka for high-performance messaging
- Redis Cluster for distributed state management
- Advanced load balancing and coordination
- Cross-region replication and synchronization
- Distributed locks and consensus protocols
"""

from .mcp_coordinator import (
    DistributedMCPCoordinator,
    MessageType,
    NodeRole,
    LoadBalanceStrategy,
    MCPMessage,
    MCPNode,
    LoadBalancer,
    get_distributed_coordinator,
    initialize_distributed_mcp
)

from .kafka_manager import (
    KafkaManager,
    TopicType,
    MessagePriority,
    TopicConfig,
    MessageMetrics,
    create_kafka_manager
)

from .redis_cluster import (
    RedisClusterManager,
    RedisConfig,
    CacheStrategy,
    LockType,
    CacheMetrics,
    DistributedLock,
    create_redis_cluster_manager
)

import logging
import asyncio
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global instances (initialized on first use)
_distributed_coordinator: Optional[DistributedMCPCoordinator] = None
_kafka_manager: Optional[KafkaManager] = None
_redis_manager: Optional[RedisClusterManager] = None

# Version and capability information
__version__ = "3.0.0"
__phase__ = "Phase 3: Distributed MCP Architecture"

DISTRIBUTED_CAPABILITIES = [
    "kafka_messaging",
    "redis_clustering", 
    "distributed_locks",
    "load_balancing",
    "cross_region_sync",
    "consensus_protocols",
    "distributed_state",
    "message_routing",
    "health_monitoring",
    "auto_scaling"
]


async def initialize_distributed_system(
    node_id: Optional[str] = None,
    role: NodeRole = NodeRole.COORDINATOR,
    region: str = "us-west-1",
    kafka_enabled: bool = True,
    redis_enabled: bool = True
) -> Dict[str, Any]:
    """
    Initialize the complete distributed MCP system
    
    Args:
        node_id: Unique identifier for this node (auto-generated if None)
        role: Role of this node in the distributed system
        region: Geographic region for optimization
        kafka_enabled: Enable Kafka messaging
        redis_enabled: Enable Redis state management
        
    Returns:
        Dict with initialization status and component information
    """
    global _distributed_coordinator, _kafka_manager, _redis_manager
    
    logger.info(f"Initializing Distributed MCP System - {__phase__}")
    
    results = {
        'success': False,
        'components': {},
        'errors': []
    }
    
    try:
        # Initialize Redis if enabled
        if redis_enabled:
            try:
                logger.info("Initializing Redis Cluster Manager...")
                _redis_manager = await create_redis_cluster_manager(
                    node_id=node_id or "auto-generated",
                )
                results['components']['redis'] = {
                    'status': 'initialized',
                    'connected': await _redis_manager.health_check()
                }
                logger.info("✅ Redis Cluster Manager initialized")
            except Exception as e:
                error_msg = f"Redis initialization failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                results['components']['redis'] = {'status': 'failed', 'error': str(e)}
        
        # Initialize Kafka if enabled
        if kafka_enabled:
            try:
                logger.info("Initializing Kafka Manager...")
                _kafka_manager = await create_kafka_manager(
                    node_id=node_id or "auto-generated",
                    region=region
                )
                results['components']['kafka'] = {
                    'status': 'initialized',
                    'connected': await _kafka_manager.health_check()
                }
                logger.info("✅ Kafka Manager initialized")
            except Exception as e:
                error_msg = f"Kafka initialization failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                results['components']['kafka'] = {'status': 'failed', 'error': str(e)}
        
        # Initialize Distributed Coordinator
        try:
            logger.info("Initializing Distributed MCP Coordinator...")
            _distributed_coordinator = await initialize_distributed_mcp(node_role=role)
            results['components']['coordinator'] = {
                'status': 'initialized',
                'node_id': _distributed_coordinator.node_id,
                'role': role,
                'region': region
            }
            logger.info("✅ Distributed MCP Coordinator initialized")
        except Exception as e:
            error_msg = f"Coordinator initialization failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            results['components']['coordinator'] = {'status': 'failed', 'error': str(e)}
        
        # Check overall success
        results['success'] = len(results['errors']) == 0
        
        if results['success']:
            logger.info("🚀 Distributed MCP System initialized successfully!")
        else:
            logger.warning(f"⚠️ Distributed MCP System initialized with {len(results['errors'])} errors")
        
        return results
        
    except Exception as e:
        error_msg = f"System initialization failed: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
        return results


async def shutdown_distributed_system():
    """Shutdown all distributed system components"""
    global _distributed_coordinator, _kafka_manager, _redis_manager
    
    logger.info("Shutting down Distributed MCP System...")
    
    # Shutdown coordinator
    if _distributed_coordinator:
        try:
            await _distributed_coordinator.stop()
            logger.info("✅ Distributed MCP Coordinator stopped")
        except Exception as e:
            logger.error(f"Error stopping coordinator: {e}")
        finally:
            _distributed_coordinator = None
    
    # Shutdown Kafka
    if _kafka_manager:
        try:
            await _kafka_manager.stop()
            logger.info("✅ Kafka Manager stopped")
        except Exception as e:
            logger.error(f"Error stopping Kafka manager: {e}")
        finally:
            _kafka_manager = None
    
    # Shutdown Redis
    if _redis_manager:
        try:
            await _redis_manager.stop()
            logger.info("✅ Redis Cluster Manager stopped")
        except Exception as e:
            logger.error(f"Error stopping Redis manager: {e}")
        finally:
            _redis_manager = None
    
    logger.info("🔄 Distributed MCP System shutdown complete")


def get_system_status() -> Dict[str, Any]:
    """Get current status of all distributed system components"""
    status = {
        'version': __version__,
        'phase': __phase__,
        'capabilities': DISTRIBUTED_CAPABILITIES,
        'components': {}
    }
    
    # Coordinator status
    if _distributed_coordinator:
        status['components']['coordinator'] = _distributed_coordinator.get_coordinator_status()
    else:
        status['components']['coordinator'] = {'status': 'not_initialized'}
    
    # Kafka status
    if _kafka_manager:
        status['components']['kafka'] = _kafka_manager.get_metrics()
    else:
        status['components']['kafka'] = {'status': 'not_initialized'}
    
    # Redis status
    if _redis_manager:
        status['components']['redis'] = _redis_manager.get_metrics()
    else:
        status['components']['redis'] = {'status': 'not_initialized'}
    
    return status


async def health_check() -> Dict[str, Any]:
    """Comprehensive health check of distributed system"""
    health_status = {
        'overall_healthy': True,
        'components': {},
        'timestamp': asyncio.get_event_loop().time()
    }
    
    # Check coordinator
    if _distributed_coordinator:
        coordinator_status = _distributed_coordinator.get_coordinator_status()
        health_status['components']['coordinator'] = {
            'healthy': coordinator_status.get('running', False),
            'details': coordinator_status
        }
    else:
        health_status['components']['coordinator'] = {
            'healthy': False,
            'details': {'error': 'Not initialized'}
        }
    
    # Check Kafka
    if _kafka_manager:
        kafka_healthy = await _kafka_manager.health_check()
        health_status['components']['kafka'] = {
            'healthy': kafka_healthy,
            'details': _kafka_manager.get_metrics()
        }
    else:
        health_status['components']['kafka'] = {
            'healthy': False,
            'details': {'error': 'Not initialized'}
        }
    
    # Check Redis
    if _redis_manager:
        redis_healthy = await _redis_manager.health_check()
        health_status['components']['redis'] = {
            'healthy': redis_healthy,
            'details': _redis_manager.get_metrics()
        }
    else:
        health_status['components']['redis'] = {
            'healthy': False,
            'details': {'error': 'Not initialized'}
        }
    
    # Determine overall health
    health_status['overall_healthy'] = all(
        comp.get('healthy', False) 
        for comp in health_status['components'].values()
    )
    
    return health_status


def get_distributed_coordinator_instance() -> Optional[DistributedMCPCoordinator]:
    """Get the global distributed coordinator instance"""
    return _distributed_coordinator


def get_kafka_manager_instance() -> Optional[KafkaManager]:
    """Get the global Kafka manager instance"""
    return _kafka_manager


def get_redis_manager_instance() -> Optional[RedisClusterManager]:
    """Get the global Redis manager instance"""
    return _redis_manager


# Convenience functions for common operations
async def broadcast_system_message(message_type: str, payload: Dict[str, Any]) -> bool:
    """Broadcast message to all nodes in the distributed system"""
    if _distributed_coordinator:
        try:
            await _distributed_coordinator.broadcast_message(message_type, payload)
            return True
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            return False
    return False


async def execute_distributed_tool(tool_name: str, parameters: Dict[str, Any], 
                                 strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_LOADED) -> Optional[Dict[str, Any]]:
    """Execute a tool across the distributed network"""
    if _distributed_coordinator:
        try:
            return await _distributed_coordinator.execute_distributed_tool(
                tool_name, parameters, strategy
            )
        except Exception as e:
            logger.error(f"Distributed tool execution failed: {e}")
            return None
    return None


async def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set value in distributed cache"""
    if _redis_manager:
        return await _redis_manager.cache_set(key, value, ttl)
    return False


async def cache_get(key: str) -> Optional[Any]:
    """Get value from distributed cache"""
    if _redis_manager:
        return await _redis_manager.cache_get(key)
    return None


async def acquire_distributed_lock(key: str, timeout: int = 10):
    """Acquire a distributed lock (context manager)"""
    if _redis_manager:
        return _redis_manager.acquire_lock(key, timeout)
    else:
        # Fallback to local lock
        return asyncio.Lock()


# Export main classes and functions
__all__ = [
    # Core classes
    'DistributedMCPCoordinator',
    'KafkaManager', 
    'RedisClusterManager',
    
    # Enums and data classes
    'MessageType',
    'NodeRole',
    'LoadBalanceStrategy',
    'TopicType',
    'MessagePriority',
    'CacheStrategy',
    'LockType',
    'MCPMessage',
    'MCPNode',
    'TopicConfig',
    'RedisConfig',
    'MessageMetrics',
    'CacheMetrics',
    'DistributedLock',
    
    # Main functions
    'initialize_distributed_system',
    'shutdown_distributed_system',
    'get_system_status',
    'health_check',
    
    # Instance getters
    'get_distributed_coordinator_instance',
    'get_kafka_manager_instance', 
    'get_redis_manager_instance',
    
    # Convenience functions
    'broadcast_system_message',
    'execute_distributed_tool',
    'cache_set',
    'cache_get',
    'acquire_distributed_lock',
    
    # Factory functions
    'create_kafka_manager',
    'create_redis_cluster_manager',
    'initialize_distributed_mcp',
    'get_distributed_coordinator',
    
    # Module info
    '__version__',
    '__phase__',
    'DISTRIBUTED_CAPABILITIES'
]

logger.info(f"🚀 {__phase__} module loaded with {len(DISTRIBUTED_CAPABILITIES)} capabilities")
