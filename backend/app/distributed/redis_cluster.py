"""
Redis Cluster Manager for Distributed MCP State
Handles distributed state management, caching, session synchronization,
and distributed locks for the Mini-XDR distributed architecture.
"""

import asyncio
import logging
import json
import hashlib
import time
import pickle
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from contextlib import asynccontextmanager

# Redis imports
import redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio import ConnectionPool
from redis.exceptions import RedisError, LockError

from ..config import settings

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache eviction strategies"""
    LRU = "allkeys-lru"           # Least Recently Used
    LFU = "allkeys-lfu"           # Least Frequently Used
    RANDOM = "allkeys-random"      # Random eviction
    TTL = "volatile-ttl"          # TTL-based eviction
    NO_EVICTION = "noeviction"    # No automatic eviction


class LockType(str, Enum):
    """Types of distributed locks"""
    EXCLUSIVE = "exclusive"       # Only one holder at a time
    SHARED = "shared"            # Multiple readers, single writer
    SEMAPHORE = "semaphore"      # Limited number of holders


@dataclass
class RedisConfig:
    """Redis cluster configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 50
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = None
    health_check_interval: int = 30
    
    def __post_init__(self):
        if self.socket_keepalive_options is None:
            self.socket_keepalive_options = {
                'TCP_KEEPIDLE': 1,
                'TCP_KEEPINTVL': 3,
                'TCP_KEEPCNT': 5
            }


@dataclass
class CacheMetrics:
    """Redis cache performance metrics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    expires: int = 0
    memory_usage: int = 0
    connections_active: int = 0
    operations_per_sec: float = 0.0
    avg_response_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0


class DistributedLock:
    """Distributed lock implementation using Redis"""
    
    def __init__(self, redis_client: AsyncRedis, key: str, timeout: int = 10, 
                 blocking_timeout: Optional[int] = None, lock_type: LockType = LockType.EXCLUSIVE):
        self.redis_client = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.blocking_timeout = blocking_timeout
        self.lock_type = lock_type
        self.identifier = str(uuid.uuid4())
        self.acquired = False
    
    async def acquire(self) -> bool:
        """Acquire the distributed lock"""
        try:
            if self.lock_type == LockType.EXCLUSIVE:
                result = await self.redis_client.set(
                    self.key, 
                    self.identifier, 
                    nx=True, 
                    ex=self.timeout
                )
                self.acquired = bool(result)
                return self.acquired
            
            elif self.lock_type == LockType.SEMAPHORE:
                # Implement semaphore-based locking
                return await self._acquire_semaphore()
            
            elif self.lock_type == LockType.SHARED:
                # Implement shared (reader-writer) locking
                return await self._acquire_shared()
            
        except RedisError as e:
            logger.error(f"Failed to acquire lock {self.key}: {e}")
            return False
    
    async def release(self) -> bool:
        """Release the distributed lock"""
        if not self.acquired:
            return False
        
        try:
            # Use Lua script for atomic release
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(lua_script, 1, self.key, self.identifier)
            self.acquired = False
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Failed to release lock {self.key}: {e}")
            return False
    
    async def _acquire_semaphore(self) -> bool:
        """Acquire semaphore-type lock with limited holders"""
        # Simplified semaphore implementation
        max_holders = getattr(self, 'max_holders', 5)
        current_holders = await self.redis_client.scard(self.key)
        
        if current_holders < max_holders:
            result = await self.redis_client.sadd(self.key, self.identifier)
            if result:
                await self.redis_client.expire(self.key, self.timeout)
                self.acquired = True
                return True
        
        return False
    
    async def _acquire_shared(self) -> bool:
        """Acquire shared (reader-writer) lock"""
        # Simplified shared lock implementation
        readers_key = f"{self.key}:readers"
        writers_key = f"{self.key}:writers"
        
        # Check if there are active writers
        writers_count = await self.redis_client.scard(writers_key)
        if writers_count > 0:
            return False
        
        # Add to readers
        result = await self.redis_client.sadd(readers_key, self.identifier)
        if result:
            await self.redis_client.expire(readers_key, self.timeout)
            self.acquired = True
            return True
        
        return False
    
    async def __aenter__(self):
        """Async context manager entry"""
        if await self.acquire():
            return self
        else:
            raise LockError(f"Failed to acquire lock: {self.key}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.release()


class RedisClusterManager:
    """
    Redis Cluster Manager for Distributed MCP State
    Provides high-performance caching, session management, and distributed coordination
    """
    
    def __init__(self, node_id: str, config: Optional[RedisConfig] = None):
        self.node_id = node_id
        self.config = config or self._load_config()
        
        # Redis connections
        self.redis_client: Optional[AsyncRedis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        
        # State management
        self.running = False
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = CacheMetrics()
        self.start_time = time.time()
        
        # Key prefixes for different data types
        self.key_prefixes = {
            'session': 'mcp:session:',
            'node': 'mcp:node:',
            'lock': 'mcp:lock:',
            'cache': 'mcp:cache:',
            'state': 'mcp:state:',
            'metrics': 'mcp:metrics:',
            'config': 'mcp:config:',
            'coordination': 'mcp:coord:'
        }
        
        logger.info(f"Redis Cluster Manager initialized for node: {node_id}")
    
    def _load_config(self) -> RedisConfig:
        """Load Redis configuration from settings"""
        return RedisConfig(
            host=getattr(settings, 'redis_host', 'localhost'),
            port=getattr(settings, 'redis_port', 6379),
            db=getattr(settings, 'redis_db', 0),
            password=getattr(settings, 'redis_password', None),
            ssl=getattr(settings, 'redis_ssl', False),
            max_connections=getattr(settings, 'redis_max_connections', 50)
        )
    
    async def start(self):
        """Start Redis cluster manager"""
        logger.info("Starting Redis Cluster Manager...")
        
        try:
            # Create connection pool
            pool_kwargs = {
                'host': self.config.host,
                'port': self.config.port,
                'db': self.config.db,
                'max_connections': self.config.max_connections,
                'retry_on_timeout': self.config.retry_on_timeout,
                'socket_keepalive': self.config.socket_keepalive,
                # Remove socket_keepalive_options for compatibility
                'decode_responses': True
            }

            # Add optional parameters if they exist
            if self.config.password:
                pool_kwargs['password'] = self.config.password

            # SSL is handled differently in newer Redis versions
            # For now, we'll skip SSL for local development
            if self.config.ssl:
                logger.info("SSL configuration requested but not implemented for local Redis")

            self.connection_pool = ConnectionPool(**pool_kwargs)
            
            # Create Redis client
            self.redis_client = AsyncRedis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            # Start health check task
            self.running = True
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Initialize node state
            await self._register_node()
            
            logger.info("Redis Cluster Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Redis Cluster Manager: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop Redis cluster manager"""
        logger.info("Stopping Redis Cluster Manager...")
        
        self.running = False
        
        # Cancel health check task
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup node state
        if self.redis_client:
            await self._unregister_node()
        
        # Close Redis connections
        if self.redis_client:
            await self.redis_client.aclose()
        
        if self.connection_pool:
            await self.connection_pool.aclose()
        
        logger.info("Redis Cluster Manager stopped")
    
    async def _register_node(self):
        """Register this node in Redis cluster"""
        node_key = f"{self.key_prefixes['node']}{self.node_id}"
        node_info = {
            'node_id': self.node_id,
            'registered_at': time.time(),
            'last_heartbeat': time.time(),
            'status': 'active'
        }
        
        await self.redis_client.hset(node_key, mapping=node_info)
        await self.redis_client.expire(node_key, 60)  # 60 second TTL
        
        logger.info(f"Node registered in Redis cluster: {self.node_id}")
    
    async def _unregister_node(self):
        """Unregister this node from Redis cluster"""
        node_key = f"{self.key_prefixes['node']}{self.node_id}"
        await self.redis_client.delete(node_key)
        logger.info(f"Node unregistered from Redis cluster: {self.node_id}")
    
    async def _health_check_loop(self):
        """Periodic health check and metrics update"""
        while self.running:
            try:
                # Update heartbeat
                node_key = f"{self.key_prefixes['node']}{self.node_id}"
                await self.redis_client.hset(node_key, 'last_heartbeat', time.time())
                await self.redis_client.expire(node_key, 60)
                
                # Update metrics
                await self._update_metrics()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)
    
    async def _update_metrics(self):
        """Update performance metrics"""
        try:
            info = await self.redis_client.info()
            
            # Update connection metrics
            self.metrics.connections_active = info.get('connected_clients', 0)
            self.metrics.memory_usage = info.get('used_memory', 0)
            
            # Calculate operations per second
            total_commands = info.get('total_commands_processed', 0)
            uptime = time.time() - self.start_time
            self.metrics.operations_per_sec = total_commands / uptime if uptime > 0 else 0
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    # Caching operations
    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None,
                       serialize: bool = True) -> bool:
        """Set value in distributed cache"""
        try:
            cache_key = f"{self.key_prefixes['cache']}{key}"
            
            if serialize:
                # Serialize complex objects
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                elif not isinstance(value, (str, int, float, bool)):
                    value = pickle.dumps(value)
            
            if ttl:
                result = await self.redis_client.setex(cache_key, ttl, value)
            else:
                result = await self.redis_client.set(cache_key, value)
            
            if result:
                self.metrics.sets += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def cache_get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """Get value from distributed cache"""
        try:
            cache_key = f"{self.key_prefixes['cache']}{key}"
            value = await self.redis_client.get(cache_key)
            
            if value is None:
                self.metrics.misses += 1
                return None
            
            self.metrics.hits += 1
            
            if deserialize:
                # Try to deserialize
                try:
                    # Try JSON first
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    try:
                        # Try pickle for complex objects
                        return pickle.loads(value)
                    except (pickle.PickleError, TypeError):
                        # Return as string
                        return value
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            self.metrics.misses += 1
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete value from distributed cache"""
        try:
            cache_key = f"{self.key_prefixes['cache']}{key}"
            result = await self.redis_client.delete(cache_key)
            
            if result:
                self.metrics.deletes += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def cache_exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            cache_key = f"{self.key_prefixes['cache']}{key}"
            return bool(await self.redis_client.exists(cache_key))
        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def cache_expire(self, key: str, ttl: int) -> bool:
        """Set expiration for cached value"""
        try:
            cache_key = f"{self.key_prefixes['cache']}{key}"
            result = await self.redis_client.expire(cache_key, ttl)
            
            if result:
                self.metrics.expires += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache expire failed for key {key}: {e}")
            return False
    
    # Session management
    async def session_create(self, session_id: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Create distributed session"""
        session_key = f"{self.key_prefixes['session']}{session_id}"
        
        session_data = {
            'session_id': session_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'node_id': self.node_id,
            **data
        }
        
        try:
            await self.redis_client.hset(session_key, mapping=session_data)
            await self.redis_client.expire(session_key, ttl)
            return True
        except Exception as e:
            logger.error(f"Session create failed for {session_id}: {e}")
            return False
    
    async def session_get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get distributed session data"""
        session_key = f"{self.key_prefixes['session']}{session_id}"
        
        try:
            session_data = await self.redis_client.hgetall(session_key)
            if session_data:
                # Update last accessed
                await self.redis_client.hset(session_key, 'last_accessed', time.time())
                return session_data
            return None
        except Exception as e:
            logger.error(f"Session get failed for {session_id}: {e}")
            return None
    
    async def session_update(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update distributed session"""
        session_key = f"{self.key_prefixes['session']}{session_id}"
        
        try:
            # Check if session exists
            if await self.redis_client.exists(session_key):
                data['last_accessed'] = time.time()
                await self.redis_client.hset(session_key, mapping=data)
                return True
            return False
        except Exception as e:
            logger.error(f"Session update failed for {session_id}: {e}")
            return False
    
    async def session_delete(self, session_id: str) -> bool:
        """Delete distributed session"""
        session_key = f"{self.key_prefixes['session']}{session_id}"
        
        try:
            result = await self.redis_client.delete(session_key)
            return bool(result)
        except Exception as e:
            logger.error(f"Session delete failed for {session_id}: {e}")
            return False
    
    # Distributed locks
    def get_lock(self, key: str, timeout: int = 10, lock_type: LockType = LockType.EXCLUSIVE) -> DistributedLock:
        """Get a distributed lock"""
        return DistributedLock(self.redis_client, key, timeout, lock_type=lock_type)
    
    @asynccontextmanager
    async def acquire_lock(self, key: str, timeout: int = 10, lock_type: LockType = LockType.EXCLUSIVE):
        """Context manager for acquiring distributed lock"""
        lock = self.get_lock(key, timeout, lock_type)
        async with lock:
            yield lock
    
    # State synchronization
    async def state_set(self, key: str, state: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set distributed state"""
        state_key = f"{self.key_prefixes['state']}{key}"
        
        state_data = {
            'data': json.dumps(state),
            'updated_at': time.time(),
            'updated_by': self.node_id
        }
        
        try:
            await self.redis_client.hset(state_key, mapping=state_data)
            if ttl:
                await self.redis_client.expire(state_key, ttl)
            return True
        except Exception as e:
            logger.error(f"State set failed for key {key}: {e}")
            return False
    
    async def state_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get distributed state"""
        state_key = f"{self.key_prefixes['state']}{key}"
        
        try:
            state_data = await self.redis_client.hgetall(state_key)
            if state_data and 'data' in state_data:
                return {
                    'data': json.loads(state_data['data']),
                    'updated_at': float(state_data.get('updated_at', 0)),
                    'updated_by': state_data.get('updated_by')
                }
            return None
        except Exception as e:
            logger.error(f"State get failed for key {key}: {e}")
            return None
    
    async def state_delete(self, key: str) -> bool:
        """Delete distributed state"""
        state_key = f"{self.key_prefixes['state']}{key}"
        
        try:
            result = await self.redis_client.delete(state_key)
            return bool(result)
        except Exception as e:
            logger.error(f"State delete failed for key {key}: {e}")
            return False
    
    # Coordination primitives
    async def coordinate_election(self, election_key: str, candidate_id: str, ttl: int = 30) -> bool:
        """Distributed leader election"""
        election_full_key = f"{self.key_prefixes['coordination']}election:{election_key}"
        
        try:
            # Try to become leader
            result = await self.redis_client.set(election_full_key, candidate_id, nx=True, ex=ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Election failed for {election_key}: {e}")
            return False
    
    async def coordinate_barrier(self, barrier_key: str, participant_count: int, participant_id: str, timeout: int = 60) -> bool:
        """Distributed synchronization barrier"""
        barrier_full_key = f"{self.key_prefixes['coordination']}barrier:{barrier_key}"
        
        try:
            # Add participant to barrier
            await self.redis_client.sadd(barrier_full_key, participant_id)
            await self.redis_client.expire(barrier_full_key, timeout)
            
            # Wait for all participants
            start_time = time.time()
            while time.time() - start_time < timeout:
                current_count = await self.redis_client.scard(barrier_full_key)
                if current_count >= participant_count:
                    return True
                await asyncio.sleep(0.1)
            
            return False
            
        except Exception as e:
            logger.error(f"Barrier coordination failed for {barrier_key}: {e}")
            return False
    
    # Utility methods
    async def get_active_nodes(self) -> List[Dict[str, Any]]:
        """Get list of active nodes"""
        try:
            node_keys = await self.redis_client.keys(f"{self.key_prefixes['node']}*")
            nodes = []
            
            for node_key in node_keys:
                node_data = await self.redis_client.hgetall(node_key)
                if node_data:
                    nodes.append(node_data)
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to get active nodes: {e}")
            return []
    
    async def cleanup_expired_data(self):
        """Clean up expired data and dead nodes"""
        try:
            current_time = time.time()
            
            # Clean up dead nodes (no heartbeat in 2 minutes)
            node_keys = await self.redis_client.keys(f"{self.key_prefixes['node']}*")
            for node_key in node_keys:
                node_data = await self.redis_client.hgetall(node_key)
                if node_data:
                    last_heartbeat = float(node_data.get('last_heartbeat', 0))
                    if current_time - last_heartbeat > 120:  # 2 minutes
                        await self.redis_client.delete(node_key)
                        logger.info(f"Cleaned up dead node: {node_key}")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Redis metrics"""
        return {
            'node_id': self.node_id,
            'running': self.running,
            'redis_connected': self.redis_client is not None,
            'uptime_seconds': time.time() - self.start_time,
            'cache_metrics': asdict(self.metrics),
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'db': self.config.db,
                'max_connections': self.config.max_connections
            }
        }
    
    async def health_check(self) -> bool:
        """Check Redis connectivity and health"""
        try:
            if not self.redis_client:
                return False
            
            # Test basic operations
            test_key = f"health_check:{self.node_id}:{int(time.time())}"
            await self.redis_client.set(test_key, "ok", ex=5)
            result = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            return result == "ok"
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


# Module-level functions
async def create_redis_cluster_manager(node_id: str, config: Optional[RedisConfig] = None) -> RedisClusterManager:
    """Create and start a Redis cluster manager instance"""
    manager = RedisClusterManager(node_id, config)
    await manager.start()
    return manager
