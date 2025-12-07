#!/usr/bin/env python3
"""
Test script to initialize and verify distributed MCP system components
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_distributed_system():
    """Test the distributed MCP system initialization"""
    print("🚀 Testing Distributed MCP System")
    print("=" * 50)

    try:
        print("1. Testing Redis connection...")
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        pong = r.ping()
        print(f"   ✅ Redis: {pong}")
    except Exception as e:
        print(f"   ❌ Redis failed: {e}")
        return False

    try:
        print("\n2. Testing Kafka connection...")
        from aiokafka import AIOKafkaProducer
        producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')
        await producer.start()
        print("   ✅ Kafka: Connected")
        await producer.stop()
    except Exception as e:
        print(f"   ❌ Kafka failed: {e}")
        return False

    try:
        print("\n3. Testing distributed module import...")
        from backend.app.distributed import get_system_status, DISTRIBUTED_CAPABILITIES
        print(f"   ✅ Import successful")
        print(f"   📋 Capabilities: {len(DISTRIBUTED_CAPABILITIES)} features")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

    try:
        print("\n4. Getting system status...")
        status = get_system_status()
        print(f"   ✅ Status retrieved")
        print(f"   📊 Components: {list(status.get('components', {}).keys())}")
    except Exception as e:
        print(f"   ❌ Status failed: {e}")
        return False

    try:
        print("\n5. Testing distributed system initialization...")
        from backend.app.distributed import initialize_distributed_system
        result = await initialize_distributed_system()
        print(f"   📋 Initialization result: {result.get('success', False)}")
        if result.get('errors'):
            print(f"   ⚠️ Errors: {result['errors']}")
        for component, details in result.get('components', {}).items():
            status = details.get('status', 'unknown')
            print(f"   📦 {component}: {status}")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        return False

    print("\n🎉 Distributed system test completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_distributed_system())
    sys.exit(0 if success else 1)