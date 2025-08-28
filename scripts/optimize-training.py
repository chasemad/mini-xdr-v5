#!/usr/bin/env python3
"""
Model Training Optimization Script
Optimizes and accelerates ML model training for adaptive detection
"""
import requests
import time
import json
import asyncio
from datetime import datetime
import argparse

BASE_URL = "http://localhost:8000"

class TrainingOptimizer:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def check_system_status(self) -> dict:
        """Check current system status"""
        try:
            response = requests.get(f"{self.base_url}/api/adaptive/status")
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}
    
    def force_learning_update(self) -> dict:
        """Force immediate learning update"""
        try:
            response = requests.post(f"{self.base_url}/api/adaptive/force_learning")
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}
    
    def adjust_sensitivity(self, level: str) -> dict:
        """Adjust detection sensitivity"""
        try:
            response = requests.post(
                f"{self.base_url}/api/adaptive/sensitivity",
                json={"sensitivity": level}
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}
    
    def retrain_ml_models(self) -> dict:
        """Trigger ML model retraining"""
        try:
            response = requests.post(
                f"{self.base_url}/api/ml/retrain",
                json={"model_type": "ensemble"}
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}
    
    def get_training_statistics(self) -> dict:
        """Get current training statistics"""
        try:
            # Get incident count for training data assessment
            incidents = requests.get(f"{self.base_url}/incidents").json()
            
            # Get ML status
            ml_status = requests.get(f"{self.base_url}/api/ml/status").json()
            
            # Get adaptive status
            adaptive_status = requests.get(f"{self.base_url}/api/adaptive/status").json()
            
            return {
                "total_incidents": len(incidents),
                "ml_models_trained": ml_status.get("metrics", {}).get("models_trained", 0),
                "learning_pipeline_running": adaptive_status.get("learning_pipeline", {}).get("running", False),
                "baseline_ips": adaptive_status.get("baseline_engine", {}).get("per_ip_baselines", 0),
                "last_update": adaptive_status.get("learning_pipeline", {}).get("learning_metrics", {}).get("last_baseline_update")
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def optimize_training_schedule(self):
        """Optimize the training schedule for faster learning"""
        print("🔧 Optimizing training schedule...")
        
        # Step 1: Assess current state
        stats = self.get_training_statistics()
        print(f"📊 Current Statistics:")
        print(f"   • Total incidents: {stats.get('total_incidents', 0)}")
        print(f"   • ML models trained: {stats.get('ml_models_trained', 0)}")
        print(f"   • Baseline IPs: {stats.get('baseline_ips', 0)}")
        print(f"   • Learning pipeline: {'✅' if stats.get('learning_pipeline_running') else '❌'}")
        
        # Step 2: Force immediate learning if needed
        if stats.get('baseline_ips', 0) < 5:
            print("\n🔄 Triggering immediate baseline learning...")
            result = self.force_learning_update()
            if result.get('success'):
                print("✅ Baseline learning triggered")
            else:
                print(f"❌ Learning failed: {result}")
        
        # Step 3: Retrain ML models if we have enough data
        if stats.get('total_incidents', 0) >= 10:
            print("\n🧠 Retraining ML models with available data...")
            result = self.retrain_ml_models()
            if result.get('success'):
                print(f"✅ ML models retrained with {result.get('training_data_size', 0)} samples")
            else:
                print(f"❌ ML retraining failed: {result.get('message', 'Unknown error')}")
        else:
            print(f"\n⚠️ Need more data for ML training (have {stats.get('total_incidents', 0)}, need 10+)")
        
        # Step 4: Adjust sensitivity for faster initial detection
        print("\n⚙️ Optimizing detection sensitivity...")
        result = self.adjust_sensitivity("high")
        if result.get('success'):
            print("✅ Detection sensitivity increased for faster learning")
        
        return stats
    
    async def continuous_training_mode(self, duration_minutes: int = 30):
        """Run continuous training optimization"""
        print(f"🔄 Starting continuous training mode for {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            print(f"\n--- Training Iteration {iteration} ---")
            
            # Force learning update every iteration
            result = self.force_learning_update()
            if result.get('success'):
                print("✅ Learning update successful")
                
                # Show results
                learning_results = result.get('results', {})
                for key, value in learning_results.items():
                    status = "✅" if value else "❌"
                    print(f"   {status} {key}: {value}")
            else:
                print(f"❌ Learning update failed: {result}")
            
            # Wait before next iteration
            await asyncio.sleep(120)  # 2 minutes between iterations
        
        print(f"\n🎉 Continuous training completed after {iteration} iterations")
        
        # Final status check
        final_stats = self.get_training_statistics()
        print("\n📊 Final Statistics:")
        for key, value in final_stats.items():
            print(f"   • {key}: {value}")

async def main():
    parser = argparse.ArgumentParser(description="Optimize ML model training")
    parser.add_argument("--mode", choices=["optimize", "continuous", "status"], 
                       default="optimize", help="Training optimization mode")
    parser.add_argument("--duration", type=int, default=30,
                       help="Duration for continuous mode (minutes)")
    
    args = parser.parse_args()
    
    optimizer = TrainingOptimizer()
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not healthy. Please start Mini-XDR first.")
            return
    except Exception:
        print("❌ Cannot connect to backend. Please start Mini-XDR first.")
        return
    
    print("🧠 ML Training Optimizer for Adaptive Detection")
    print("=" * 50)
    
    if args.mode == "status":
        stats = optimizer.get_training_statistics()
        print("📊 Current Training Status:")
        for key, value in stats.items():
            print(f"   • {key}: {value}")
        
        # Check adaptive status
        status = optimizer.check_system_status()
        if 'adaptive_engine' in status:
            print(f"\n🎯 Detection Configuration:")
            print(f"   • Behavioral threshold: {status['adaptive_engine'].get('behavioral_threshold', 'N/A')}")
            print(f"   • Learning pipeline: {'Running' if status.get('learning_pipeline', {}).get('running') else 'Stopped'}")
    
    elif args.mode == "optimize":
        await optimizer.optimize_training_schedule()
        
        print("\n🎯 Training Optimization Complete!")
        print("\n💡 Tips for Better Training:")
        print("   • Generate more data: python scripts/generate-training-data.py")
        print("   • Run continuous training: python scripts/optimize-training.py --mode continuous")
        print("   • Monitor with: python scripts/optimize-training.py --mode status")
    
    elif args.mode == "continuous":
        await optimizer.continuous_training_mode(args.duration)

if __name__ == "__main__":
    asyncio.run(main())
