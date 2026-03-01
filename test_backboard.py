"""
test_backboard.py — Quick test to verify Backboard integration.

Run this before the full pipeline to ensure API key and connection work.
"""

import os
from dotenv import load_dotenv
from utils.backboard_client import get_backboard_client

load_dotenv()


def test_connection():
    """Test basic Backboard API connection."""
    print("=" * 60)
    print("Testing Backboard Integration")
    print("=" * 60)
    
    # Check environment variables
    api_key = os.getenv("BACKBOARD_API_KEY")
    assistant_id = os.getenv("BACKBOARD_ASSISTANT_ID")
    model_provider = os.getenv("BACKBOARD_MODEL_PROVIDER", "cohere")
    model_name = os.getenv("BACKBOARD_MODEL_NAME", "command-r7b-12-2024")
    
    print(f"\n✓ API Key: {'*' * 20}{api_key[-4:] if api_key else 'NOT SET'}")
    print(f"✓ Assistant ID: {assistant_id or 'Will create new'}")
    print(f"✓ Model: {model_provider}/{model_name}")
    
    if not api_key:
        print("\n❌ ERROR: BACKBOARD_API_KEY not set in .env file")
        return False
    
    # Test client initialization
    try:
        print("\n[1/3] Initializing Backboard client...")
        client = get_backboard_client()
        print("✓ Client initialized")
        
        print("\n[2/3] Testing simple query...")
        response = client.invoke("What is machine learning? Answer in one sentence.", memory=False)
        print(f"✓ Response received: {response[:100]}...")
        
        print("\n[3/3] Testing memory feature...")
        client.reset_thread()  # Start fresh
        client.invoke("Remember: My research topic is federated learning.", memory=True)
        recall = client.invoke("What is my research topic?", memory=True)
        print(f"✓ Memory test: {recall[:100]}...")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! Backboard is ready to use.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Verify BACKBOARD_API_KEY is correct in .env")
        print("2. Check internet connection")
        print("3. Ensure model is available on your Backboard account")
        return False


if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
