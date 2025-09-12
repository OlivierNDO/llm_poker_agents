#!/usr/bin/env python3
"""
Test script to debug hand analysis server lifecycle issues
"""

import subprocess
import json
import time
import sys

def test_server_lifecycle():
    """Test if server survives multiple requests"""
    
    print("Starting hand analysis server...")
    
    # Start the server process
    process = subprocess.Popen(
        [sys.executable, "hand_analysis_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    try:
        # Test request 1 - tools/list
        print("\n--- Test 1: List Tools ---")
        request1 = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        request1_json = json.dumps(request1) + "\n"
        
        print(f"Sending: {request1_json.strip()}")
        process.stdin.write(request1_json)
        process.stdin.flush()
        
        # Read response
        response1 = process.stdout.readline()
        print(f"Received: {response1.strip()}")
        
        # Check if process is still alive
        if process.poll() is not None:
            print(f"❌ Server died after first request! Exit code: {process.poll()}")
            stderr_output = process.stderr.read()
            print(f"Stderr: {stderr_output}")
            return False
        
        print("✅ Server survived first request")
        
        # Test request 2 - hand analysis
        print("\n--- Test 2: Hand Analysis ---")
        request2 = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_hand_features",
                "arguments": {
                    "hole_cards": ["Ah", "Kd"],
                    "board_cards": ["Qh", "Jc", "Ts"],
                    "num_opponents": 3
                }
            }
        }
        request2_json = json.dumps(request2) + "\n"
        
        print(f"Sending: {request2_json.strip()}")
        process.stdin.write(request2_json)
        process.stdin.flush()
        
        # Read response
        response2 = process.stdout.readline()
        print(f"Received: {response2.strip()}")
        
        # Check if process is still alive
        if process.poll() is not None:
            print(f"❌ Server died after second request! Exit code: {process.poll()}")
            stderr_output = process.stderr.read()
            print(f"Stderr: {stderr_output}")
            return False
        
        print("✅ Server survived second request")
        
        # Test request 3 - another hand analysis (simulate second player)
        print("\n--- Test 3: Second Player Hand Analysis ---")
        request3 = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_hand_features",
                "arguments": {
                    "hole_cards": ["7h", "7d"],
                    "board_cards": ["Qh", "Jc", "Ts"],
                    "num_opponents": 3
                }
            }
        }
        request3_json = json.dumps(request3) + "\n"
        
        print(f"Sending: {request3_json.strip()}")
        process.stdin.write(request3_json)
        process.stdin.flush()
        
        # Read response
        response3 = process.stdout.readline()
        print(f"Received: {response3.strip()}")
        
        # Check if process is still alive
        if process.poll() is not None:
            print(f"❌ Server died after third request! Exit code: {process.poll()}")
            stderr_output = process.stderr.read()
            print(f"Stderr: {stderr_output}")
            return False
        
        print("✅ Server survived third request")
        
        # Test request 4 - one more to be sure
        print("\n--- Test 4: Final Test ---")
        request4 = {"jsonrpc": "2.0", "id": 4, "method": "tools/list"}
        request4_json = json.dumps(request4) + "\n"
        
        print(f"Sending: {request4_json.strip()}")
        process.stdin.write(request4_json)
        process.stdin.flush()
        
        # Read response
        response4 = process.stdout.readline()
        print(f"Received: {response4.strip()}")
        
        # Final check
        if process.poll() is not None:
            print(f"❌ Server died after fourth request! Exit code: {process.poll()}")
            stderr_output = process.stderr.read()
            print(f"Stderr: {stderr_output}")
            return False
        
        print("✅ All tests passed! Server is stable.")
        return True
        
    finally:
        # Clean shutdown
        print("\n--- Shutting Down Server ---")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
                print("Server terminated gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                print("Server killed forcefully")
        
        # Print any stderr output
        stderr_output = process.stderr.read()
        if stderr_output:
            print(f"\nServer stderr output:\n{stderr_output}")

if __name__ == "__main__":
    test_server_lifecycle()