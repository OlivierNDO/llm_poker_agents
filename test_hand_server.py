#!/usr/bin/env python3
"""
Simple test to check if hand_analysis_server.py can start at all
"""

import subprocess
import sys
import os
import time

def find_server_script():
    """Find the hand analysis server script"""
    possible_paths = [
        "hand_analysis_server.py",
        "mcp_servers/hand_analysis_server.py", 
        "../hand_analysis_server.py",
        os.path.join(os.path.dirname(__file__), "hand_analysis_server.py"),
        os.path.join(os.path.dirname(__file__), "mcp_servers", "hand_analysis_server.py")
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        print(f"Checking: {abs_path}")
        if os.path.exists(path):
            print(f"✅ Found: {abs_path}")
            return path
        else:
            print(f"❌ Not found: {abs_path}")
    
    return None

def test_server_import():
    """Test if the server script can be imported/run"""
    script_path = find_server_script()
    
    if not script_path:
        print("\n❌ Could not find hand_analysis_server.py")
        return False
    
    print(f"\n🧪 Testing server script: {script_path}")
    
    # Try to run the script with --help or similar to see if it can start
    try:
        # Start the process but don't send any input
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Process started, waiting 2 seconds...")
        time.sleep(2)
        
        # Check if process is still running (it should be waiting for input)
        if process.poll() is None:
            print("✅ Server process started and is running")
            
            # Try to terminate gracefully
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Server terminated gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                print("Server killed forcefully")
            
            return True
        else:
            print(f"❌ Server process exited immediately with code: {process.poll()}")
            
            # Read any error output
            stdout, stderr = process.communicate()
            if stdout:
                print(f"Stdout: {stdout}")
            if stderr:
                print(f"Stderr: {stderr}")
            
            return False
            
    except Exception as e:
        print(f"❌ Failed to start server process: {e}")
        return False

def test_python_imports():
    """Test if the required Python modules can be imported"""
    print("\n🧪 Testing Python imports...")
    
    required_modules = [
        'json',
        'sys', 
        'os',
        'numpy',
        'typing'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🔍 Diagnosing hand_analysis_server.py startup issues...\n")
    
    # Test 1: Python imports
    if not test_python_imports():
        print("\n❌ Python import test failed")
        sys.exit(1)
    
    # Test 2: Find and test server script
    if not test_server_import():
        print("\n❌ Server startup test failed")
        sys.exit(1)
    
    print("\n✅ All tests passed! The server should be able to start properly.")