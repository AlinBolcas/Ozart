"""
OpenAI API Test - Tool Calling Verification

This script tests the tool calling capabilities of the OpenAI API wrapper.
It creates a dummy function and verifies that the LLM can properly call it
and integrate the results into its response.
"""

import os
import sys
import json
from pathlib import Path
import hashlib
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from I_integrations.openai_API import OpenAIAPI

def generate_secret_code(topic: str, length: int = 6, include_numbers: bool = False) -> str:
    """
    Generate a unique secret code based on the provided topic.
    
    :param topic: The topic to base the code on
    :param length: The length of the code (default: 6)
    :param include_numbers: Whether to include numbers in the code (default: False)
    :return: A unique secret code
    """
    print("\n" + "="*60)
    print(f"🔴 FUNCTION CALLED: generate_secret_code(topic='{topic}', length={length}, include_numbers={include_numbers})")
    
    # Create a deterministic but unique code based on input
    hash_value = hashlib.md5(topic.encode()).hexdigest()
    
    # Generate the code
    consonants = 'bcdfghjklmnpqrstvwxyz'
    vowels = 'aeiou'
    digits = '0123456789'
    
    code = ""
    for i in range(length):
        hash_char = int(hash_value[i % len(hash_value)], 16)
        
        if i % 2 == 0:
            # Even positions get consonants
            code += consonants[hash_char % len(consonants)]
        else:
            # Odd positions get vowels
            code += vowels[hash_char % len(vowels)]
            
        # Add a digit if requested and we're at position 2 or 4
        if include_numbers and i in [2, 4]:
            code += digits[hash_char % len(digits)]
    
    print(f"🔵 SECRET CODE GENERATED: '{code}'")
    print("="*60 + "\n")
    
    return code

def test_tool_calling():
    """Test the tool calling functionality of the OpenAI API."""
    print("\n" + "="*80)
    print("🚀 TESTING OPENAI API TOOL CALLING")
    print("="*80)
    
    # Initialize the API
    api = OpenAIAPI(model="gpt-4o-mini", temperature=0.7)
    
    # Convert our function to a tool schema
    tool_schema = api.convert_function_to_schema(generate_secret_code)
    
    # Display the schema
    print("\n📋 FUNCTION SCHEMA:")
    print(json.dumps(tool_schema, indent=2))
    
    # Define available tools
    available_tools = {"generate_secret_code": generate_secret_code}
    
    # First test: Simple direct tool calling
    print("\n🔧 TEST 1: DIRECT TOOL CALLING")
    prompt_1 = "I need a secret code for my 'cybersecurity' project. Can you generate one with a length of 8 that includes numbers?"
    
    print(f"\nUser: {prompt_1}")
    
    response_1 = api.chat_completion(
        user_prompt=prompt_1,
        system_prompt="You are a helpful assistant with access to special tools. Use them when appropriate.",
        tools=[tool_schema],
        available_tools=available_tools
    )
    
    print(f"\nAssistant: {response_1}")
    
    # Second test: More complex reasoning
    print("\n🔧 TEST 2: COMPLEX REASONING WITH TOOL")
    prompt_2 = """I'm working on three different projects:
1. A machine learning algorithm for image recognition
2. A blockchain application for supply chain
3. A virtual reality game set in ancient Egypt

Can you generate appropriate secret codes for each of these projects? For the VR game, make sure to include numbers in the code."""
    
    print(f"\nUser: {prompt_2}")
    
    response_2 = api.chat_completion(
        user_prompt=prompt_2,
        system_prompt="You are a helpful assistant with access to special tools. Use them when appropriate.",
        tools=[tool_schema],
        available_tools=available_tools
    )
    
    print(f"\nAssistant: {response_2}")
    
    # Third test: Tool not required
    print("\n🔧 TEST 3: TOOL NOT REQUIRED")
    prompt_3 = "What are three key benefits of using AI in healthcare?"
    
    print(f"\nUser: {prompt_3}")
    
    response_3 = api.chat_completion(
        user_prompt=prompt_3,
        system_prompt="You are a helpful assistant with access to special tools. Use them when appropriate.",
        tools=[tool_schema],
        available_tools=available_tools
    )
    
    print(f"\nAssistant: {response_3}")
    
    print("\n" + "="*80)
    print("✅ TOOL CALLING TESTS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_tool_calling() 