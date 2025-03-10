import os
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any, Type, Iterator
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import sys
import time
import tempfile
import requests
import logging

# Disable noisy HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

class OpenAIAPI:
    """
    Streamlined wrapper for OpenAI's API services.
    Provides simplified access to key OpenAI capabilities.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_message: str = "You are a helpful assistant.",
        api_key: Optional[str] = None
    ):  
        # Load from .env file
        load_dotenv(override=True)
        
        # Use provided API key if available, otherwise use environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: No OpenAI API key provided or found in environment")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Settings
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        
        # Log initialization without showing API key
        logging.info(f"OpenAI API initialized with model: {self.model}")

    def _create_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Create properly formatted message list for OpenAI API."""
        messages = []
        
        # Add system message first if provided
        system_msg = system_prompt or self.system_message
        if system_msg:
            messages.append({
                "role": "system",
                "content": system_msg
            })
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Add current user prompt
        if isinstance(user_prompt, str):
            messages.append({"role": "user", "content": user_prompt})
        else:
            # Handle case when user_prompt is a list (for vision analysis)
            messages.append({"role": "user", "content": user_prompt})
        
        return messages

    def chat_completion(
        self, 
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        message_history: List[Dict[str, str]] = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        tools: List[Dict] = None,
        available_tools: Dict[str, callable] = None,
        **kwargs
    ) -> str:
        """
        Generate a chat completion with optional tools.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: The system prompt
            message_history: History of previous messages
            model: Model to use
            temperature: Controls randomness
            max_tokens: Max tokens to generate
            tools: List of tool schemas
            available_tools: Dictionary mapping tool names to callable functions
            
        Returns:
            Generated response
        """
        try:
            # Prepare messages
            messages = message_history or []
            
            # Add system message if not already in history
            if not any(msg.get("role") == "system" for msg in messages):
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": user_prompt})
            
            # Prepare parameters
            params = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                **kwargs
            }
            
            # Add tools if provided
            if tools:
                params["tools"] = tools
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            # Check for tool calls
            message = response.choices[0].message
            
            # If no tool calls, return content
            if not (hasattr(message, 'tool_calls') and message.tool_calls):
                return message.content
            
            # Process tool calls
            if tools and available_tools:
                # Store original assistant message
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id, 
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                # Process each tool call
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    
                    if func_name in available_tools:
                        # Parse arguments
                        try:
                            func_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            func_args = {}
                        
                        # Call the function
                        try:
                            result = available_tools[func_name](**func_args)
                        except Exception as e:
                            result = f"Error executing {func_name}: {str(e)}"
                        
                        # Convert result to string if needed
                        if not isinstance(result, str):
                            result = json.dumps(result)
                        
                        # Add tool response
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                
                # Make second API call with tool results
                second_response = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens
                )
                
                return second_response.choices[0].message.content
            
            return message.content
            
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return f"An error occurred: {str(e)}"

    def reasoned_completion(
        self,
        user_prompt: str,
        reasoning_effort: str = "low",
        message_history: List[Dict[str, str]] = None,
        model: str = "o3-mini",
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        Generate a reasoned completion with explicit reasoning steps.
        
        Args:
            user_prompt: The user's prompt
            reasoning_effort: Level of reasoning detail (low, medium, high)
            message_history: Previous conversation history (non-system messages only)
            model: Model to use (default: o1-mini)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response with reasoning steps
        """
        try:
            # Print with better formatting
            print(f"\n{'>'*20} REASONED THINKING {'<'*20}")
            print(f"🧠 Using reasoning model '{model}' with reasoning_effort='{reasoning_effort}'")
            
            # Prepare messages - IMPORTANT: o1-mini doesn't support system messages
            # Filter out any system messages and only keep user/assistant messages
            messages = []
            if message_history:
                messages = [msg for msg in message_history if msg.get("role") != "system"]
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Make API call with the proper parameters for o1-mini model
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_tokens or self.max_tokens
            )
            
            print(f"{'='*20} REASONING COMPLETE {'='*20}\n")
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"\n❌ Error in reasoned completion: {e}")
            return f"Error: {e}"

    def vision_analysis(
        self,
        image_path: str,
        user_prompt: str = "What's in this image?",
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """Analyze images with vision capabilities."""
        try:
            # Handle URL or local path for image
            if image_path.startswith(('http://', 'https://')):
                image_data = {"url": image_path}
            else:
                # Read local file as base64
                import base64
                with open(image_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                    image_data = {"url": f"data:image/jpeg;base64,{base64_image}"}
            
            # Create multipart content for vision
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": image_data}
            ]
            
            # Use chat_completion with our prepared content
            return self.chat_completion(
                user_prompt=content,
                system_prompt=system_prompt,
                model=model or "gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                message_history=message_history,
                **kwargs
            )

        except Exception as e:
            print(f"Error in vision analysis: {e}")
            return f"Error: {str(e)}"

    def structured_output(
        self,
        user_prompt: str,
        output_schema: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Any:
        """Generate structured JSON outputs."""
        try:
            # Enhance system prompt for JSON output
            json_system_prompt = (
                (system_prompt or self.system_message) + "\n\n"
                "IMPORTANT: You must respond with a valid JSON object. "
                "No other text or explanation should be included in your response."
            )
            
            # Create messages
            messages = self._create_messages(
                user_prompt=user_prompt,
                system_prompt=json_system_prompt,
                message_history=message_history
            )
            
            # Add schema if provided
            response_format = {"type": "json_object"} 
            if output_schema:
                # Advanced: can include schema validation requirements
                response_format["schema"] = output_schema
            
            # Make API request
            response_obj = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature or 0.2,  # Lower temperature for structured output
                max_tokens=max_tokens or self.max_tokens,
                response_format=response_format,
                **kwargs
            )
            
            # Extract and parse JSON
            response = response_obj.choices[0].message.content
            
            try:
                # Parse the JSON response
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback: Try to extract JSON from markdown or code blocks
                content = response.strip()
                
                # Extract from markdown code block
                if "```json" in content:
                    try:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        return json.loads(json_str)
                    except (IndexError, json.JSONDecodeError):
                        pass
                
                # Extract from generic code block
                if "```" in content:
                    try:
                        json_str = content.split("```")[1].strip()
                        return json.loads(json_str)
                    except (IndexError, json.JSONDecodeError):
                        pass
                
                # Return error if parsing fails
                print(f"Failed to parse JSON from response: {content}")
                return {"error": "JSON parsing failed", "raw_response": content}

        except Exception as e:
            print(f"Error in structured output: {e}")
            return {"error": str(e)}

    def create_embeddings(
        self, 
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """
        Generate embeddings for given text(s).
        
        Args:
            texts: Text or list of texts to embed
            model: Model to use for embeddings
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            model=model,
            input=texts
        )
        
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings, dtype='float32')

    def text_to_speech(
        self, 
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        speed: float = 1.0,
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """Convert text to speech using OpenAI's TTS."""
        try:
            # Create speech
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                speed=speed,
                response_format="mp3"
            )
            
            # Save to file if path provided
            if output_path:
                response.stream_to_file(output_path)
                return output_path
            
            # Return audio content
            return response.content
            
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return f"Error: {str(e)}"

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        model: str = "dall-e-3",
        **kwargs
    ) -> List[str]:
        """Generate images using DALL-E models."""
        try:
            # Generate images
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=n,
                **kwargs
            )
            
            # Return URLs of generated images
            return [img.url for img in response.data]
            
        except Exception as e:
            print(f"Error in image generation: {e}")
            return [f"Error: {str(e)}"]

    def convert_function_to_schema(self, func) -> Dict:
        """Convert a Python function to OpenAI's function schema format."""
        try:
            import inspect
            from typing import get_type_hints
            
            # Get function signature and docstring
            sig = inspect.signature(func)
            doc = func.__doc__ or ""
            type_hints = get_type_hints(func)
            
            # Create schema
            schema = {
                "name": func.__name__,
                "description": doc.split("\n")[0] if doc else "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Type mapping
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object"
            }
            
            # Add parameters
            for param_name, param in sig.parameters.items():
                # Skip self parameter
                if param_name == "self":
                    continue
                    
                # Get type and description
                param_type = type_hints.get(param_name, str)
                param_schema = {
                    "type": type_map.get(param_type, "string"),
                    "description": ""
                }
                
                # Try to extract parameter description from docstring
                param_desc_marker = f":param {param_name}:"
                if param_desc_marker in doc:
                    param_desc = doc.split(param_desc_marker)[1]
                    param_desc = param_desc.split("\n")[0].split(":param")[0].strip()
                    param_schema["description"] = param_desc
                
                # Add to schema
                schema["parameters"]["properties"][param_name] = param_schema
                
                # Add to required list if no default value
                if param.default == param.empty:
                    schema["parameters"]["required"].append(param_name)
            
            # Return formatted schema for OpenAI
            return {
                "type": "function",
                "function": schema
            }
            
        except Exception as e:
            print(f"Error converting function to schema: {e}")
            return {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": func.__doc__ or "",
                    "parameters": {"type": "object", "properties": {}}
                }
            }

    def _execute_tool_call(self, tool_call, available_tools: Dict[str, callable]) -> Any:
        """Execute a tool call with error handling."""
        try:
            # Extract function name and arguments
            func_name = tool_call.function.name
            args_str = tool_call.function.arguments
            
            # Parse arguments
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in arguments: {args_str}"
            
            # Check if function exists
            if func_name not in available_tools:
                return f"Error: Unknown tool '{func_name}'"
            
            # Execute function with arguments
            return available_tools[func_name](**args)
            
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _process_stream(self, response):
        """Process streaming response from OpenAI."""
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def transcribe_audio(
        self,
        audio_file_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0
    ) -> str:
        """Transcribe audio using OpenAI's Whisper model."""
        try:
            # Prepare parameters
            params = {
                "model": model,
                "file": open(audio_file_path, "rb"),
                "response_format": response_format,
                "temperature": temperature
            }
            
            # Add optional parameters if provided
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
            
            # Make API call
            response = self.client.audio.transcriptions.create(**params)
            
            # Return transcription text
            return response
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return f"Error: {str(e)}"


# Example usage
if __name__ == "__main__":
    import os
    import time
    import tempfile
    import subprocess
    from pathlib import Path
    
    print("\n" + "="*50)
    print("🚀 OPENAI API WRAPPER TEST SUITE")
    print("="*50)
    
    # Initialize API
    api = OpenAIAPI(model="gpt-4o-mini")
    
    # 1. Basic chat completion
    print("\n📝 TEST: Basic Chat Completion")
    
    system_prompt = "You are a helpful AI assistant specializing in creative technology."
    user_prompt = "What are three innovative ways AI can be used in music production?"
    
    result = api.chat_completion(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.8,
        max_tokens=200
    )
    
    print(f"System: {system_prompt}")
    print(f"User: {user_prompt}")
    print(f"Assistant: {result}")
    
    # 2. Message history and context retention
    print("\n🔄 TEST: Message History and Context Retention")
    
    conversation = [
        {"role": "system", "content": "You are a creative assistant."},
        {"role": "user", "content": "I want to write a song about AI."},
        {"role": "assistant", "content": "That's a great idea! What genre are you thinking of?"},
    ]
    
    follow_up = "I'm thinking of an electronic pop song with philosophical lyrics."
    
    history_result = api.chat_completion(
        user_prompt=follow_up,
        message_history=conversation
    )
    
    print("Previous conversation:")
    for msg in conversation:
        print(f"- {msg['role']}: {msg['content']}")
    print(f"User follow-up: {follow_up}")
    print(f"Assistant: {history_result}")
    
    # Add this response to conversation for future use
    conversation.append({"role": "user", "content": follow_up})
    conversation.append({"role": "assistant", "content": history_result})
    
    # 3. Reasoned completion
    print("\n🧠 TEST: Reasoned Completion")
    
    reasoned_prompt = "Explain the key differences between supervised and unsupervised learning."
    
    # Try to use o1-mini model, but don't fall back if it fails
    try:
        print("Attempting to use reasoning model o1-mini...")
        reasoned_result = api.reasoned_completion(
            user_prompt=reasoned_prompt,
            reasoning_effort="low",
            model="o1-mini",
            max_tokens=1000  # Give enough tokens for reasoning
        )
        
        # If we get an error message back, raise an exception
        if reasoned_result.startswith("Error:"):
            raise Exception(reasoned_result)
        
        print(f"User: {reasoned_prompt}")
        print(f"Assistant (reasoned): {reasoned_result}")
        
    except Exception as e:
        print(f"⚠️ Reasoning models test failed: {e}")
        print("Skipping reasoned completion test. This requires access to OpenAI's reasoning models.")
    
    # 4. Create embeddings
    print("\n🔢 TEST: Creating Embeddings")
    
    text_for_embedding = history_result  # Use our previous assistant response
    
    embeddings = api.create_embeddings(text_for_embedding)
    
    print(f"Text: '{text_for_embedding[:50]}...'")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Sample values: {embeddings[0][:5]}...")
    
    # 5. Structured output
    print("\n🧩 TEST: Structured JSON Output")
    
    struct_prompt = "Create a character profile for a sci-fi novel featuring an AI musician."
    
    json_result = api.structured_output(
        user_prompt=struct_prompt,
        temperature=0.7
    )
    
    print(f"User: {struct_prompt}")
    print(f"Structured Response:")
    print(json.dumps(json_result, indent=2))
    
    # 6. Tool/Function conversion and usage
    print("\n🔧 TEST: Tool Conversion and Usage")
    
    # Define a simple function that returns a secret word based on input
    def generate_secret_word(theme: str, length: int = 6) -> str:
        """Generate a 'secret' word based on the theme.
        
        :param theme: The theme to base the word on
        :param length: Desired length of the word (default: 6)
        :return: A secret word
        """
        import hashlib
        
        # Add a visible message when function is executed
        print("\n" + "="*50)
        print("🔴 FUNCTION EXECUTED! Generating secret word for theme: " + theme)
        print("="*50)
        
        # Create a deterministic but seemingly random word
        hash_object = hashlib.md5(theme.encode())
        hex_dig = hash_object.hexdigest()
        
        # Use the hash to create a word-like string of requested length
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        
        result = ""
        for i in range(length):
            # Use each hex digit to select a letter
            hex_val = int(hex_dig[i], 16)
            if i % 2 == 0:
                # Even positions get consonants
                result += consonants[hex_val % len(consonants)]
            else:
                # Odd positions get vowels
                result += vowels[hex_val % len(vowels)]
        
        # Print the result clearly
        print("🔵 Function result: '" + result + "'")
        print("="*50 + "\n")
        return result
    
    # Convert the function to a tool schema
    tool_schema = api.convert_function_to_schema(generate_secret_word)
    
    print(f"Function converted to tool schema:")
    print(json.dumps(tool_schema, indent=2))
    
    # Use the tool in a chat completion
    tool_prompt = "I need a secret word based on the theme of 'artificial intelligence'. Please generate one for me with a length of 7 characters."
    
    tool_result = api.chat_completion(
        user_prompt=tool_prompt,
        tools=[tool_schema],
        available_tools={"generate_secret_word": generate_secret_word}
    )
    
    print(f"User: {tool_prompt}")
    print(f"Assistant (with tool): {tool_result}")
    
    # 7. Image generation based on character and secret word
    print("\n🖼️ TEST: Image Generation")
    
    # Create a safe image prompt
    secret_word = generate_secret_word('music', 5)
    image_theme = f"A futuristic AI musician with the secret code '{secret_word}' performing on stage"
    
    try:
        image_urls = api.generate_image(
            prompt=image_theme,
            size="1024x1024",
            quality="standard",
            style="vivid",
            n=1
        )
        
        image_url = image_urls[0] if image_urls else None
        
        if image_url:
            print(f"Image prompt: {image_theme}")
            print(f"Generated image URL: {image_url}")
            
            # Open the image - first try to download it locally
            print("\n🖥️ Opening image...")
            try:
                # Create a temporary file to save the image
                temp_image_path = os.path.join(tempfile.gettempdir(), "openai_generated_image.png")
                
                # Download the image
                print(f"Downloading image to {temp_image_path}...")
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(temp_image_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    
                    # Open with different methods based on platform
                    if sys.platform == "darwin":  # Mac
                        # Try QuickLook first
                        print("Opening with QuickLook...")
                        ql_result = subprocess.run(["qlmanage", "-p", temp_image_path], 
                                                  stdout=subprocess.DEVNULL, 
                                                  stderr=subprocess.DEVNULL)
                        
                        # If QuickLook fails, try open command
                        if ql_result.returncode != 0:
                            print("QuickLook failed, trying open command...")
                            subprocess.run(["open", temp_image_path])
                            
                    elif sys.platform == "win32":  # Windows
                        os.startfile(temp_image_path)
                    else:  # Linux
                        try:
                            subprocess.run(["xdg-open", temp_image_path])
                        except:
                            print(f"Image saved to {temp_image_path}")
                else:
                    print(f"Failed to download image: HTTP {response.status_code}")
                    print("Try opening the URL in your browser manually:")
                    print(image_url)
            
            except Exception as e:
                print(f"Error displaying image: {e}")
                print("You can manually view the image by opening this URL in your browser:")
                print(image_url)
            
            # 8. Vision analysis
            print("\n👁️ TEST: Vision Analysis")
            
            vision_prompt = "Describe this image in detail, focusing on the musical aspects."
            
            vision_result = api.vision_analysis(
                image_path=image_url, 
                user_prompt=vision_prompt
            )
            
            print(f"Vision prompt: {vision_prompt}")
            print(f"Vision analysis: {vision_result}")
            
            # 9. Text-to-Speech
            print("\n🔊 TEST: Text-to-Speech")
            
            # Create a temporary file for the audio
            temp_audio_path = os.path.join(tempfile.gettempdir(), "ai_description.mp3")
            
            tts_result = api.text_to_speech(
                text=vision_result[:200],  # First 200 chars to keep it brief
                voice="nova",  # A good voice for descriptions
                output_path=temp_audio_path
            )
            
            print(f"Text converted to speech: '{vision_result[:50]}...'")
            print(f"Audio saved to: {temp_audio_path}")
            
            # Play the audio (ensure foreground playback)
            print("\n🔈 Playing audio...")
            if sys.platform == "darwin":  # Mac
                print("Playing audio with afplay...")
                subprocess.run(["afplay", temp_audio_path], check=True)
            elif sys.platform == "win32":  # Windows
                print("Opening audio with default player...")
                os.startfile(temp_audio_path)
            else:  # Linux
                try:
                    print("Playing audio with aplay...")
                    subprocess.run(["aplay", temp_audio_path], check=True)
                except:
                    print("Could not play audio automatically. File saved at:", temp_audio_path)
            
            # Wait a moment to ensure audio finished playing
            print("Waiting for audio playback to complete...")
            time.sleep(3)
            
            # 10. Audio Transcription
            print("\n🎤 TEST: Audio Transcription")
            transcript = api.transcribe_audio(
                audio_file_path=temp_audio_path,
                language="en"
            )
            
            print(f"Original text: '{vision_result[:50]}...'")
            print(f"Transcription: '{transcript}'")
            print(f"Transcription accuracy: {'High' if vision_result[:20].lower() in transcript.lower() else 'Low'}")
            
            print("\nTest cycle completed successfully!")
        else:
            print("❌ Image generation failed, skipping vision and TTS tests")
    
    except Exception as e:
        print(f"❌ Error during image/vision tests: {e}")
    
    print("\n" + "="*50)
    print("🏁 ALL TESTS COMPLETED")
    print("="*50) 