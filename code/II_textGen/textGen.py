"""
textGen.py

Unified TextGen interface integrating OpenAI API, memory management, 
retrieval augmented generation (RAG), and tools functionality.

This module serves as the central integration point for AI text generation,
providing a simplified interface for accessing all functionality with proper context
and memory management.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths for importing other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "I_integrations")))

# Import required modules
try:
    from I_integrations.openai_API import OpenAIAPI
    from II_textGen.tools import Tools
    from II_textGen.memory import Memory
    from II_textGen.rag import RAG
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure all required modules are available in their respective directories.")
    raise


class TextGen:
    """
    Unified TextGen: Integration Hub for OpenAI API, Memory, RAG, and Tools.

    Provides a simplified interface for:
    - Context-aware LLM completions and chat
    - Short and long-term memory retrieval
    - Tool integration with LLM calls
    - Vision and multimodal capabilities
    - Structured output parsing
    """
    
    def __init__(self, short_term_limit: int = 8000, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize TextGen with all required components.
        
        Args:
            short_term_limit: Token limit for short-term memory
            chunk_size: Size of chunks for RAG
            chunk_overlap: Overlap between chunks for RAG
        """
        # Initialize API clients
        self.openai = OpenAIAPI()
        self.tools = Tools()
        
        # Initialize memory and RAG
        self.memory = Memory(short_term_limit)
        self.rag = RAG(chunk_size, chunk_overlap)
        self.embedding = lambda text: self.openai.create_embeddings(text)
        
        # Register available tools
        self._register_tools()
        
        logger.info("TextGen initialized successfully")
        
    def _register_tools(self):
        """Register available tools from the Tools class."""
        self.available_tools = {}
        
        if not hasattr(self, "tools") or not self.tools:
            logger.warning("Tools not available for registration")
            return
            
        for attr_name in dir(self.tools):
            if not attr_name.startswith("_") and callable(getattr(self.tools, attr_name)):
                tool_func = getattr(self.tools, attr_name)
                self.available_tools[attr_name] = tool_func
                
        logger.info(f"Registered {len(self.available_tools)} tools")
        
    def _prepare_prompts(self, user_prompt: str, system_context: str = None, context: str = None) -> tuple:
        """
        Prepare final system and user messages with appropriate context.
        
        Args:
            user_prompt: The main user prompt
            system_context: Additional context to add to system message
            context: Additional context to add to user message
            
        Returns:
            Tuple of (final_system, final_user) with all context included
        """
        # Retrieve long-term context
        long_term_context = self.retrieve_memory_context(user_prompt)
            
        # Build final system message
        final_system = ""
        if system_context:
            final_system += system_context + "\n"
        if long_term_context:
            final_system += "Long term context:\n" + long_term_context
            
        # Build final user message
        final_user = ""
        if context:
            final_user += context + "\n"
        final_user += user_prompt
        
        return final_system, final_user
        
    def retrieve_memory_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant context from long-term memory.
        
        Args:
            query: The query to search for relevant context
            top_k: Number of top results to include
            
        Returns:
            String containing relevant context from memory
        """
        try:
            # Get insights from long-term memory
            insights = self.memory.retrieve_long_term()
            if not insights or len(insights) == 0:
                return ""
                
            # Process through RAG
            document = "\n".join([str(insight) for insight in insights])
            self.rag.ingest_documents(document, self.embedding)
            
            # Get context from RAG
            context = self.rag.retrieve_context(query, self.embedding, top_k=top_k)
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return ""
            
    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Get a list of all available tools and their descriptions.
        
        Returns:
            List of dictionaries with tool names and descriptions
        """
        tool_list = []
        
        for tool_name, tool_func in self.available_tools.items():
            description = tool_func.__doc__.strip() if tool_func.__doc__ else "No description available"
            tool_list.append({
                "name": tool_name,
                "description": description
            })
            
        return tool_list
        
    def chat_completion(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        system_context: str = None,
        context: str = None,
        tool_names: List[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> str:
        """
        Generate a chat completion with context and tool integration.
        
        Args:
            user_prompt: The main user prompt
            system_prompt: Base system prompt for the LLM
            system_context: Additional context for system message
            context: Additional context for user message
            tool_names: List of tool names to make available
            temperature: Generation temperature (higher = more random)
            max_tokens: Maximum tokens to generate
            store_interaction: Whether to store the interaction in memory
            
        Returns:
            Generated response text
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
        
        # Prepare tools if specified
        tools = None
        if tool_names and self.tools:
            tools = []
            for tool_name in tool_names:
                if tool_name in self.available_tools:
                    tool_func = self.available_tools[tool_name]
                    # Convert tool function to schema
                    tool_schema = self._convert_function_to_schema(tool_func)
                    tools.append(tool_schema)
        
        # Make API call with tools
        response = self.openai.chat_completion(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            available_tools=self.available_tools if tools else None,
            **kwargs
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, response)
            
        return response

    def _convert_function_to_schema(self, func) -> Dict:
        """Convert a Python function to OpenAI function schema format."""
        import inspect
        from typing import get_type_hints
        
        # Get function signature and docstring
        sig = inspect.signature(func)
        doc = func.__doc__ or ""
        type_hints = get_type_hints(func)
        
        # Create parameter schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Map Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        # Process each parameter
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Get parameter type
            param_type = type_hints.get(param_name, str)
            json_type = type_map.get(param_type, "string")
            
            # Get parameter description from docstring
            param_desc = ""
            if doc:
                for line in doc.split('\n'):
                    if f":param {param_name}:" in line:
                        param_desc = line.split(f":param {param_name}:")[1].strip()
                        break
            
            # Add parameter to schema
            parameters["properties"][param_name] = {
                "type": json_type,
                "description": param_desc
            }
            
            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        # Create function schema
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc.split('\n')[0] if doc else "",
                "parameters": parameters
            }
        }
        
        return schema
        
    def structured_output(
        self,
        user_prompt: str,
        system_prompt: str = "Return the output in structured JSON format.",
        system_context: str = None,
        context: str = None,
        temperature: float = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> Any:
        """
        Generate structured JSON output from the LLM.
        
        Args:
            user_prompt: The main user prompt
            system_prompt: Base system prompt for the LLM
            system_context: Additional context for system message
            context: Additional context for user message
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            store_interaction: Whether to store the interaction in memory
            
        Returns:
            Parsed JSON response
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
            
        # Get structured output
        response = self.openai.structured_output(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, str(response))
            
        return response
    
    def vision_analysis(
        self,
        image_url: str,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant with image analysis capabilities.",
        system_context: str = None,
        context: str = None,
        temperature: float = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> str:
        """
        Analyze an image with vision capabilities.
        
        Args:
            image_url: URL of the image to analyze
            user_prompt: The main user prompt
            system_prompt: Base system prompt for the LLM
            system_context: Additional context for system message
            context: Additional context for user message
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            store_interaction: Whether to store the interaction in memory
            
        Returns:
            Generated response text
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
            
        # Use vision_analysis instead of vision_completion
        response = self.openai.vision_analysis(
            image_path=image_url,
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, response)
            
        return response

    def reasoned_completion(
        self,
        user_prompt: str,
        reasoning_effort: str = "low",
        context: str = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> str:
        """Generate a reasoned completion."""
        # Get message history - filter out system messages for o1-mini
        message_history = self.memory.retrieve_short_term_formatted()
        if message_history:
            message_history = [msg for msg in message_history if msg.get("role") != "system"]
        
        # Prepare context-enriched user prompt (don't use system_context)
        final_user = user_prompt
        if context:
            final_user = context + "\n\n" + user_prompt
        
        # Log what we're doing
        print(f"🧠 Using reasoning model with context-enriched prompt")
        
        # Remove temperature and system_prompt for o1-mini model
        response = self.openai.reasoned_completion(
            user_prompt=final_user,
            message_history=message_history,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            **{k:v for k,v in kwargs.items() if k not in ['temperature', 'system_prompt']}
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term("Reasoned Completion", user_prompt, response)
        
        return response
        
    # Memory delegation methods
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")
        
    def save_history(self, filename: Optional[str] = None) -> str:
        """
        Save conversation history.
        
        Args:
            filename: Optional filename (ignored in current implementation)
            
        Returns:
            Path to the memory directory
        """
        # With the updated memory implementation, we don't need a separate filename
        # since it now just syncs the existing memory files
        memory_dir = self.memory.save()
        logger.info(f"Memory saved to directory: {memory_dir}")
        return memory_dir
        
    def load_history(self, filename: str) -> bool:
        """
        Load conversation history from a file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure we have the correct JSON file format
            if not filename.lower().endswith('.json'):
                filename += '.json'
            
            # Use memory's load method (which now expects a specific format)
            self.memory.load(filename)
            logger.info(f"Loaded memory from: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return False


# Example usage focused on AI Art & Music project brainstorming
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 🎵 ✨ AI ART & MUSIC PROJECT BRAINSTORMING ✨ 🎵 🎨")
    print("="*60 + "\n")
    
    text_gen = TextGen()
    
    try:
        # Demo 1: Initial Project Scoping
        print("\n" + "="*50)
        print("🏗️  PHASE 1: PROJECT SCOPING & ARCHITECTURE")
        print("="*50)
        response = text_gen.chat_completion(
            user_prompt="""Give an extremely concise 1-paragraph summary of a system architecture for generating art from music. Cover only the essential components.""",
            system_prompt="You are a concise AI systems architect. Keep all responses under 200 tokens in exactly one paragraph.",
            max_tokens=150
        )
        print("\n" + "-"*50)
        print("📋 ARCHITECTURAL INSIGHTS:")
        print("-"*50)
        print(f"\n{response}\n")
        
        # Demo 2: Research Current Approaches
        print("\n" + "="*50)
        print("🔬 PHASE 2: RESEARCH & STATE OF THE ART")
        print("="*50)
        response = text_gen.chat_completion(
            user_prompt="Summarize in one very brief paragraph the latest in AI music-to-image translation.",
            system_prompt="You are an extremely concise researcher. Keep responses under 150 tokens in one paragraph.",
            tool_names=["web_crawl"],
            system_context="Search current info, but return only the most critical findings in one paragraph.",
            max_tokens=150
        )
        print("\n" + "-"*50)
        print("📚 CURRENT RESEARCH:")
        print("-"*50)
        print(f"\n{response}\n")
        
        # Demo 3: Memory System Design
        print("\n" + "="*50)
        print("💾 PHASE 3: MEMORY SYSTEM DESIGN")
        print("="*50)
        structured = text_gen.structured_output(
            user_prompt="""Design an ultra-compact memory system for music-to-image generation with minimal details. Limit your response to bare essentials.""",
            system_prompt="You are a minimalist database architect. Keep responses extremely concise.",
            max_tokens=200
        )
        print("\n" + "-"*50)
        print("📊 MEMORY SYSTEM DESIGN:")
        print("-"*50)
        print(json.dumps(structured, indent=2))
        
        # Demo 4: Iterative Refinement Process
        print("\n" + "="*50)
        print("🔄 PHASE 4: REFINEMENT PIPELINE DESIGN")
        print("="*50)
        try:
            reasoned_response = text_gen.reasoned_completion(
                user_prompt="""Describe in exactly one short paragraph a basic refinement pipeline for music-to-image generation.""",
                reasoning_effort="low",
                max_tokens=150
            )
            print("\n" + "-"*50)
            print("📝 REFINEMENT PIPELINE:")
            print("-"*50)
            print(f"\n{reasoned_response}\n")
        except Exception as e:
            print(f"\n❌ Error in refinement pipeline design: {e}")
        
        # Demo 5: Visual Style Analysis
        print("\n" + "="*50)
        print("👁️ PHASE 5: VISUAL STYLE ANALYSIS")
        print("="*50)
        try:
            # Example of an AI-generated artwork - use a more reliable image URL
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
            vision_response = text_gen.vision_analysis(
                image_url=image_url,
                user_prompt="""In exactly one short paragraph, analyze how this image's visual elements might relate to music.""",
                system_prompt="You are a brief art critic. Limit response to 100 tokens in one paragraph.",
                max_tokens=100
            )
            print("\n" + "-"*50)
            print("🎭 VISUAL ANALYSIS:")
            print("-"*50)
            print(f"\n{vision_response}\n")
        except Exception as e:
            print(f"\n❌ Error in visual analysis: {e}")
        
        # Demo 6: Generate Test Prompts
        print("\n" + "="*50)
        print("✍️ PHASE 6: PROMPT ENGINEERING")
        print("="*50)
        
        test_songs = [
            "Beethoven's Moonlight Sonata",
            "Pink Floyd - Time",
            "Miles Davis - So What"
        ]
        
        for song in test_songs:
            print("\n" + "-"*40)
            print(f"🎵 PROMPT FOR: {song}")
            print("-"*40)
            
            response = text_gen.chat_completion(
                user_prompt=f"""Write a one-sentence image generation prompt for {song}.""",
                system_prompt="You are a minimalist prompt engineer. Limit to ONE sentence of 50 tokens max.",
                max_tokens=50
            )
            print(f"\n{response}\n")
        
        # Demo 7: Save Project Memory
        print("\n" + "="*50)
        print("💾 PHASE 7: SAVING PROJECT MEMORY")
        print("="*50)
        
        history_path = text_gen.save_history()
        print(f"📁 Project saved to: {history_path}")
        
        print("\n" + "="*60)
        print("✅ COMPLETE ✅")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("="*60 + "\n") 