from typing import Dict, List, Optional
import logging
import os
import threading
import time

from llama_index.llms.openai import OpenAI as OpenAI_LLAMA
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import PromptTemplate
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool

# Set environment variable for tiktoken to use /tmp which is writable in Vercel
os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"
os.makedirs("/tmp/tiktoken_cache", exist_ok=True)

from app.templates.prompt_templates import AIOFFICER_SYSTEM_TEMPLATE
from app.tools.search.aiofficer_semantic_search_tool import AIOfficerSemanticSearchTool
from app.config.env_config import config

# Check if running in serverless environment
IS_SERVERLESS = os.environ.get("VERCEL") == "1" or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

class AgentAIOfficer:
    """
    AI-Officer agent for answering queries.
    This agent uses semantic search to provide accurate information.
    Uses lazy initialization and state tracking to avoid first-call failures.
    """
    
    # Class-level variables to track initialization state
    _instance = None
    _initialization_lock = threading.Lock()
    _is_initializing = False
    _is_initialized = False
    _initialization_start_time = 0
    _max_init_wait_time = 10  # Reduced from 30 to 10 seconds for serverless
    
    def __new__(cls):
        # Implement singleton pattern to ensure only one agent instance
        if cls._instance is None:
            cls._instance = super(AgentAIOfficer, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.qa_template = PromptTemplate(AIOFFICER_SYSTEM_TEMPLATE)
            cls._instance.gpt4_llm = None
            cls._instance.agent = None
            
            # In serverless environments, initialize immediately in the main thread
            if IS_SERVERLESS:
                cls._instance.logger.info("Serverless environment detected, initializing agent immediately")
                cls._instance._initialize_agent()
            else:
                # Start initialization in the background for non-serverless environments
                threading.Thread(target=cls._instance._initialize_agent, daemon=True).start()
                
        return cls._instance
    
    def _initialize_agent(self):
        """Initialize the agent in the background or immediately."""
        with self._initialization_lock:
            if self._is_initializing or self._is_initialized:
                return
                
            self._is_initializing = True
            self._initialization_start_time = time.time()
            self.logger.info("Starting agent initialization")
            
        try:
            # Initialize the LLM with a timeout for API calls
            self.gpt4_llm = OpenAI_LLAMA(
                model=config.llm_model,
                timeout=20  # Add timeout for API calls
            )
            
            # Create semantic search tool
            aiofficer_semantic_search_tool = AIOfficerSemanticSearchTool()
            
            # Create llama_index FunctionTool object
            aiofficer_semantic_search_function_tool = FunctionTool.from_defaults(
                name=aiofficer_semantic_search_tool.name,
                description=aiofficer_semantic_search_tool.description,
                fn=aiofficer_semantic_search_tool.__call__
            )
            
            try:
                # Set up memory with configurable token limit
                memory = ChatMemoryBuffer.from_defaults(token_limit=config.memory_token_limit)
            except Exception as e:
                self.logger.warning(f"Error setting up chat memory with token limit: {e}")
                # Fall back to a simpler memory implementation without tokenization
                memory = ChatMemoryBuffer(token_limit=100000)
            
            # Initialize agent with tools
            self.agent = OpenAIAgent.from_tools(
                tools=[aiofficer_semantic_search_function_tool],
                llm=self.gpt4_llm,
                memory=memory,
                verbose=True,
                system_prompt=AIOFFICER_SYSTEM_TEMPLATE
            )
            
            with self._initialization_lock:
                self._is_initialized = True
                self._is_initializing = False
                
            init_time = time.time() - self._initialization_start_time
            self.logger.info(f"Agent initialization completed successfully in {init_time:.2f} seconds")
            
        except Exception as e:
            with self._initialization_lock:
                self._is_initializing = False
            self.logger.error(f"Error during agent initialization: {e}")
            
            # In serverless, attempt to reinitialize immediately with simpler config
            if IS_SERVERLESS:
                self.logger.info("Attempting simplified initialization for serverless environment")
                self._initialize_simple_agent()
    
    def _initialize_simple_agent(self):
        """Initialize a simplified agent without complex tools for fallback."""
        try:
            # Initialize the LLM with a timeout for API calls
            self.gpt4_llm = OpenAI_LLAMA(
                model=config.llm_model,
                timeout=20  # Add timeout for API calls
            )
            
            # Create semantic search tool
            aiofficer_semantic_search_tool = AIOfficerSemanticSearchTool()
            
            # Create llama_index FunctionTool object
            aiofficer_semantic_search_function_tool = FunctionTool.from_defaults(
                name=aiofficer_semantic_search_tool.name,
                description=aiofficer_semantic_search_tool.description,
                fn=aiofficer_semantic_search_tool.__call__
            )
            try:
                # Set up memory with configurable token limit
                memory = ChatMemoryBuffer.from_defaults(token_limit=config.memory_token_limit)
            except Exception as e:
                self.logger.warning(f"Error setting up chat memory with token limit: {e}")
                # Fall back to a simpler memory implementation without tokenization
                memory = ChatMemoryBuffer(token_limit=100000)
            

            # Initialize agent with tools
            self.agent = OpenAIAgent.from_tools(
                tools=[aiofficer_semantic_search_function_tool],
                llm=self.gpt4_llm,
                memory=memory,
                verbose=True,
                system_prompt=AIOFFICER_SYSTEM_TEMPLATE
            )
            
            with self._initialization_lock:
                self._is_initialized = True
                self._is_initializing = False
                
            self.logger.info("Simplified agent initialized as fallback")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simplified agent: {e}")
            # If we can't even initialize a simple agent, we'll have to use direct responses
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to process queries."""
        return self._is_initialized and self.agent is not None
    
    def initialization_status(self) -> Dict[str, any]:
        """Get the current initialization status."""
        if self._is_initialized:
            return {"status": "ready", "message": "Agent is initialized and ready"}
        
        if self._is_initializing:
            elapsed = time.time() - self._initialization_start_time
            return {
                "status": "initializing", 
                "message": f"Agent is initializing (elapsed: {elapsed:.1f}s)",
                "elapsed_seconds": elapsed
            }
            
        return {"status": "not_started", "message": "Agent initialization has not started"}
    
    def agent_query(self, query: str) -> str:
        """
        Query the agent with a user question.
        
        Args:
            query: The user's question.
            
        Returns:
            The agent's response.
        """
        # Check if agent is still initializing
        if not self._is_initialized:
            # If in serverless and still initializing after timeout, provide a direct response
            if IS_SERVERLESS and time.time() - self._initialization_start_time > self._max_init_wait_time:
                self.logger.warning("Serverless initialization timed out, providing direct response")
                return self._direct_response(query)
                
            # If initialization is taking too long, we should retry
            if self._is_initializing:
                elapsed = time.time() - self._initialization_start_time
                if elapsed < self._max_init_wait_time:
                    # Still within acceptable wait time
                    self.logger.info(f"Agent still initializing, waited {elapsed:.1f}s")
                    message = "The chatbot is still initializing, please try again in a few seconds."
                    return f"{message} (Elapsed: {elapsed:.1f}s)"
                else:
                    # Initialization is taking too long, try to restart it
                    self.logger.warning(f"Agent initialization timed out after {elapsed:.1f}s, attempting restart")
                    with self._initialization_lock:
                        self._is_initializing = False
                    # Restart initialization
                    if IS_SERVERLESS:
                        self._initialize_simple_agent()  # Use simpler initialization for serverless
                    else:
                        threading.Thread(target=self._initialize_agent, daemon=True).start()
            else:
                # Not initialized and not initializing, start initialization
                self.logger.info("Agent not initialized, starting initialization")
                if IS_SERVERLESS:
                    self._initialize_agent()  # In serverless, initialize immediately
                else:
                    threading.Thread(target=self._initialize_agent, daemon=True).start()
            
            # In serverless, return a direct response if initialization is still pending
            if IS_SERVERLESS:
                return self._direct_response(query)
            
            # Otherwise return a message indicating the agent is initializing
            return "I apologize, the AI-Officer is still initializing. Please try again in a few seconds."
        
        # Agent is initialized, process the query
        try:
            response = self.agent.chat(query)
            return str(response)
        except Exception as e:
            self.logger.error(f"Error querying agent: {e}")
            # Return a fallback response in case of an error
            return self._direct_response(query)
    
    def _direct_response(self, query: str) -> str:
        """Provide a direct response without using the agent for serverless environments."""
        # Basic answers to common questions
        common_responses = {
            "what": "The AI Officer is a professional who knows how to orchestrate AI tools and strategies to help organizations thrive in an AI-driven world.",
            "who": "The AI Officer Institute is built for ambitious professionals across industries including Product Managers, AI Engineers, Freelancers, Consultants, and industry professionals.",
            "how": "The AI Officer Institute offers practical application of AI through real client work, a global network, practical tools and templates, and meaningful certification.",
            "where": "You can find more information at the AI Officer Institute website or join our community through Slack or Discord channels.",
            "when": "The AI Officer Institute offers regular 'AI Days' to learn, demo, and build together, both in-person (starting in Vietnam) and online.",
            "why": "Becoming an AI Officer helps you future-proof your career by learning how to apply AI tools to real-world business challenges.",
            "cost": "The AI Officer Institute offers a Free Tier to join AI Days and access foundational resources, as well as a Pro Membership ($48/month) for premium features."
        }
        
        # Find the most relevant common response
        for key, response in common_responses.items():
            if key.lower() in query.lower():
                return response
                
        # Default response if no matches
        return "I'm here to help with information about the AI Officer Institute. Please ask about what an AI Officer is, who the institute is for, our certification, community, cost, or other aspects of our program." 