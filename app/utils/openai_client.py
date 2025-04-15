"""
OpenAI client utility.

This module provides functionality for interacting with OpenAI's API for chat completions
and the Assistants API.
"""

import logging
import time
from typing import List, Dict, Any, Optional

class OpenAIClient:
    """Client for interacting with OpenAI APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (will be verified during initialization)
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.is_available = False
        
        # Check if OpenAI is available
        if self.api_key:
            try:
                import openai
                self.openai = openai
                self.openai.api_key = self.api_key
                self.is_available = True
                self.logger.info("OpenAI client initialized successfully")
            except ImportError:
                self.logger.warning("OpenAI module not available")
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}")

    def create_chat_completion(self, 
                              messages: List[Dict[str, str]], 
                              model: str = "gpt-3.5-turbo",
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = 1000,
                              function_responses: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a completion using OpenAI's chat API with function call support.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            model: Model to use for completion
            temperature: Temperature parameter for response generation
            max_tokens: Maximum tokens to generate
            function_responses: Dictionary mapping function names to predefined responses
            
        Returns:
            Generated text response or predefined function response
            
        Raises:
            ValueError: If OpenAI is not available or there's an API error
        """
        if not self.is_available:
            raise ValueError("OpenAI API is not available")
            
        try:
            # Note: The function definitions are not explicitly passed here.
            # We're assuming the model is already trained to call functions based on the conversation.
            response = self.openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Get the message from the response
            message = response.choices[0].message
            
            # Check if a function call was generated
            if hasattr(message, 'function_call') and message.function_call:
                function_name = message.function_call.name
                function_args = message.function_call.arguments
                
                self.logger.info(f"Function call detected: {function_name} with arguments: {function_args}")
                
                # If we have a predefined response for this function, return it
                if function_responses and function_name in function_responses:
                    self.logger.info(f"Returning predefined response for function: {function_name}")
                    return function_responses[function_name]
                else:
                    # If no predefined response is available, return the original content or a placeholder
                    return message.content or f"[Function call to {function_name} would happen here]"
            
            # If no function call, just return the content
            return message.content
        except Exception as e:
            self.logger.error(f"Error in OpenAI chat completion: {e}")
            raise ValueError(f"OpenAI API error: {str(e)}")
            
    def create_thread_with_assistant(self,
                                    messages: List[Dict[str, str]],
                                    assistant_id: str,
                                    additional_messages: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Create a thread and run it with an OpenAI Assistant.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            assistant_id: ID of the OpenAI Assistant to use
            additional_messages: Optional additional messages to include (e.g., context, web search results)
            
        Returns:
            Assistant's response text
            
        Raises:
            ValueError: If OpenAI is not available or there's an API error
        """
        if not self.is_available:
            raise ValueError("OpenAI API is not available")
            
        try:
            # Create a thread
            thread = self.openai.beta.threads.create()
            
            # Add user messages to the thread
            for msg in messages:
                if msg.get('role') == 'user':
                    self.openai.beta.threads.messages.create(
                        thread_id=thread.id,
                        role="user",
                        content=msg.get('content', '')
                    )
            
            # Add any additional messages (context, web search, etc.)
            if additional_messages and len(additional_messages) > 0:
                for msg in additional_messages:
                    self.openai.beta.threads.messages.create(
                        thread_id=thread.id,
                        role=msg.get('role', 'user'),
                        content=msg.get('content', '')
                    )
            
            # Run the assistant
            run = self.openai.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            
            # Poll for completion
            run_status = run.status
            max_attempts = 30  # 30 seconds max wait time
            attempts = 0
            
            while run_status in ["queued", "in_progress", "cancelling"] and attempts < max_attempts:
                time.sleep(1)
                run = self.openai.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                run_status = run.status
                attempts += 1
            
            # Get the assistant's messages
            messages = self.openai.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Get the last assistant message
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            if assistant_messages:
                latest_message = assistant_messages[0]
                return latest_message.content[0].text.value
            else:
                return "The assistant did not provide a response."
                
        except Exception as e:
            self.logger.error(f"Error in OpenAI Assistant API: {e}")
            raise ValueError(f"OpenAI Assistant API error: {str(e)}")