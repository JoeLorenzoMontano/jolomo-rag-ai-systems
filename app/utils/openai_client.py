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
            # Set up API call parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Enable tools/functions if function_responses are provided
            if function_responses:
                # Create tool definitions for each function_response
                tools = []
                for function_name in function_responses.keys():
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": function_name,
                            # Add a simple schema - OpenAI requires at least a name and description
                            "description": f"Function: {function_name}",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    })
                
                # Add tools to API parameters if any were defined
                if tools:
                    params["tools"] = tools
                    params["tool_choice"] = "auto"  # Let model decide when to call functions
                    self.logger.info(f"Enabled {len(tools)} tools for OpenAI API call")
            
            # Make the API call
            response = self.openai.chat.completions.create(**params)
            
            # Get the message from the response
            message = response.choices[0].message
            
            # Check for tool_calls first (newer API versions)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Process the first function call we find
                for tool_call in message.tool_calls:
                    if tool_call.type == "function":
                        function_name = tool_call.function.name
                        function_args = tool_call.function.arguments
                        
                        self.logger.info(f"Tool call detected: {function_name}")
                        
                        # If we have a predefined response for this function, return it
                        if function_responses and function_name in function_responses:
                            self.logger.info(f"Returning predefined response for function: {function_name}")
                            return function_responses[function_name]
                
                # No matching function response found, return content
                return message.content or "[Tool calls were made but no predefined responses were found]"
            
            # Check for the older function_call property
            elif hasattr(message, 'function_call') and message.function_call:
                function_name = message.function_call.name
                function_args = message.function_call.arguments
                
                self.logger.info(f"Function call detected: {function_name}")
                
                # If we have a predefined response for this function, return it
                if function_responses and function_name in function_responses:
                    self.logger.info(f"Returning predefined response for function: {function_name}")
                    return function_responses[function_name]
                else:
                    # If no predefined response is available, return the original content
                    return message.content or f"[Function call to {function_name} would happen here]"
            
            # If no function/tool call, just return the content
            return message.content
        except Exception as e:
            self.logger.error(f"Error in OpenAI chat completion: {e}")
            raise ValueError(f"OpenAI API error: {str(e)}")
            
    def create_thread_with_assistant(self,
                                    messages: List[Dict[str, str]],
                                    assistant_id: str,
                                    additional_messages: Optional[List[Dict[str, str]]] = None,
                                    function_responses: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a thread and run it with an OpenAI Assistant.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            assistant_id: ID of the OpenAI Assistant to use
            additional_messages: Optional additional messages to include (e.g., context, web search results)
            function_responses: Dictionary mapping function names to predefined responses
            
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
            
            while run_status in ["queued", "in_progress", "cancelling", "requires_action"] and attempts < max_attempts:
                time.sleep(1)
                run = self.openai.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                run_status = run.status
                
                # Handle requires_action status when function calls are needed
                if run_status == "requires_action":
                    self.logger.info("Assistant requires function execution")
                    
                    # Check if required actions are tool calls
                    if hasattr(run, 'required_action') and run.required_action.type == "submit_tool_outputs":
                        tool_outputs = []
                        
                        # Process each tool call
                        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                            if tool_call.type != "function":
                                continue
                                
                            name = tool_call.function.name
                            try:
                                import json
                                args = json.loads(tool_call.function.arguments)
                            except:
                                args = {}
                                
                            self.logger.info(f"Processing function call: {name} with args: {args}")
                            
                            # Use our predefined response if available
                            if function_responses and name in function_responses:
                                result = function_responses[name]
                                self.logger.info(f"Using predefined response for function: {name}")
                            else:
                                # Default response if no predefined response available
                                result = f"Function {name} could not be executed"
                                
                            # Add the result to tool outputs
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": str(result)
                            })
                        
                        # Submit the tool outputs back to the assistant
                        if tool_outputs:
                            self.logger.info(f"Submitting tool outputs: {tool_outputs}")
                            run = self.openai.beta.threads.runs.submit_tool_outputs(
                                thread_id=thread.id,
                                run_id=run.id,
                                tool_outputs=tool_outputs
                            )
                            # Reset the poll counter since we've taken action
                            attempts = 0
                        else:
                            # If we can't handle the action, increment attempts
                            attempts += 1
                else:
                    attempts += 1
            
            # Get the assistant's messages
            messages = self.openai.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Get the last assistant message
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            if assistant_messages:
                latest_message = assistant_messages[0]
                response_text = latest_message.content[0].text.value
                
                # Check if any function responses should be inserted
                if function_responses:
                    for function_name, function_response in function_responses.items():
                        if function_name in response_text:
                            self.logger.info(f"Function response applied for {function_name} in Assistant API")
                            return function_response
                
                return response_text
            else:
                return "The assistant did not provide a response."
                
        except Exception as e:
            self.logger.error(f"Error in OpenAI Assistant API: {e}")
            raise ValueError(f"OpenAI Assistant API error: {str(e)}")