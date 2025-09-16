"""
Azure Bot Service integration for healthcare Q&A.
"""

import asyncio
from typing import Dict, Any, Optional, List
from botbuilder.core import (
    ActivityHandler,
    MessageFactory,
    TurnContext,
    ConversationState,
    UserState,
    MemoryStorage,
)
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes
import torch

from ..training.qlora_trainer import QLoRATrainer
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

class HealthcareBot(ActivityHandler):
    """Healthcare Q&A bot using fine-tuned QLoRA model."""
    
    def __init__(
        self, 
        conversation_state: ConversationState,
        user_state: UserState,
        model_path: Optional[str] = None
    ):
        """Initialize healthcare bot."""
        self.conversation_state = conversation_state
        self.user_state = user_state
        self.config = get_config()
        
        # Initialize model
        self.model_trainer = QLoRATrainer()
        if model_path:
            try:
                self.model_trainer.load_model(model_path)
                logger.info("Healthcare model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model_trainer = None
        else:
            logger.warning("No model path provided, bot will use fallback responses")
            self.model_trainer = None
        
        # Healthcare-specific settings
        self.max_response_length = self.config.get("bot.max_response_length", 256)
        self.confidence_threshold = self.config.get("bot.confidence_threshold", 0.7)
        
        # Predefined responses for common queries
        self.fallback_responses = {
            "greeting": [
                "Hello! I'm a healthcare assistant. How can I help you today?",
                "Hi there! I'm here to help with your healthcare questions.",
                "Welcome! I'm your healthcare assistant. What would you like to know?"
            ],
            "emergency": [
                "If this is a medical emergency, please call emergency services immediately (911 in the US).",
                "For urgent medical concerns, please contact your healthcare provider or emergency services."
            ],
            "disclaimer": [
                "Please note that I'm an AI assistant and cannot replace professional medical advice.",
                "Always consult with a qualified healthcare provider for medical decisions.",
                "This information is for educational purposes only."
            ],
            "default": [
                "I'm here to help with healthcare questions. Could you please provide more details?",
                "I'd be happy to help with your healthcare question. Can you be more specific?",
                "I'm not sure I understand. Could you rephrase your healthcare question?"
            ]
        }
    
    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages."""
        user_message = turn_context.activity.text.strip()
        logger.info(f"Received message: {user_message}")
        
        # Check for emergency keywords
        if self._is_emergency_query(user_message):
            response = self._get_emergency_response()
        # Check for greeting
        elif self._is_greeting(user_message):
            response = self._get_greeting_response()
        # Generate healthcare response
        else:
            response = await self._generate_healthcare_response(user_message)
        
        # Add disclaimer for medical advice
        if self._needs_disclaimer(user_message):
            response += "\n\n" + self._get_disclaimer()
        
        # Send response
        await turn_context.send_activity(MessageFactory.text(response))
        
        # Save conversation state
        await self.conversation_state.save_changes(turn_context)
        await self.user_state.save_changes(turn_context)
    
    async def on_welcome_message(self, turn_context: TurnContext):
        """Send welcome message to new users."""
        welcome_text = (
            "Welcome to the Healthcare Assistant Bot! 🏥\n\n"
            "I can help answer questions about:\n"
            "• General health information\n"
            "• Symptoms and conditions\n"
            "• Medications and treatments\n"
            "• Health maintenance tips\n\n"
            "**Important:** I'm an AI assistant and cannot replace professional medical advice. "
            "Always consult with a qualified healthcare provider for medical decisions.\n\n"
            "How can I help you today?"
        )
        
        await turn_context.send_activity(MessageFactory.text(welcome_text))
    
    async def on_members_added_activity(
        self, members_added: List[ChannelAccount], turn_context: TurnContext
    ):
        """Handle new members joining the conversation."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await self.on_welcome_message(turn_context)
    
    def _is_emergency_query(self, message: str) -> bool:
        """Check if message indicates a medical emergency."""
        emergency_keywords = [
            "emergency", "urgent", "911", "ambulance", "heart attack", 
            "stroke", "bleeding", "chest pain", "can't breathe", 
            "unconscious", "suicide", "overdose", "severe pain"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in emergency_keywords)
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting."""
        greeting_keywords = [
            "hello", "hi", "hey", "good morning", "good afternoon", 
            "good evening", "greetings", "start", "begin"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in greeting_keywords)
    
    def _needs_disclaimer(self, message: str) -> bool:
        """Check if response needs medical disclaimer."""
        medical_keywords = [
            "diagnose", "treatment", "medicine", "medication", "symptom",
            "disease", "condition", "doctor", "should i", "what if"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in medical_keywords)
    
    def _get_emergency_response(self) -> str:
        """Get emergency response."""
        import random
        return random.choice(self.fallback_responses["emergency"])
    
    def _get_greeting_response(self) -> str:
        """Get greeting response."""
        import random
        return random.choice(self.fallback_responses["greeting"])
    
    def _get_disclaimer(self) -> str:
        """Get medical disclaimer."""
        import random
        return random.choice(self.fallback_responses["disclaimer"])
    
    async def _generate_healthcare_response(self, user_message: str) -> str:
        """Generate response using the fine-tuned model."""
        if self.model_trainer is None:
            import random
            return random.choice(self.fallback_responses["default"])
        
        try:
            # Format the prompt for the model
            instruction = "You are a helpful healthcare assistant. Please provide accurate and helpful medical information."
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_message}\n\n### Response:\n"
            
            # Generate response
            response = self.model_trainer.generate_response(
                prompt, 
                max_new_tokens=self.max_response_length
            )
            
            # Clean up response
            response = self._clean_response(response)
            
            # Validate response quality
            if self._is_valid_response(response):
                return response
            else:
                logger.warning("Generated response failed validation, using fallback")
                import random
                return random.choice(self.fallback_responses["default"])
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import random
            return random.choice(self.fallback_responses["default"])
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the model response."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive patterns
        lines = response.split('\n')
        unique_lines = []
        for line in lines:
            if line.strip() and line not in unique_lines:
                unique_lines.append(line)
        
        response = '\n'.join(unique_lines)
        
        # Limit length
        if len(response) > self.max_response_length * 2:
            response = response[:self.max_response_length * 2] + "..."
        
        return response
    
    def _is_valid_response(self, response: str) -> bool:
        """Validate response quality."""
        if not response or len(response.strip()) < 10:
            return False
        
        # Check for inappropriate content (basic filtering)
        inappropriate_keywords = [
            "i don't know", "i can't help", "error", "undefined",
            "null", "none", "###", "instruction:", "input:", "response:"
        ]
        
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in inappropriate_keywords):
            return False
        
        return True

class HealthcareBotManager:
    """Manager for healthcare bot deployment and operations."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize bot manager."""
        self.config = get_config()
        self.model_path = model_path
        
        # Setup bot state
        memory_storage = MemoryStorage()
        self.conversation_state = ConversationState(memory_storage)
        self.user_state = UserState(memory_storage)
        
        # Create bot instance
        self.bot = HealthcareBot(
            self.conversation_state,
            self.user_state,
            model_path
        )
    
    def get_bot(self) -> HealthcareBot:
        """Get bot instance."""
        return self.bot
    
    async def process_message(self, activity: Activity) -> Optional[Activity]:
        """Process incoming message activity."""
        try:
            # Create turn context
            from botbuilder.core import TurnContext
            
            # This is a simplified version - in production, you'd use proper adapter
            turn_context = TurnContext(None, activity)
            
            # Process the message
            if activity.type == ActivityTypes.message:
                await self.bot.on_message_activity(turn_context)
            elif activity.type == ActivityTypes.conversation_update:
                await self.bot.on_members_added_activity(
                    activity.members_added, turn_context
                )
            
            return turn_context.activity
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None

# Azure Bot Service deployment utilities
def create_bot_app():
    """Create bot application for Azure deployment."""
    from aiohttp import web
    from aiohttp.web import Request, Response, json_response
    import json
    
    # Initialize bot manager
    bot_manager = HealthcareBotManager()
    
    async def messages(req: Request) -> Response:
        """Handle incoming bot messages."""
        if "application/json" in req.headers["Content-Type"]:
            body = await req.json()
        else:
            return Response(status=415)
        
        # Create activity from request
        activity = Activity().deserialize(body)
        
        # Process message
        response_activity = await bot_manager.process_message(activity)
        
        if response_activity:
            return json_response(
                data=response_activity.serialize(),
                status=200
            )
        else:
            return Response(status=500)
    
    # Create web application
    app = web.Application()
    app.router.add_post("/api/messages", messages)
    
    return app

def deploy_to_azure():
    """Deploy bot to Azure Bot Service."""
    # This would contain Azure deployment logic
    # For now, we'll just log the configuration
    config = get_config()
    
    logger.info("Azure Bot Service Configuration:")
    logger.info(f"Bot App ID: {config.get('bot.app_id', 'Not configured')}")
    logger.info(f"Bot Endpoint: {config.get('bot.endpoint', 'Not configured')}")
    
    # In production, this would use Azure CLI or Azure SDK to deploy
    logger.info("Bot deployment configuration ready")
    logger.info("Use Azure CLI or Azure Portal to complete deployment")