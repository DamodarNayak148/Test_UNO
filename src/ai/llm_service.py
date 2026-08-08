import json
import random
from typing import Dict, Any

class LLMService:
    """LLM Service orchestrating Game Master AI commentary and evaluation."""

    def __init__(self, provider: str = "mock", api_key: str = "", model_name: str = "gemini-2.5-flash"):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name

    def generate_gm_response(self, personality: Dict[str, Any], context: str, vision_telemetry: Dict[str, Any]) -> str:
        """Generate GM commentary based on personality, game context, and vision telemetry."""
        if self.provider == "gemini" and self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                prompt = (
                    f"You are {personality.get('name')}, a game master with personality style '{personality.get('style')}'.\n"
                    f"Game Context: {context}\n"
                    f"Vision Telemetry: {json.dumps(vision_telemetry)}\n"
                    f"Respond in 1-2 dramatic sentences as the Game Master!"
                )
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"[LLMService] Gemini call failed: {e}. Falling back to mock engine.")

        # Default Mock LLM engine (Deterministic & responsive for testing)
        return self._generate_mock_response(personality, context, vision_telemetry)

    def _generate_mock_response(self, personality: Dict[str, Any], context: str, vision_telemetry: Dict[str, Any]) -> str:
        name = personality.get("name", "Game Master")
        style = personality.get("style", "")
        
        success = vision_telemetry.get("hands_raised", False) or vision_telemetry.get("has_colorful_item", False) or vision_telemetry.get("face_detected", False)

        if "pose" in context.lower():
            if success:
                phrases = personality.get("success_phrases", ["Magnificent pose! You have proven your worth!"])
                return f"{name}: {random.choice(phrases)}"
            else:
                phrases = personality.get("failure_phrases", ["What kind of pose was that? Fails miserably!"])
                return f"{name}: {random.choice(phrases)}"
        elif "item" in context.lower():
            if vision_telemetry.get("has_colorful_item", False):
                return f"{name}: Ah! I see the vibrant artifact in your hand! A worthy offering!"
            else:
                return f"{name}: I see nothing worthy in front of my lens! Try presenting something brighter!"
        
        # General dialogue
        if success:
            return f"{name}: Excellent action! My physical sensors approve of your maneuver!"
        else:
            return f"{name}: Hmmm... an unexpected outcome. The Game Master remains unimpressed!"
