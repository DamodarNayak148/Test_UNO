from enum import Enum, auto

class GameState(Enum):
    IDLE = auto()
    INTRO = auto()
    PROMPT_CHALLENGE = auto()
    WAITING_PLAYER_ACTION = auto()
    SCANNING_VISION = auto()
    EVALUATING = auto()
    REACTION = auto()
    GAME_OVER = auto()
