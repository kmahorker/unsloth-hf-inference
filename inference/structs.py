from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Any, Optional

class TaskType(str, Enum):
    QA = 'qa'
    INSTRUCT = 'instruct'
    GENERIC = 'generic'
    CHAT = 'chat'

class PredictionConfig(BaseModel):
    max_tokens: int

class PredictionInput(BaseModel):
    input: Dict[str, Any]
    task_type: TaskType
    config: Optional[PredictionConfig] = None


class Prediction(BaseModel):
    response: str

class ChatResponse(BaseModel):
    last_message: str
    messages: List[Dict[str, Any]]

class ChatConfig(BaseModel):
    max_tokens: int

class ChatInput(BaseModel):
    messages: List[Dict[str, Any]]
    config: Optional[ChatConfig] = None

class PromptTemplate(BaseModel):
    template: str
    input_variables: List[str]
    answer_column: str

