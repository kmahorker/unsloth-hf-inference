from typing import Any

import fastapi
from pydantic import BaseModel
from unsloth import FastLanguageModel
from .defaults import default_prompt_templates
from .structs import PromptTemplate

class ML(BaseModel):
    model: Any
    tokenizer: Any


def load_model(app: fastapi.FastAPI, model_dir: str) -> fastapi.FastAPI:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    ml = ML(model=model, tokenizer=tokenizer)
    app.ml = ml
    return app

def get_input_template(task_type: str) -> PromptTemplate:
    return default_prompt_templates[task_type]