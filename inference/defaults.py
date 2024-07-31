from typing import Dict
from .structs import PromptTemplate, TaskType

default_prompt_templates: Dict[TaskType, PromptTemplate] = {
    
    TaskType.GENERIC: PromptTemplate(
        template="""{input}
        
{output}""",
        input_variables=["input", "output"],
        answer_column="output"
    ),
    # Instruct Prompt Template
    TaskType.INSTRUCT: PromptTemplate(
        template ="""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""",
    input_variables=["instruction", "input", "output"],
    answer_column="output"
    ),

    # Question Answer Prompt Template
    TaskType.QA: PromptTemplate(
    template="""You are an AI assistant trained to answer questions about a context. Given a context and a question about it, provide a concise and accurate answer based on the information in the context.

Context:
{context}

Question:
{question}

Answer:
{answer}
""",
    input_variables=["context", "question", "answer"],
    answer_column="answer"
    )
}

DEFAULT_MAX_NEW_TOKENS = 2000