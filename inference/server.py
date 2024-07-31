from contextlib import asynccontextmanager
import fastapi
from .loading import load_model, ML, get_input_template
from .structs import Prediction, PredictionInput, ChatInput, ChatResponse
from unsloth import FastLanguageModel
from .defaults import DEFAULT_MAX_NEW_TOKENS


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    model_dir = "/repository"
    app = load_model(app=app, model_dir=model_dir)
    yield

app = fastapi.FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return "Running"

@app.post("/chat")
def chat(chat_input: ChatInput, request: fastapi.Request) -> ChatResponse:
    ml: ML = request.app.ml
    FastLanguageModel.for_inference(ml.model)

    #apply chat template to messages
    tokenized_messages = ml.tokenizer.apply_chat_template(chat_input.messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # Generate model response
    max_tokens = chat_input.config.max_tokens if chat_input.config else DEFAULT_MAX_NEW_TOKENS
    gen_tokens = ml.model.generate(tokenized_messages, max_new_tokens=max_tokens, use_cache=True)

    # Extract outputs
    prediction = ml.tokenizer.batch_decode(gen_tokens, clean_up_tokenization_spaces=True)[0]

    #Post processing
    prompt = ml.tokenizer.apply_chat_template(chat_input.messages, tokenize=False, add_generation_prompt=True)
    response_only = prediction.replace(prompt, "").replace(ml.tokenizer.eos_token, "")

    assistant_msg = [{
        "role": "assistant",
        "content": response_only,
    }]

    all_messages = list(chat_input.messages) + assistant_msg
    return ChatResponse(last_message=response_only, messages=all_messages)

@app.post("/predict")
def prediction(prediction_input: PredictionInput, request: fastapi.Request) -> Prediction:
    ml: ML = request.app.ml
    FastLanguageModel.for_inference(ml.model)
    prompt_template = get_input_template(prediction_input.task_type)

    #validate prediction input contains required fields in template minus answer field
    required_fields = [x for x in prompt_template.input_variables if x != prompt_template.answer_column]
    if set(prediction_input.input.keys()) != set(required_fields):
        raise fastapi.HTTPException(status_code=400, detail=f"Input is missing required fields {required_fields}")

    # Run inference
    prepared_input = {k: v for k, v in prediction_input.input.items() if k in prompt_template.input_variables}
    prepared_input[prompt_template.answer_column] = "" # Set the answer field to an empty string
    inputs = ml.tokenizer(
    [
        prompt_template.template.format(**prepared_input)
    ], return_tensors = "pt").to("cuda")

    max_tokens = prediction_input.config.max_tokens if prediction_input.config else DEFAULT_MAX_NEW_TOKENS
    gen_tokens = ml.model.generate(**inputs, max_new_tokens = max_tokens, use_cache = True)

    # Extract only the answer from the generated tokens
    prediction = ml.tokenizer.batch_decode(gen_tokens[:, inputs.input_ids.shape[1]:])[0]
    response_only = prediction.replace(ml.tokenizer.eos_token, "")

    return Prediction(response = response_only)