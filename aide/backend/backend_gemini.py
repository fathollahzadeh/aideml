"""Backend for OpenAI API."""

import logging
import time
import google.generativeai as genai
from .utils import FunctionSpec, OutputType


logger = logging.getLogger("aide")


def _setup_gemini_client():
    from ..utils.config import _LLM_API_Key
    _, api_key = _LLM_API_Key.get_API_Key()
    genai.configure(api_key=api_key)


def query(
        system_message: str | None,
        user_message: str | None,
        func_spec: FunctionSpec | None = None,
        **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_gemini_client()
    filtered_kwargs = dict()
    time_start = time.time()
    output, req_time, in_tokens, out_tokens, info = __submit_Request_Gemini_LLM(user_message=user_message, system_message=system_message, **filtered_kwargs)
    time_end = time.time()
    wait_time = time_end - time_start - req_time
    return output, wait_time, in_tokens, out_tokens, info


def __submit_Request_Gemini_LLM(user_message: str, system_message: str, **kwargs) -> tuple[OutputType, float, int, int, dict]:
    from ..utils.config import _llm_model, _temperature, _top_p, _top_k, _max_out_token_limit

    time_start = time.time()
    generation_config = {
        "temperature": _temperature,
        "top_p": _top_p,
        "top_k": _top_k,
        "max_output_tokens": _max_out_token_limit,
    }
    if user_message is None and system_message is not None:
        user_message, system_message = system_message, user_message

    model = genai.GenerativeModel(model_name=_llm_model,
                                  generation_config=generation_config,
                                  # safety_settings=safety_settings,
                                  system_instruction=system_message,
                                  **kwargs,
                                  )
    try:
        prompt = []
        if system_message is not None:
            prompt.append(system_message)
        if user_message is not None:
            prompt.append(user_message)

        message = "\n".join(prompt)
        in_tokens = model.count_tokens(message).total_tokens

        chat_session = model.start_chat(
            history=[{
                "role": "user",
                "parts": [user_message],
            }])

        response = chat_session.send_message("INSERT_INPUT_HERE")
        code = response.text
        out_tokens = model.count_tokens(code).total_tokens
        time_end = time.time()
        return code, time_end - time_start, in_tokens, out_tokens, dict()

    except Exception as err:
        _setup_gemini_client()
        return __submit_Request_Gemini_LLM(user_message=user_message, system_message=system_message)
