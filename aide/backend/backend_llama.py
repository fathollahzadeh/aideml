"""Backend for OpenAI API."""

import logging
import time
from groq import Groq
import re
import tiktoken

from .utils import FunctionSpec, OutputType

logger = logging.getLogger("aide")
client: None


def _setup_llama_client():
    global client
    from ..utils.config import _LLM_API_Key
    _, api_key = _LLM_API_Key.get_API_Key()
    client = Groq(api_key=api_key)


def query(
        system_message: str | None,
        user_message: str | None,
        func_spec: FunctionSpec | None = None,
        **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_llama_client()
    time_start = time.time()
    output, req_time, in_tokens, out_tokens, info = __submit_Request_LLaMa_LLM(user_message=user_message,
                                                                               system_message=system_message,
                                                                               client=client)
    time_end = time.time()
    wait_time = time_end - time_start - req_time
    return output, wait_time, in_tokens, out_tokens, info


def __submit_Request_LLaMa_LLM(user_message: str, system_message: str, client) -> tuple[
    OutputType, float, int, int, dict]:
    from ..utils.config import _llm_model, _temperature

    if system_message is not None and user_message is None:
        messages = [{"role": "user", "content": system_message}]
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    try:
        time_start = time.time()
        completion = client.chat.completions.create(
            model=_llm_model,
            messages=messages,
            temperature=_temperature
        )
        content = completion.choices[0].message.content
        content = __refine_text(content)
        codes = []
        code_blocks = __match_code_blocks(content)
        if len(code_blocks) > 0:
            for code in code_blocks:
                codes.append(code)

            code = "\n".join(codes)
        else:
            code = content
        in_tokens = get_number_tokens(user_message=user_message, system_message=system_message)
        out_tokens = get_number_tokens(user_message=code, system_message="")

        return code, time.time() - time_start, in_tokens, out_tokens, dict()
    except Exception as ee:
        _setup_llama_client()
        return __submit_Request_LLaMa_LLM(user_message=user_message, system_message=system_message, client=client)


def __match_code_blocks(text):
    pattern = re.compile(r'```(?:python)?[\n\r](.*?)```', re.DOTALL)
    return pattern.findall(text)


def __refine_text(text):
    ind1 = text.find('\n')
    ind2 = text.rfind('\n')

    begin_txt = text[0: ind1]
    end_text = text[ind2 + 1:len(text)]
    begin_index = 0
    end_index = len(text)
    if begin_txt == "<CODE>":
        begin_index = ind1 + 1

    if end_text == "</CODE>":
        end_index = ind2
    text = text[begin_index:end_index]
    text = text.replace("<CODE>", "# <CODE>")
    text = text.replace("</CODE>", "# </CODE>")
    text = text.replace("```", "@ ```")

    from .GenerateLLMCode import GenerateLLMCode
    text = GenerateLLMCode.refine_source_code(code=text)
    return text


def get_number_tokens(user_message: str, system_message: str):
    enc = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    sm = system_message
    um = user_message
    if system_message is None:
        sm = ""
    if user_message is None:
        um = ""
    token_integers = enc.encode(sm + um)
    num_tokens = len(token_integers)
    return num_tokens
