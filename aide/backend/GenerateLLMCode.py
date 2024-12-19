from .utils import FunctionSpec, OutputType
from . import backend_openai, backend_gemini, backend_llama


class GenerateLLMCode:

    @staticmethod
    def generate_llm_code(
            system_message: str | None,
            user_message: str | None,
            func_spec: FunctionSpec | None = None,
            **model_kwargs,
    ) -> tuple[OutputType, float, int, int, dict]:

        from ..utils.config import _llm_platform, _OPENAI, _META, _GOOGLE

        if _llm_platform is None:
            raise Exception("Select a LLM Platform: OpenAI (GPT) or Meta (Lama)")
        elif _llm_platform == _OPENAI:
            output, req_time, in_tok_count, out_tok_count, info = backend_openai.query(
                system_message=system_message, user_message=user_message, func_spec=func_spec, **model_kwargs,)
            return output, req_time, in_tok_count, out_tok_count, info
        elif _llm_platform == _META:
            output, req_time, in_tok_count, out_tok_count, info = backend_llama.query(
                system_message=system_message, user_message=user_message, func_spec=func_spec, **model_kwargs, )
            return output, req_time, in_tok_count, out_tok_count, info
        elif _llm_platform == _GOOGLE:
            output, req_time, in_tok_count, out_tok_count, info = backend_gemini.query(
                system_message=system_message, user_message=user_message, func_spec=func_spec, **model_kwargs, )
            return output, req_time, in_tok_count, out_tok_count, info

        else:
            raise Exception(f"Model {_llm_platform} is not implemented yet!")

    @staticmethod
    def refine_source_code(code: str):
        final_code = []
        for line in code.splitlines():
            if not line.startswith('#'):
                final_code.append(line)
        final_code = "\n".join(final_code)
        return final_code.replace("@ ```", "# ```")
