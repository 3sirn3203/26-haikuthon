"""vLLM client with standard-library HTTP and whitespace-flexible JSON grammars.

Importing this module performs no network requests. Construct VLLMClient only
when an already-running server is available; this module never starts one.
There is exactly one HTTP attempt per LLM request. xgrammar is imported lazily
to build the grammars; it is already included in the pinned vLLM environment.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import default_config


PROMPT_SCHEMA = {'type': 'object', 'properties': {'prompt_text': {'type': 'string'}},
                 'required': ['prompt_text'], 'additionalProperties': False}
CRITIC_SCHEMA = {'type': 'object', 'properties': {
    'issues': {'type': 'array', 'items': {'type': 'string'}}},
    'required': ['issues'], 'additionalProperties': False}


@lru_cache(maxsize=8)
def json_grammar(schema_json):
    # A request-level grammar preserves JSON constraints while allowing ordinary
    # whitespace, including on an external server with compact JSON defaults.
    # Import lazily: importing Stage2 must not initialize inference dependencies.
    import xgrammar
    return str(xgrammar.Grammar.from_json_schema(schema_json, any_whitespace=True))


class VLLMClient:
    def __init__(self, config=None):
        config = default_config() if config is None else config
        self.url = config['server_url'].rstrip('/')
        self.model = config['model']['id']
        self.enable_thinking = config['model']['enable_thinking']
        self.calls = []
        with urlopen(self.url + '/v1/models', timeout=10) as response:
            models = json.load(response)
        if self.model not in {model['id'] for model in models['data']}:
            raise ValueError(f'Requested model is not served: {self.model}')
        with urlopen(self.url + '/version', timeout=10) as response:
            self.server_version = json.load(response)
        for schema in [PROMPT_SCHEMA, CRITIC_SCHEMA]:
            json_grammar(json.dumps(schema))

    def _request(self, conversation, settings, seed):
        sampled = settings.get('do_sample', True)
        thinking = settings.get('enable_thinking', self.enable_thinking)
        schema = settings.get('json_schema', PROMPT_SCHEMA if sampled else CRITIC_SCHEMA)
        body = {
            'model': self.model,
            'messages': conversation,
            'max_tokens': settings['max_new_tokens'],
            'temperature': settings.get('temperature', 0.0) if sampled else 0.0,
            'seed': seed,
            'chat_template_kwargs': {'enable_thinking': thinking},
            'structured_outputs': {'grammar': json_grammar(json.dumps(schema))},
        }
        if thinking and settings.get('reasoning_effort'):
            body['reasoning_effort'] = settings['reasoning_effort']
        if sampled:
            for key in ['top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty']:
                if key in settings:
                    body[key] = settings[key]
        request = Request(
            self.url + '/v1/chat/completions',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        )
        # Never resend: an ambiguous timeout may already have run inference.
        try:
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
        except HTTPError as exc:
            message = exc.read().decode(errors='replace')[:1200]
            raise RuntimeError(f'vLLM HTTP {exc.code}: {message}') from exc
        choice = result['choices'][0]
        return choice['message']['content'] or '', result.get('usage', {}), choice['finish_reason']

    def generate(self, conversations, settings, seed):
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=settings['batch_size']) as pool:
            futures = [
                pool.submit(self._request, messages, settings, seed + index)
                for index, messages in enumerate(conversations)
            ]
            responses = [future.result() for future in futures]
        elapsed = time.monotonic() - start
        tokens = sum(usage.get('completion_tokens', 0) for _, usage, _ in responses)
        self.calls.append({
            'batch_size': len(conversations),
            'seconds': round(elapsed, 3),
            'sampled': settings.get('do_sample', True),
            'request_seeds': [seed + index for index in range(len(conversations))],
            'completion_tokens': tokens,
            'completion_tokens_per_second': round(tokens / elapsed, 2),
            'finish_reasons': [reason for _, _, reason in responses],
        })
        return [text for text, _, _ in responses]
