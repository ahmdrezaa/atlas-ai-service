"""Offline OpenAI provider for tests: the real ChatOpenAI + OpenAI SDK run, only HTTP is
intercepted (httpx.MockTransport). Nothing leaves the process."""

import json

import httpx
from langchain_openai import ChatOpenAI


class OfflineOpenAI:
    """Each queued reply is a dict (returned as the JSON message content) or an int
    HTTP status (returned as an error with a private body that must never leak)."""

    def __init__(self, *replies, input_tokens=120, output_tokens=30):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.model_kwargs: list[dict] = []
        self.usage = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, **body})
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            return httpx.Response(
                reply,
                json={"error": {"message": "private provider body", "type": "x"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(reply)},
                    }
                ],
                "usage": {
                    **self.usage,
                    "total_tokens": sum(self.usage.values()),
                },
            },
        )

    def __call__(self, **kwargs):
        """Use as the model factory: records the kwargs the service passes."""
        self.model_kwargs.append(kwargs)
        client = httpx.Client(transport=httpx.MockTransport(self._handle))
        return ChatOpenAI(**kwargs, http_client=client)
