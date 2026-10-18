"""Single-submission native tools, same-task recovery and service-specific terminal checks."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from acedatacloud import AceDataCloud, AsyncAceDataCloud
from llama_index.core.tools.tool_spec.base import BaseToolSpec

API_BASE = "https://api.acedata.cloud"
_PENDING = {"pending", "processing", "running", "queued", "submitted", "streaming", "in_progress"}
_FAILED = {"failed", "failure", "error", "cancelled", "canceled", "rejected"}
_DONE = {"succeeded", "success", "completed", "complete", "finished"}


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain text")
    return value.strip()


def _url(value: str) -> str:
    value = _required(value, "image URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(
            "Use a public HTTP(S) image URL without embedded credentials; local files/data URIs are not accepted here"
        )
    return value


def _submitted(task_id: str) -> dict[str, Any]:
    if not task_id:
        raise RuntimeError("No task ID returned. Outcome unknown: inspect task history, do not automatically resubmit.")
    return {
        "task_id": task_id,
        "status": "pending",
        "success": None,
        "media_urls": [],
        "next_tool": "happyhorse_get_task",
        "polling_interval_seconds": 15,
        "message": "Save this ID. Query the same task after 15 seconds; never generate again to check progress.",
    }


def _task_id(raw: dict[str, Any]) -> str:
    value = raw.get("task_id") or raw.get("id")
    data = raw.get("data")
    if not value and isinstance(data, dict):
        value = data.get("task_id")
    return str(value or "")


def _result(task_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {
            "task_id": task_id,
            "status": "not_found",
            "success": None,
            "media_urls": [],
            "trace_id": None,
            "polling_interval_seconds": None,
            "message": "No task matched this ID. Verify ID and application access; do not submit a new task.",
        }
    returned_id = raw.get("id") or raw.get("task_id")
    if returned_id and str(returned_id) != task_id:
        raise ValueError("Task response ID differs from the requested ID")
    response = raw.get("response", raw)
    if response is None:
        response = {}
    if not isinstance(response, dict):
        raise ValueError("Task response must be an object; inspect the same task, do not resubmit")
    data = response.get("data", [])
    items = data if isinstance(data, list) else [data]
    words = {
        str(node.get(key, "")).lower()
        for node in [raw, response, *items]
        if isinstance(node, dict)
        for key in ("state", "status")
    }
    if False and raw.get("finished_at") is None:
        status, success = "pending", None
    elif words & _FAILED or response.get("success") is False or response.get("error"):
        status, success = "failed", False
    elif words & _PENDING:
        status, success = "pending", None
    elif response.get("success") is True or words & _DONE or (False and raw.get("finished_at") is not None):
        status, success = "succeeded", True
    else:
        status, success = "pending", None
    urls: list[str] = []
    if status == "succeeded":
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("video_url"), str):
                url = item["video_url"]
                parsed = urlsplit(url)
                if (
                    parsed.scheme in {"https", "http"}
                    and parsed.hostname
                    and not parsed.username
                    and not parsed.password
                ):
                    if url not in urls:
                        urls.append(url)
        if not urls:
            raise ValueError("Task reports success without final media URLs; inspect the same task, do not resubmit")
    output = {
        "task_id": task_id,
        "status": status,
        "success": success,
        "media_urls": urls,
        "trace_id": raw.get("trace_id") or response.get("trace_id"),
        "polling_interval_seconds": 15 if status == "pending" else None,
    }
    if status == "failed":
        output["error"] = response.get("error") or raw.get("error") or "Task failed"
    return output


class HappyHorseToolSpec(BaseToolSpec):
    """HappyHorse native tools provided by Ace Data Cloud; explicit model and no generation retries.

    Construction and to_tool_list send no request. Generation may consume credits.
    """

    spec_functions = [
        ("happyhorse_generate_video", "ahappyhorse_generate_video"),
        ("happyhorse_generate_video_from_image", "ahappyhorse_generate_video_from_image"),
        ("happyhorse_get_task", "ahappyhorse_get_task"),
    ]

    def __init__(self, api_token: str | None = None) -> None:
        token = api_token if api_token is not None else os.environ.get("ACEDATACLOUD_API_TOKEN", "")
        token = _required(token, "ACEDATACLOUD_API_TOKEN")
        if token.lower().startswith("bearer ") or token.startswith("platform-"):
            raise ValueError("Use a raw application API token, without Bearer; not an account-management token")
        self._client = AceDataCloud(api_token=token, base_url=API_BASE, max_retries=0)
        self._async_client = AsyncAceDataCloud(api_token=token, base_url=API_BASE, max_retries=0)

    def happyhorse_generate_video(
        self, prompt: str, model: str, duration: int, resolution: str, ratio: str = "16:9"
    ) -> dict[str, Any]:
        """Submit HappyHorse text-to-video once. Consumes credits; poll the returned ID, never resubmit."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not isinstance(duration, int) or isinstance(duration, bool):
            raise ValueError("duration must be an integer number of seconds")
        if not 3 <= duration <= 15:
            raise ValueError("HappyHorse duration must be 3 through 15 seconds")
        if resolution not in {"720P", "1080P"} or ratio not in {"16:9", "9:16", "1:1", "4:3", "3:4"}:
            raise ValueError("Unsupported HappyHorse resolution or ratio")
        if model not in ["happyhorse-1.0-t2v", "happyhorse-1.1-t2v"]:
            raise ValueError("Choose a model for this HappyHorse action; no automatic substitution")
        handle = self._client.happyhorse.generate(
            action="generate",
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            async_=True,
        )
        return _submitted(handle.id)

    async def ahappyhorse_generate_video(
        self, prompt: str, model: str, duration: int, resolution: str, ratio: str = "16:9"
    ) -> dict[str, Any]:
        """Submit HappyHorse text-to-video once. Consumes credits; poll the returned ID, never resubmit."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not isinstance(duration, int) or isinstance(duration, bool):
            raise ValueError("duration must be an integer number of seconds")
        if not 3 <= duration <= 15:
            raise ValueError("HappyHorse duration must be 3 through 15 seconds")
        if resolution not in {"720P", "1080P"} or ratio not in {"16:9", "9:16", "1:1", "4:3", "3:4"}:
            raise ValueError("Unsupported HappyHorse resolution or ratio")
        if model not in ["happyhorse-1.0-t2v", "happyhorse-1.1-t2v"]:
            raise ValueError("Choose a model for this HappyHorse action; no automatic substitution")
        handle = await self._async_client.happyhorse.generate(
            action="generate",
            prompt=prompt,
            model=model,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            async_=True,
        )
        return _submitted(handle.id)

    def happyhorse_generate_video_from_image(
        self, image_url: str, prompt: str, model: str, duration: int, resolution: str, ratio: str = "16:9"
    ) -> dict[str, Any]:
        """Submit HappyHorse first-frame animation once. Consumes credits; query the same task ID."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not isinstance(duration, int) or isinstance(duration, bool):
            raise ValueError("duration must be an integer number of seconds")
        if not 3 <= duration <= 15:
            raise ValueError("HappyHorse duration must be 3 through 15 seconds")
        if resolution not in {"720P", "1080P"} or ratio not in {"16:9", "9:16", "1:1", "4:3", "3:4"}:
            raise ValueError("Unsupported HappyHorse resolution or ratio")
        if model not in ["happyhorse-1.0-i2v", "happyhorse-1.1-i2v"]:
            raise ValueError("Choose a model for this HappyHorse action; no automatic substitution")
        handle = self._client.happyhorse.generate(
            action="image_to_video",
            prompt=prompt,
            image_url=_url(image_url),
            model=model,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            async_=True,
        )
        return _submitted(handle.id)

    async def ahappyhorse_generate_video_from_image(
        self, image_url: str, prompt: str, model: str, duration: int, resolution: str, ratio: str = "16:9"
    ) -> dict[str, Any]:
        """Submit HappyHorse first-frame animation once. Consumes credits; query the same task ID."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not isinstance(duration, int) or isinstance(duration, bool):
            raise ValueError("duration must be an integer number of seconds")
        if not 3 <= duration <= 15:
            raise ValueError("HappyHorse duration must be 3 through 15 seconds")
        if resolution not in {"720P", "1080P"} or ratio not in {"16:9", "9:16", "1:1", "4:3", "3:4"}:
            raise ValueError("Unsupported HappyHorse resolution or ratio")
        if model not in ["happyhorse-1.0-i2v", "happyhorse-1.1-i2v"]:
            raise ValueError("Choose a model for this HappyHorse action; no automatic substitution")
        handle = await self._async_client.happyhorse.generate(
            action="image_to_video",
            prompt=prompt,
            image_url=_url(image_url),
            model=model,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
            async_=True,
        )
        return _submitted(handle.id)

    def happyhorse_get_task(self, task_id: str) -> dict[str, Any]:
        """Retrieve an existing HappyHorse task once; never generates. Save/reuse the same task_id.

        Pending preview URLs are hidden. Wait at least 15 seconds before another pending query.
        A new tool instance can recover a prior task ID with the same application's authorized token.
        """
        task_id = _required(task_id, "task_id")
        return _result(task_id, self._client.tasks.get(task_id, service="happyhorse"))

    async def ahappyhorse_get_task(self, task_id: str) -> dict[str, Any]:
        """Read the same existing HappyHorse task asynchronously, without an internal polling loop."""
        task_id = _required(task_id, "task_id")
        return _result(task_id, await self._async_client.tasks.get(task_id, service="happyhorse"))

    def close(self) -> None:
        """Close the synchronous client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close both HTTP clients after sync or async usage."""
        self._client.close()
        await self._async_client.close()
