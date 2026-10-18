"""Actual native tools with synthetic API responses only; no real generation."""

import asyncio
import json

import respx

from llama_index.tools.happyhorse_acedatacloud import HappyHorseToolSpec

CLASS = HappyHorseToolSpec
GEN = "happyhorse_generate_video"
EDIT = "happyhorse_generate_video_from_image"
QUERY = "happyhorse_get_task"
ROUTE = "/happyhorse/videos"
EDIT_ROUTE = "/happyhorse/videos"
TASK = "/happyhorse/tasks"
KEY = "video_url"
EXT = "mp4"
KWARGS = {
    "prompt": "Camera moves through a sunrise garden",
    "model": "happyhorse-1.1-t2v",
    "duration": 5,
    "resolution": "720P",
    "ratio": "16:9",
}
EDIT_KWARGS = {
    "prompt": "Camera moves through a sunrise garden",
    "model": "happyhorse-1.1-i2v",
    "duration": 5,
    "resolution": "720P",
    "ratio": "16:9",
    "image_url": "https://example.org/first.png",
}
EXPECTED = {
    "async": True,
    "model": "happyhorse-1.1-t2v",
    "action": "generate",
    "prompt": "Camera moves through a sunrise garden",
    "duration": 5,
    "resolution": "720P",
}
REQUIRE_FINISHED = False


async def main():
    with respx.mock(assert_all_mocked=True) as router:
        router.post("https://api.acedata.cloud" + ROUTE).respond(200, json={"task_id": "synthetic-task"})
        router.post("https://api.acedata.cloud" + TASK).respond(
            200,
            json={
                "id": "synthetic-task",
                "finished_at": 123,
                "response": {
                    "success": True,
                    "data": [{KEY: "https://example.invalid/synthetic." + EXT, "state": "succeeded"}],
                },
            },
        )
        spec = CLASS(api_token="offline-dummy")
        try:
            generate, _, query = spec.to_tool_list()
            submission = generate.call(**KWARGS).raw_output
            recovered = CLASS(api_token="offline-dummy")
            try:
                result = (await recovered.to_tool_list()[2].acall(task_id=submission["task_id"])).raw_output
            finally:
                await recovered.aclose()
            assert result["status"] == "succeeded" and len(router.calls) == 2
            print(
                json.dumps(
                    {
                        "mode": "synthetic-offline",
                        "tools": [t.metadata.name for t in spec.to_tool_list()],
                        "submission": submission,
                        "recovered_result": result,
                    },
                    indent=2,
                )
            )
        finally:
            await spec.aclose()


if __name__ == "__main__":
    asyncio.run(main())
