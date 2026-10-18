"""Query a previously submitted task once, without generating."""

import argparse
import asyncio
import json

from llama_index.tools.happyhorse_acedatacloud import HappyHorseToolSpec


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    args = parser.parse_args()
    spec = HappyHorseToolSpec()
    try:
        print(json.dumps(await spec.ahappyhorse_get_task(args.task_id), indent=2))
    finally:
        await spec.aclose()


if __name__ == "__main__":
    asyncio.run(main())
