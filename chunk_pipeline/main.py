import os
import argparse

import chunk_pipeline
import chunk_pipeline.pipelines as pipelines
from chunk_pipeline.configs import Config

import sys
import logging

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="Configuration file(s) to read, filename without directory",
    )
    parser.add_argument("--pipeline", nargs=1, help="Pipeline to run")
    args = parser.parse_args()
    logging.info(args)
    if args.pipeline is None:
        raise ValueError("No pipeline specified")

    cfg = Config(os.path.join(os.path.dirname(chunk_pipeline.__file__), "configs"))
    for file in ["default.py"] + args.config:
        cfg.from_pyfile(file)

    pipeline = getattr(pipelines, args.pipeline[0])
    pipeline = pipeline(cfg)

    pipeline.run()
