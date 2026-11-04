import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select the best trained model")
    parser.add_argument("--artifacts-dir", default="forecasting/artifacts")
    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    metadata = json.loads((artifacts_dir / "model_metadata.json").read_text())
    source = artifacts_dir / Path(metadata["best"]["checkpoint"]).name
    target = artifacts_dir / "best_model.pt"
    shutil.copy2(source, target)
    print(f"Selected {metadata['best']['model']} -> {target}")


if __name__ == "__main__":
    main()
