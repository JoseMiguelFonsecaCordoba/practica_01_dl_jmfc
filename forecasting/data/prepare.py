import argparse

from forecasting.data.series import download_jena


def main():
    parser = argparse.ArgumentParser(description="Download the Jena dataset")
    parser.add_argument("--data-dir", default="forecasting/data")
    args = parser.parse_args()
    csv_path = download_jena(args.data_dir)
    print(f"Dataset ready: {csv_path}")


if __name__ == "__main__":
    main()
