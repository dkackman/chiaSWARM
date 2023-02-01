import argparse
from .chat.agent import load, chat


def main(model_name, device):
    tokenizer, model = load(model_name, device)
    chat(tokenizer, model, device)


def _parse_args_from_argv() -> argparse.Namespace:
    """Parses arguments coming in from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name",
        help="HuggingFace Transformers model name.",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda",
        help="The device to run the model on - cpu or cuda.",
    )
    parser.add_argument(
        "-p",
        "--persona",
        help="Path to a configuration json that has the persona setup",
    )
    parser.add_argument(
        "-c",
        "--chat",
        help="Path to a json file with chat history",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args_from_argv()

    main(model_name=args.model_name, device=args.device)
