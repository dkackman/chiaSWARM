import argparse
from .LLM.agent import load, run_raw_inference
import typing as t
from .LLM.parsing import parse_messages_from_str
from .LLM.prompting import build_prompt_for
import logging
import re


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.CRITICAL)


def main(model_name, device):
    tokenizer, model = load(model_name, device)
    char_settings = {
        "char_name": "AI",
        "char_persona": "A helpful AI",
        "char_greeting": "What do you require?",
        "world_scenario": "A human talks to a powerful AI that follows the human's instructions.",
        "example_dialogue": "",
    }
    generation_settings = {
        "do_sample": True,
        "max_new_tokens": 196,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 0,
        "typical_p": 1.0,
        "repetition_penalty": 1.05,
        "penalty_alpha": 0.6,
    }
    history = []
    while True:
        user_input = input("You: ")
        inference_result = inference_fn(
            tokenizer,
            model,
            history,
            user_input,
            generation_settings,
            char_settings,
            device,
        )

        # sometimes it comes back with a random character + newline at the front
        newline = inference_result.find("\n", 0, 3)
        if newline > -1:
            inference_result = inference_result[newline + 1 :]

        print(inference_result)
        history.append(f"You: {user_input}")
        history.append(inference_result)


def inference_fn(
    tokenizer,
    model,
    history: t.List[str],
    user_input: str,
    generation_settings: t.Dict[str, t.Any],
    char_settings: t.Dict[str, t.Any],
    device,
) -> str:

    # If we're just starting the conversation and the character has a greeting
    # configured, return that instead. This is a workaround for the fact that
    # Gradio assumed that a chatbot cannot possibly start a conversation, so we
    # can't just have the greeting there automatically, it needs to be in
    # response to a user message.
    # if len(history) == 0 and char_settings["char_greeting"] is not None:
    #     return f"{char_settings['char_name']}: {char_settings['char_greeting']}"

    prompt = build_prompt_for(
        history=history,
        user_message=user_input,
        char_name=char_settings["char_name"],
        char_persona=char_settings["char_persona"],
        example_dialogue=char_settings["example_dialogue"],
        world_scenario=char_settings["world_scenario"],
    )

    model_output = run_raw_inference(
        model, tokenizer, prompt, user_input, device, **generation_settings
    )

    generated_messages = parse_messages_from_str(
        model_output, ["You", char_settings["char_name"]]
    )

    bot_message = generated_messages[0]

    return bot_message


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
    # parser.add_argument(
    #     "-p",
    #     "--persona",
    #     help="Path to a configuration json that has the persona setup",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--chat",
    #     help="Path to a json file with chat history",
    # )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args_from_argv()

    main(model_name=args.model_name, device=args.device)
