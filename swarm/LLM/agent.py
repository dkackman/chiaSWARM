import transformers
import torch
import typing as t


def load(model_name, device):
    print(f"Loading {model_name} to {device}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    bad_words_ids = [
        tokenizer(bad_word, add_special_tokens=False).input_ids
        for bad_word in ["Persona:", "Scenario:", "<START>"]
    ]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, bad_words_ids=bad_words_ids
    )
    model.config.pad_token_id = model.config.eos_token_id
    if device == "cuda":
        model = model.eval().half().to(device)
    else:
        model = model.to(device)

    print("done.")
    return tokenizer, model


def run_raw_inference(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    prompt: str,
    user_message: str,
    device: str,
    **kwargs: t.Any,
) -> str:
    """
    Runs inference on the model, and attempts to returns only the newly
    generated text.

    :param model: Model to perform inference with.
    :param tokenizer: Tokenizer to tokenize input with.
    :param prompt: Input to feed to the model.
    :param user_message: The user's raw message, exactly as appended to the end
        of `prompt`. Used for trimming the original input from the model output.
    :return: Decoded model generation.
    """
    tokenized_items = tokenizer(prompt, return_tensors="pt").to(device)  # type: ignore
    # Atrocious code to stop generation when the model outputs "\nYou: " in
    # freshly generated text. Feel free to send in a PR if you know of a
    # cleaner way to do this.
    stopping_criteria_list = transformers.StoppingCriteriaList(
        [
            _SentinelTokenStoppingCriteria(
                sentinel_token_ids=tokenizer(
                    "\nYou:",
                    add_special_tokens=False,
                    return_tensors="pt",  # type: ignore
                ).input_ids.to(
                    device
                ),  # type: ignore
                starting_idx=tokenized_items.input_ids.shape[-1],
            )
        ]
    )

    logits = model.generate(  # type: ignore
        stopping_criteria=stopping_criteria_list, **tokenized_items, **kwargs
    )  # type: ignore
    output = tokenizer.decode(logits[0], skip_special_tokens=True)  # type: ignore

    # Trim out the input prompt from the generated output.
    if (idx := prompt.rfind(user_message)) != -1:
        trimmed_output = output[idx + len(user_message) - 1 :].strip()

        return trimmed_output

    raise Exception("Couldn't find user message in the model's output. What?")


class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, sentinel_token_ids: torch.LongTensor, starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx :]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                0, self.sentinel_token_ids.shape[-1], 1
            ):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False
