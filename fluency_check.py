import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def scoring_sentence_fluency(
    sent: str,
    tokenizer: GPT2Tokenizer,
    lm_model: GPT2LMHeadModel,
    device: torch.device,
) -> float:
    """Scoring the fluency of sentence using LM (GPT2)

    Args:
        sent (str): sentence to be scored
    Returns:
        score (float): fluency of the given sentence, measured by perplexity from GPT2.
    """
    tokenized_sentence = tokenizer(sent, return_tensors="pt")["input_ids"]
    if device is not None:
        tokenized_sentence = tokenized_sentence.to(device)
    loss = lm_model(tokenized_sentence, labels=tokenized_sentence, return_dict=True)[
        "loss"
    ]
    perplexity = torch.exp(loss)
    return float(perplexity.cpu().detach().numpy())


def get_tokenizer_and_model():
    """
    Tokenizer와 LM을 불러와서 반환합니다.

    Returns:
        [type]: [description]
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, lm_model


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else None
    sample_sentence = "This town is famous for its beautiful buildings."
    tokenizer, lm_model = get_tokenizer_and_model()
    if device is not None:
        lm_model.to(device)
    score = scoring_sentence_fluency(sample_sentence, tokenizer, lm_model, device)
    print("Perplexity: {}".format(score))
