#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-28 06:10:36 (ywatanabe)"
# /mnt/ssd/ripple-wm-code/scripts/bloom.py


"""
This script does XYZ.
"""


"""
Imports
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import mngs
import seaborn as sns

mngs.gen.reload(mngs)
import warnings
from glob import glob
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from icecream import ic
from natsort import natsorted
from tqdm import tqdm

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()



"""
Functions & Classes
"""

import transformers
from transformers import AutoTokenizer
import torch

model = "meta-llama/Llama-2-70b-chat"
#model = “meta-llama/Llama-2-70b-chat-hf”

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

def main():
    # cat  | xargs huggingface-cli login --token

    import torch
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BloomForCausalLM, BloomTokenizerFast

    hf_token = mngs.io.load(os.getenv("HOME") +
                            '/.bash.d/secrets/access-tokens/hugging-face-fast-token.txt')[0]

    # Check if CUDA is available and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    # Load the model and tokenizer
    model_name = "bigscience/bloomz-7b1"
    # model_name = "bigscience/bloom-1b3"
    model = BloomForCausalLM.from_pretrained(model_name, token=hf_token)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    __import__("ipdb").set_trace()



    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, torch_dtype=torch.float16, device_map="auto"
    # )

    # # Move the model to GPU if available
    # model.to(device)

    # # Function to generate text
    # def generate_text(prompt, max_length=100):
    #     inputs = tokenizer(prompt, return_tensors="pt").to(device)

    #     # Generate
    #     with torch.no_grad():
    #         generated_ids = model.generate(
    #             **inputs,
    #             max_length=max_length,
    #             num_return_sequences=1,
    #             temperature=0.7,
    #             top_k=50,
    #             top_p=0.95,
    #         )

    #     return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # # Example usage
    # prompt = "Explain the concept of artificial intelligence in simple terms:"
    # generated_text = generate_text(prompt)
    # print(f"Prompt: {prompt}\n")
    # print(f"Generated text: {generated_text}")

    # # Interactive loop
    # while True:
    #     user_input = input("\nEnter a prompt (or 'quit' to exit): ")
    #     if user_input.lower() == "quit":
    #         break

    #     generated_text = generate_text(user_input)
    #     print(f"\nGenerated text: {generated_text}")

    # print("Thank you for using BLOOMZ-7b1!")

    # pass


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
