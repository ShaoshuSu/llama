# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    instructions_text = "You are a very helpful assistant. Please read the message and help me to judge if the following person is " \
                "a professor or not. For the first question only answer 'YES', 'NO' or 'NOT SURE'. "

    content_text =\
        "Yan Zhuang (Senior Member, IEEE) received the Ph.D. degree in control theory and control engineering from the Dalian University of Technology, \
    Dalian, China.,He is a Professor with the School of Artificial Intelligence, Dalian University of Technology, leading the Intelligent Robotics Laboratory \
    (DUT Robotics Laboratory). His research interests include intelligent sensing, modeling, scene recognition, and understanding for mobile robots and other \
    autonomous systems.(Based on document published on 8 August 2023)."

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {
                "role": "system",
                "content": instructions_text          
            },
            {"role": "user", "content": content_text
             },
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    import time
    start = time.time()
    fire.Fire(main)
    print("Time taken: ", time.time() - start, " seconds.")
