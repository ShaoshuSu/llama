# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


# import os

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12356'
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'

class LLAMA_Generation:
    def __init__(self,
        ckpt_dir='llama-2-7b-chat/',
        tokenizer_path='tokenizer.model',
        max_seq_len=512,
        max_batch_size=8):

        self.generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        )
        self.temperature = 0.6
        self.top_p = 0.9
        self.max_gen_len = None
        self.instructions_text = "You are a very helpful assistant. Please read the message and help me to judge if the following person is " \
                "a professor or not. For the first question only answer 'YES', 'NO' or 'NOT SURE'. "


    def generate(self, 
                #  instructions_text: str,
                 content_text: str,
                 verbose: bool = True):
        
        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": self.instructions_text          
                },
                {"role": "user", "content": content_text
                },
            ],
        ]
        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                if verbose:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            if verbose:
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n")

        return result['generation']['content']