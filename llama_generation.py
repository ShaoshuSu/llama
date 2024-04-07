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
        self.instructions_text = "You are tasked with analyzing the information provided to determine" \
            "if the individual in question holds the position of a professor." \
            "Use your analytical capabilities to review the details given and make a judgment " \
            "based on the information available." \
            "Please provide your response in the following format: 'YES', 'NO', or 'NOT SURE'." \
            # "For example, the following is not a professor:" \
            # "Changhwan Kim (Member, IEEE) received the B.S. degree in mechanical engineering and the M.S. degree in machine design engineering from Hanyang University, Seoul, South Korea, in 1993 and 1995, respectively, and the Ph.D. degree in mechanical engineering from The University of Iowa, Iowa City, IA, USA, in 2002. From 2002 to 2004, he was a Research Associate with the Robotics and Automation Laboratory, University of Notre Dame, Notre Dame, IN, USA. Since 2004, he has been working at the Korea Institute of Science and Technology (KIST), Seoul. His research interests include task and motion planning for robot manipulation and social robots.(Based on document published on 26 December 2022)." \
            # "For another example, the following is not a professor:" \
            # "Younbaek Lee received the M.S. degree in mechanical engineering from Korea Advanced Institute of Science and Technology, Daejeon, South Korea, in 2003.,He is currently a research staff member of the Samsung Advanced Institute of Technology, South Korea. His research interests include electromechanical design, medical/rehabilitation robotics, humanoids, manipulators and physical human-machine interaction.(Based on document published on 14 June 2019)." \
            # "For one more example, the following is not a professor:" \
            # "Shenghai Yuan received the bachelor’s and Ph.D. degrees in electrical and electronic engineering from the Nanyang Technological University, Singapore, in 2013 and 2019, respectively.,He is a Research Fellow with the EEE Internet of Things Lab, Nanyang Technological University. His research interests include the area of perception, sensor fusion for robust navigation, machine learning, and autonomous system.(Based on document published on 15 August 2023)." \
            # "For the third example, the following is a professor:" \
            # "Guyue Zhou (Member, IEEE) received the B.E. degree from the Harbin Institute of Technology, Harbin, China, in 2010, and the Ph.D. degree from the Hong Kong University of Science and Technology, Hong Kong, China, in 2014. He is currently an Associate Professor with the Institute for AI Industry Research (AIR), Tsinghua University. His research interests include advanced manufacturing, robotics, computer vision, and human–machine interaction.(Based on document published on 22 August 2022)."



    def judge_professorship(self, 
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
                # {"role": "user", "content": "For example, the following is not a professor:" \
                #  "Younbaek Lee received the M.S. degree in mechanical engineering from Korea Advanced Institute of Science and Technology, Daejeon, South Korea, in 2003.,He is currently a research staff member of the Samsung Advanced Institute of Technology, South Korea. His research interests include electromechanical design, medical/rehabilitation robotics, humanoids, manipulators and physical human-machine interaction.(Based on document published on 14 June 2019)."
                # },

                # {"role": "user", "content": "For another example, the following is not a professor:" \
                #  "Shenghai Yuan received the bachelor’s and Ph.D. degrees in electrical and electronic engineering from the Nanyang Technological University, Singapore, in 2013 and 2019, respectively.,He is a Research Fellow with the EEE Internet of Things Lab, Nanyang Technological University. His research interests include the area of perception, sensor fusion for robust navigation, machine learning, and autonomous system.(Based on document published on 15 August 2023)."
                # },

                # {"role": "user", "content": "For the third example, the following is a professor:" \
                # "Guyue Zhou (Member, IEEE) received the B.E. degree from the Harbin Institute of Technology, Harbin, China, in 2010, and the Ph.D. degree from the Hong Kong University of Science and Technology, Hong Kong, China, in 2014. He is currently an Associate Professor with the Institute for AI Industry Research (AIR), Tsinghua University. His research interests include advanced manufacturing, robotics, computer vision, and human–machine interaction.(Based on document published on 22 August 2022)." \
                # },
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

        if "YES" in result['generation']['content']:
            return "YES", result['generation']['content']
        elif "NO" in result['generation']['content']:
            return "NO", result['generation']['content']
        else:
            return "NOT SURE", result['generation']['content']

        # return result['generation']['content']