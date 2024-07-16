# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

import csv
import os
import pandas as pd
import re
import pylcs
from tqdm import tqdm

import time     


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
            "based on the information available." \
            "Please provide your response in the following format: 'YES', 'NO', or 'NOT SURE'." \
            "You can answer 'YES' only when that text mentions that the individual is a professor." \
            "You can answer 'YES' only when that text has direct confirmation of professorship" \
            "Researcher is not a direct confirmation of professorship." 
            # "And provide me a score of your confidence for if that person is a professor." \
            # "Use your analytical capabilities to review the details given and make a judgment " \
            # "\nFor example, for the following, the answer is 'NO': " \
            # "Changhwan Kim (Member, IEEE) received the B.S. degree in mechanical engineering and the M.S. degree in machine design engineering from Hanyang University, Seoul, South Korea, in 1993 and 1995, respectively, and the Ph.D. degree in mechanical engineering from The University of Iowa, Iowa City, IA, USA, in 2002. From 2002 to 2004, he was a Research Associate with the Robotics and Automation Laboratory, University of Notre Dame, Notre Dame, IN, USA. Since 2004, he has been working at the Korea Institute of Science and Technology (KIST), Seoul. His research interests include task and motion planning for robot manipulation and social robots.(Based on document published on 26 December 2022)." \
            # "\nFor another example, the following is not a professor:" \
            # "Younbaek Lee received the M.S. degree in mechanical engineering from Korea Advanced Institute of Science and Technology, Daejeon, South Korea, in 2003.,He is currently a research staff member of the Samsung Advanced Institute of Technology, South Korea. His research interests include electromechanical design, medical/rehabilitation robotics, humanoids, manipulators and physical human-machine interaction.(Based on document published on 14 June 2019)." \
            # "\nFor one more example, the following is not a professor: " \
            # "Shenghai Yuan received the bachelor’s and Ph.D. degrees in electrical and electronic engineering from the Nanyang Technological University, Singapore, in 2013 and 2019, respectively.,He is a Research Fellow with the EEE Internet of Things Lab, Nanyang Technological University. His research interests include the area of perception, sensor fusion for robust navigation, machine learning, and autonomous system.(Based on document published on 15 August 2023)." \
            # "\nFor the fourth example, the following is a professor: " \
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

        if "not sure" in result['generation']['content'].lower():
            return "NOT SURE", result['generation']['content']
        if "yes" in result['generation']['content'].lower():
            return "YES", result['generation']['content']
        elif "no" in result['generation']['content'].lower():
            return "NO", result['generation']['content']
        else:
            return None, None

        # return result['generation']['content']
    

    def judge_affiliation(self, 
                content_text: str,
                uni_string: str,
                verbose: bool = True):
        
        instruct_text = \
        """
        Given a list, identify the corresponding institute in that list based on the input institute information. 
        If the input institute is not in the list, respond with 'N/A'.
        Your response should only just include only the full name of that institute in the given list.
        Example: use ""University of California, Berkeley (UCB)"" instead of "University of California, Berkeley"
        The output format is just one institute name or  'N/A'.
        Do not include any other information, such as information about other institutes.
        """
        # Example: use "Nanyang Technological University, Singapore (NTU)" instead of "Nanyang Technological University"
        
        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": instruct_text          
                },
                {   "role": "user", 
                 "content": 
                "The qurey institute is " + content_text + \
                "\n\nHere is the given list:\n" + uni_string   
                },
            ],
            # [
            #     {
            #         "role": "user",
            #         "content": 
            #         "The qurey institute is " + content_text + \
            #         "\n\nHere is the given list:\n" + uni_string           
            #     },
            # ]
        ]
        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # result_list = []
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                if verbose:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            if verbose:
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n")
            # result_list.append(result['generation']['content'])

        # return result_list
        return result['generation']['content']

        # if "N/A" in result['generation']['content']:
        #     return "N/A", result['generation']['content']
        # else:
        #     return result['generation']['content'], result['generation']['content']


    # @staticmethod
    def llama_find_affiliation(self,content_text,verbose = False,n=16):
        count = 0
        max_try = 5

        result_dict = {}

        self.divide_university_list('./country-info3_5.csv', n=n)


        for idx in tqdm(range(max_try), disable=not verbose):
            count += 1

            # if verbose:
            #     print(f"{count}/{max_try} th try")
            
            result, len_lcs = self.llama_find_affiliation_one_try(content_text,verbose=verbose,n=n)

            # if len_lcs > 0:
            if verbose:
                print(f'{result}:  lcs:{len_lcs}')
            result_dict[result] = len_lcs

        if verbose:
            print(result_dict)
        # find the maximum key
        max_key = max(result_dict, key=result_dict.get)
        max_len_lcs = result_dict[max_key]

        if max_len_lcs > len(content_text)/6:
            result = max_key
        else:
            result = 'N/A'

        return result, max_len_lcs


    # from tqdm import tqdm
    def llama_find_affiliation_one_try(self,content_text,verbose=False,n=16):
        """
        Find the affiliation of the author by comparing the content text with the university list.
        
        Parameters:
        - content_text: The text to be compared.
        - verbose: Whether to print the progress.
        - n: The number of files to split the university list into.
        """
        result_dict = {}
        
        # Divide the university list into n parts
        # try to find the affiliation from each part
        for idx in tqdm(range(1, n+1), disable=not verbose):

            # print(f'\n{idx}/{n}')

            file_path = f'./data/country-info3_5_{idx}_{n}.csv'  # Replace this with your file path

            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                # next(reader, None)  # Skip the header row
                first_column_values = [row[0] for row in reader if row]  # Check if row is not empty
                uni_string = '\n'.join(first_column_values)

            count = 0

            # find a proper response with less than 300 characters and less than 10 tries
            while True:
                count += 1
                # affiliation, full_answer = self.judge_affiliation(content_text, uni_string, verbose=False)
                response = self.judge_affiliation(content_text, uni_string, verbose=False)

                if 'N/A' in response:
                    # return 'N/A', 0
                    response = 'N/A'

                if verbose:
                    print()
                    # print(affiliation)
                    print(response)

                if len(response) < 300:
                    break

                if count > 10:
                    print('Too many tries')
                    print(response)
                    break
            

            uni_list = self.load_university_list(file_path)
            affiliation = self.split_string_by_delimiters(response)
            # print(affiliation)

            # compounds = affiliation.split('\n',)

            for element in affiliation:
                part = element.strip()
                # elements = compound.split(':')
                # print(compound)
                # for element in elements:
                # result, len_lcs = exam_result(part)
                result, len_lcs = self.exam_result(content_text, part , uni_list)

                # print(result, len_lcs)
                if len_lcs > 0:
                    # print(f'{part}:  lcs:{len_lcs}')
                    result_dict[part] = len_lcs

        if verbose:
            print(result_dict)

        # find the maximum key
        if len(result_dict) == 0:
            return 'N/A', 0
        else:
            max_key = max(result_dict, key=result_dict.get)

            return  max_key, result_dict[max_key]

    @staticmethod
    def exam_result(query_text, query_result , uni_list):
        if query_result in uni_list:
            # lcs_len = pylcs.lcs_sequence_length(query_text, query_result)
            lcs_len = pylcs.lcs_string_length(query_text, query_result)
            # print('Correct exist')
            # print('Length of LCS:', lcs_len)

            return True, lcs_len
        else:
            # print('Not exist')
            return False, 0

    @staticmethod
    def divide_university_list(file_path, n=16):
        # Load the original CSV file
        df = pd.read_csv(file_path)

        # delete data folder if it exists
        import shutil
        shutil.rmtree('./data', ignore_errors=True)

        # Create a new data folder
        os.makedirs('./data', exist_ok=False)

        # Calculate the number of rows per file
        rows_per_file = len(df) // n

        # Split the dataframe into n parts and save them
        for i in range(n):
            start_index = i * rows_per_file
            if i == n-1:  # Ensure the last file includes any leftover rows
                end_index = len(df)
            else:
                end_index = start_index + rows_per_file
            part_df = df.iloc[start_index:end_index]
            part_df.to_csv(f'./data/country-info3_5_{i+1}_{n}.csv', index=False)

    @staticmethod
    def load_university_list(file_path):
        uni_list = []

        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # next(reader, None)  # Skip the header row
            uni_list = [row[0] for row in reader if row]  # Check if row is not empty
            # csv_string = '\n'.join(first_column_values)

        # print(uni_list)

        return uni_list
    
    @staticmethod
    def split_string_by_delimiters(text):
        # Split the text by either ":" or "\n"
        split_result = re.split(r'[:\n]+', text)
        return split_result


    def translate(self, 
                content_text: str,
                verbose: bool = True):
        
        instruct_text = \
        """
        Extract the university name from the given institute information.
        And then translate the given institute to its official English Name.
        You only need to provide the translated result using the format
        English: [translated university name]
        """
        
        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": instruct_text          
                },
                {   "role": "user", 
                    "content": 
                "The information of the input institute is " + content_text 
                },
            ],
        ]
        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # result_list = []
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                if verbose:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            if verbose:
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n")
            # result_list.append(result['generation']['content'])

        # return result_list
        print(result['generation']['content'])
        return result['generation']['content']

        # if "N/A" in result['generation']['content']:
        #     return "N/A", result['generation']['content']
        # else:
        #     return result['generation']['content'], result['generation']['content']


    def llm_extract_uni_name(self, 
                content_text: str,
                verbose: bool = True):
        
        instruct_text = \
        """
        Extract the university name from the given institute information.
        Put the answer after a colon.
        And do not use any other punctuation marks.
        """
        
        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": instruct_text          
                },
                {   "role": "user", 
                    "content": 
                "The information of the input institute is " + content_text 
                },
            ],
        ]
        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # result_list = []
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                if verbose:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            if verbose:
                print(
                    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                )
                print("\n==================================\n")
            # result_list.append(result['generation']['content'])

        # return result_list
        # print(result['generation']['content'])

        # output = result['generation']['content'].split(':')[-1]
        # return output

        return result['generation']['content']

        # if "N/A" in result['generation']['content']:
        #     return "N/A", result['generation']['content']
        # else:
        #     return result['generation']['content'], result['generation']['content']


    def pre_exam(self,affiliation_string):
        file_path = 'country-info3_5.csv'  # Replace this with your file path
        uni_list = self.load_university_list(file_path)

        affiliations = affiliation_string.split(',')
        for part in affiliations:
            # print(part)
            part = part.strip()
            if part in uni_list:
                print('Correct exist')
                print(part)
                return True, part
            
        return False, None
    


    def extract_uni_name(self,content_text, verbose = False):
        if verbose:
            print("Extract University Name for:")
            print(content_text)

        last_result = ''
        count = 0
        while True:
            count += 1
            extract_content_text = self.llm_extract_uni_name(content_text, verbose=False)

            if verbose:
                print(f'\nExtraction Try {count}: {extract_content_text}')

            # count the number of ";" in the string
            num_semicolon = extract_content_text.count(':')

            # print(num_semicolon)

            if last_result == extract_content_text: 
                if verbose:
                    print('Same result')
                time.sleep(5)

            if num_semicolon <= 1:
                break
            else:
                last_result = extract_content_text
            
            if count > 10:
                # break
                return "N/A"

        extract_result = self.split_string_by_delimiters(extract_content_text)[-1].strip()

        if verbose:
            print('\nFinal extracted:', extract_result)
            print('==================================\n')

        return extract_result
