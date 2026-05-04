import random
import re
import heapq
import numpy as np
from random import randint
from collections import Counter, defaultdict
from conllu import parse, parse_tree
from tqdm.autonotebook import tqdm
import warnings
warnings.filterwarnings('ignore')
import math
import os
"""
This is a slightly changed version of the Conllu_UD_Parser of https://github.com/RohinV/Probing.
This parser is used to create SentEval datasets made from Universal Dependencies datasets.

"""


class Conllu_UD_Parser:
    def __init__(self,file_mapping):
        self.file_mapping=file_mapping
        

    def filter_sents(self, token_data):
        """
        Function to filter sentences that have
        emoticons, emopjis, urls, and email addresses from
        the corpus.
        Args:
        token_data: the corpus received from parsing.
        """

        def check_sent_len(token_data):
            """
            Function the checks whether the sentences
            have at least 2 words.
            """
            return len(token_data.metadata['text'].split(" ")) >= 2

        def contains_noise(token_data):
            """
            Function that checks for emojis,
            emoticons, urls, and email addresses.
            """
            text = token_data.metadata['text']
            emoji_pattern = re.compile(
                "["
                u"\U0001F600-\U0001F64F"
                u"\U0001F300-\U0001F5FF" 
                u"\U0001F680-\U0001F6FF"  
                u"\U0001F1E0-\U0001F1FF" 
                u"\U00002500-\U00002BEF"
                u"\U00002700-\U000027BF"  
                u"\U000024C2-\U0001F251" 
                u"\U0001F900-\U0001F9FF"  
                u"\U0001FA70-\U0001FAFF"  
                u"\U0001F018-\U0001F270"  
                u"\U0001F000-\U0001F02F"  
                "]+", flags=re.UNICODE
            )

            emoticon_pattern = re.compile(
                r'(?:(?:[:;=8][\-o\*\'"]?[)\]dDpP3]+)|(?:[)\]dDpP3]+[\-o\*\'"]?[:;=8])|<3|:\'\(|:-?\(|:-?\/|:-?\\|:‑?O|:-?\||:-?S)',
                re.IGNORECASE
            )

            url_pattern = re.compile(r'https?://\S+|www\.\S+')

            email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')

            return bool(1-bool(
                emoji_pattern.search(text) or
                emoticon_pattern.search(text) or
                url_pattern.search(text) or
                email_pattern.search(text)
            ))

        length_check = check_sent_len(token_data)
        noise_check = contains_noise(token_data)
        return all([length_check, noise_check])
    


    def read_conllu_file(self, file_path, parsing: str):
        """
        Function that reads a conllu file and returns the corpus
        Args:
        file_path: path of the connlu file.
        parsing: argument that takes in a string to convert the connlu
                file into a TokenList or TokenTree data structure.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            if parsing == "PARSE_TREE":
                token_trees = list(filter(self.filter_sents, parse_tree(data)))
                return token_trees
            elif parsing == "PARSE":
                token_lists = list(filter(self.filter_sents, parse(data)))
                return token_lists


    def binning_algorithm(self, data, k):
        """
        Partition sorted class labels into k bins with consecutive ranges
        such that total frequency per bin is balanced.
        
        Returns:
            List of tuples representing ranges: [(start1, end1), (start2, end2), ...]
        """
        
        freq = Counter(data)
        sorted_classes = sorted(freq.keys())

       
        class_freqs = [(cls, freq[cls]) for cls in sorted_classes]

        total = sum(freq.values())
        target_per_bin = total / k

        bins = []
        current_bin = []
        current_sum = 0

        for cls, f in class_freqs:
            current_bin.append(cls)
            current_sum += f
            
            if current_sum >= target_per_bin and len(bins) < k - 1:
                bins.append((current_bin[0], current_bin[-1]))
                current_bin = []
                current_sum = 0

        
        if current_bin:
            bins.append((current_bin[0], current_bin[-1]))

        bins_range=[range(start, end + 1) for start, end in bins]
        bins = {i: bins_range[i] for i in range(len(bins_range))}
        return bins
    


    def train_test_val_split(self,gen_text:list,train=0.8,val=0.1,test=0.1):
            """
            Function that splits the data into train, validation, and test sets.
            """
            total=len(gen_text)
            train_size = int(total * train)  
            test_size = int(total * test)    
            val_size = int(total* val)  

            remaining = total - (train_size + test_size + val_size)
            train_size += remaining  
            
            indices = np.arange(total)
            np.random.shuffle(indices)


            train_indices = indices[:train_size]
            test_indices = indices[train_size:train_size + test_size]
            val_indices = indices[train_size + test_size:]

            train_data=np.array(gen_text)[train_indices]
            test_data=np.array(gen_text)[test_indices]
            val_data=np.array(gen_text)[val_indices]

            train_data=["tr\t"+ i for i in train_data]
            test_data=["te\t"+ i for i in test_data]
            val_data=["va\t"+ i for i in val_data]
            
            return  train_data,test_data,val_data
        
    
    def sorting(self,data):
        """
        Function to sort the tasks with numerical label
        in ascending order.
        """
        return sorted(data, key=lambda x: x.split('\t')[1])
    

    
    def remove_skew(self, data):
        """
        Function that removes any skewness in the labels. 
        A label must be at least 10% of the entire corpus. 
        Lastly, it takes the label with least amount of data and 
        normalizes the data with repect to rest of the labels.
        """
        min_count = len(data)*0.1
        classes=Counter([i.split('\t')[0] for i in data])
        filtered_classes = {k: v for k, v in classes.items() if v >= min_count}

        if filtered_classes:
            min_value = min(filtered_classes.values())
            selected_keys = [k for k, v in filtered_classes.items() if v >= min_value]
        
        new_data=[]
        for i in selected_keys:
            new_data += list(filter(lambda x: x.split("\t")[0] == i, data))[:min_value]
            
        return new_data
    

    def write(self, data, language, task): 
        """
        Function that writes the data of each language and task
        into text files.
        Args:
        data: the data returned from the task.
        task: name of the task
        language: name of the language
        """
        os.makedirs(os.path.join('SentEval_Data', language), exist_ok=True)
        file_path = os.path.join('SentEval_Data', language, task+".txt")
        with open(file_path, "w", encoding="utf-8") as file:
            for line in data:
                file.write(line + "\n")
        return file_path



######################--------------SENT_LEN------------##################
    def sent_len(self, file_path):
        """
        Function that find the lenght of all the sentences
        in the corpus of the language.
        Args:
        file_path: the file path of the language corpus.
        """
        token_lists = self.read_conllu_file(file_path, parsing="PARSE")
        sent_length = []
        for token_list in token_lists:
            sent_length.append(len(token_list))

        bins = self.binning_algorithm(sent_length, 6)

        all_text = """"""
        for token_list in token_lists:
            sent_length = len(token_list)
            tokens = [token['form'] for token in token_list]

            for bin_num, wrd_range in bins.items():
                if sent_length in wrd_range:
                    sentence = " ".join(tokens)
                    all_text += f"{bin_num}\t{sentence}\n"
                else:
                    continue

        return all_text.split('\n')[:-1]

    
############################-----------TENSE_EXTRACT---------####################   
    def tense_extract(self, file_path):
        """
        Function that finds and extracts the tense of all
        the sentences in the corpus.
        """
        token_trees = self.read_conllu_file(file_path, parsing="PARSE_TREE")

        all_data = ""

        def find_tense(node):
            if token_check(node):  
                return node
            for child in node.children:
                result = find_tense(child)
                if result:
                    return result
            return None

        def token_check(token):
            try:
                if token.token['feats']:
                    if 'Tense' in token.token['feats']:
                            return True
            except:
                return False

            return False


        for token_tree in (token_trees):
            head = token_tree
            
            if token_check(head.token):
                all_data += f"{head.token['feats']['Tense']}\t{token_tree.metadata['text']}\n"
                
            else:
                tense_node = find_tense(head)
                
                if tense_node:
                    all_data += f"{tense_node.token['feats']['Tense']}\t{token_tree.metadata['text']}\n"
                    
        return all_data.split('\n')[:-1]
    


################################-------------WORD CONTENT------------######################
    def word_content(self, token_lists):
        """
        Function for the word content task.
        """

        tokens=[]
        for token_list in token_lists:
            for token in token_list:
                if token['upos'] and (token['upos'] not in ['PUNCT', 'NUM']):
                    if token['upos'] in {'NOUN', 'VERB', 'ADJ',}:
                        if len(token['form']) > 4:
                            tokens.append(token['form'].lower())

        word_counts = Counter(tokens)
        print(len(word_counts))

        most_common_words = []
        all_common_words = word_counts.most_common(6000)

        for item in range(0, 500):
            if item < len(all_common_words):
                most_common_words.append(all_common_words[item][0])
        
        
        processed_sentences = """"""
        for token_list in token_lists:
            forms = [token['form'] for token in token_list]
            forms_lower = [token.lower() for token in forms]

            matches = [w for w in forms_lower if w in most_common_words]

            if len(matches) == 1:
                target = matches[0]
                sentence = " ".join(forms)
                processed_sentences += f"{target}\t{sentence}\n"
        print("Number of usable sentences: {}".format(
            len(processed_sentences.split("\n")[:-1])
        ))
        return processed_sentences.split('\n')[:-1]
    

#############################--------------OBJ NUMBER-------------####################
    def obj_num(self, file_path):
        """
        Function that finds sentences that have a direct object 
        in the sentence and extracts the number of the object 
        for all sentences in the corpus.
        """
        token_trees = self.read_conllu_file(file_path, parsing="PARSE_TREE")
        
        all_data = """"""

        def find_first_noun(node):
            if token_check(node):  
                return node
            for child in node.children:
                result = find_first_noun(child)
                if result:
                    return result
            return None

        def token_check(token):
            try:
                if token.token['upos'] == 'NOUN':
                    if token.token['deprel'] and token.token['deprel'] in ('obj', 'iobj', 'dobj'):
                        return True
            except:
                return False

            return False


        for token_tree in tqdm(token_trees):
            head = token_tree
            
            if token_check(head.token):
                if head.token['feats'] and 'Number' in head.token['feats']:
                    if head.token['feats']['Number'] == 'Sing':
                        all_data += f"{head.token['feats']['Number']}\t{token_tree.metadata['text']}\n"
                    else:
                        all_data += f"{head.token['feats']['Number']}\t{token_tree.metadata['text']}\n"
                
            else:
                noun_node = find_first_noun(head)
                
                if noun_node:
                    if noun_node.token['feats'] and 'Number' in noun_node.token['feats']:
                        if noun_node.token['feats']['Number'] == 'Sing':
                            all_data += f"{noun_node.token['feats']['Number']}\t{token_tree.metadata['text']}\n"
                        else:
                            all_data += f"{noun_node.token['feats']['Number']}\t{token_tree.metadata['text']}\n"
                            
                
        return all_data.split('\n')[:-1]
    

#############################-------------SUBJ NUMBER----------###########################
    def subj_num(self, file_path):
        """
        Function that finds sentences that have a direct subject 
        in the sentence and extracts the number of the subject 
        for all sentences in the corpus
        """
        token_trees = self.read_conllu_file(file_path, parsing="PARSE_TREE")
        
        all_data = ""

        def find_first_noun(node):
            if token_check(node):  
                return node
            for child in node.children:
                result = find_first_noun(child)
                if result:
                    return result
            return None

        def token_check(token):
            try:
                if token.token['upos'] == 'NOUN':
                    if token.token['deprel'] and token.token['deprel'] in ('nsubj', 'csubj', 'nsubj:pass'):
                        return True
            except:
                return False

            return False


        for token_tree in tqdm(token_trees):
            head = token_tree
            
            if token_check(head.token):
                if head.token['feats'] and 'Number' in head.token['feats']:
                    all_data += f"{head.token['feats']['Number']}\t{token_tree.metadata['text']}\n"
                
            else:
                noun_node = find_first_noun(head)
                
                if noun_node:
                    if noun_node.token['feats'] and 'Number' in noun_node.token['feats']:
                        all_data += f"{noun_node.token['feats']['Number']}\t{token_tree.metadata['text']}\n"
                
        return all_data.split('\n')[:-1]
    
    
     
############################------------TREE DEPTH------------###############################
    def tree_depth(self,file_path):
        """
        Computes the depth of the dependency tree for each sentence
        in a CoNLL-U file using TokenTree from the conllu module.

        Depth is defined as the length of the longest path from root to any leaf.
        """

        def compute_tree_depth(tree):
            """
            Recursively compute depth of a TokenTree.
            """
            if not tree.children:
                return 1
            return 1 + max(compute_tree_depth(child) for child in tree.children)

        token_trees = self.read_conllu_file(file_path, parsing="PARSE_TREE")
        token_lists = self.read_conllu_file(file_path, parsing="PARSE")
        all_data = ""
        all_depths = []

        for tree in tqdm(token_trees):
            depth = compute_tree_depth(tree)
            all_depths.append(depth)

        bins = self.binning_algorithm(all_depths, 6)

        for tree, depth in zip(token_trees, all_depths):
            for tkl in token_lists:
                if tkl.metadata.get("text", "") == tree.metadata.get("text", ""):
                    tokens = [token["form"] for token in tkl]
                    sentence_text = " ".join(tokens)
                    break
            for bin_num, depth_range in bins.items():
                if depth in depth_range:
                    all_data += f"{bin_num}\t{sentence_text}\n"
                    break

        return all_data.strip().split('\n')
    
    
#############################-----------BIGRAM SHIFT--------------############################   
    def bigram_shift(self, file_path):
        """
        Function that randomly selects a sentence and shifts a 
        random bigram in a sentence in the entire corpus
        """
        token_lists = self.read_conllu_file(file_path, parsing="PARSE")

        def sentence_validity(token_list):
            for token in token_list:
                if token['form'] == '"':
                    return False
            return True

        def token_validity(token):
            return token['upos'] not in ['PUNCT', 'NUM']

        def get_token_form(tokens):
            sw_sent = []
            for token in tokens:
                sw_sent.append(token['form'])
            return " ".join(sw_sent)

        def select_random_sentences(token_lists, invert_percentage=0.5):

            eligible_tokenlists = [token_list for token_list in token_lists if all(
                [sentence_validity(token_list), len(token_list) > 5])]
            rand_num_sentences = round(
                len(eligible_tokenlists)*invert_percentage)
            selected_tokenlists = random.sample(
                eligible_tokenlists, rand_num_sentences)
            remaining_tokenlists = [
                token_list for token_list in eligible_tokenlists if token_list not in selected_tokenlists]
            return selected_tokenlists, remaining_tokenlists

        def shift_bigrams(token_list):
            tokens = [token for token in token_list if isinstance(
                token['id'], int)]

            max_attempts = 10
            for i in range(max_attempts):
                idx = random.randint(1, len(tokens)-3)

                if token_validity(tokens[idx]) and token_validity(tokens[idx + 1]):
                    tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
                    break

            out = get_token_form(tokens)

            return out

        selected, remaining = select_random_sentences(token_lists)
        biragam_shifted_sents = [shift_bigrams(
            token_list) for token_list in selected]

        remaining_sent = []
        for token_list in remaining:
            remaining_sent.append(get_token_form(
                [token for token in token_list if isinstance(token['id'], int)]))

        combined = list(map(lambda x: "I\t"+x, biragam_shifted_sents)
                        )+list(map(lambda x: "O\t"+x, remaining_sent))
        random.shuffle(combined)
        return combined
    
###############################---------SOMO-------------#################################
    def odd_man_out(self, file_path):
        """
        Function that randomly selects a noun or verb and replaces 
        it with the next closest noun or verb in log probabiltites 
        for the entire corpus.
        """
        token_lists = self.read_conllu_file(file_path, parsing="PARSE")


        def get_log_freq(token_lists):
            tokens = []
            for token_list in token_lists:
                for token in token_list:
                    if token['upos'] not in ['NUM','PUNCT']:
                        tokens.append((token['form'],token['upos']))

            uni_list = dict(Counter(tokens))
            log_uni_list = {i: math.log(j) for i, j in uni_list.items()}
            return dict(sorted(log_uni_list.items(), key=lambda item: item[1],reverse=True))

        
        def get_replaceable_n(target_key, upos):
            try:
                target_log_prob = log_freq.get((target_key, upos))
                closest_key = None
                closest_diff = float('inf')

                for key, val in log_freq.items():
                    if key[1] == upos and key[0]!= target_key:
                        diff = abs(val - target_log_prob)
                        if diff < closest_diff:
                            closest_diff = diff
                            closest_key = key[0]
                return closest_key

            except:
                return None

        log_freq = get_log_freq(token_lists)

        altered_sents = []
        not_altered = []
        target_altered_count = len(token_lists) // 2  
        altered_count = 0 

        for token_list in tqdm(token_lists):
            if altered_count >= target_altered_count:
               
                original_sentence = " ".join([t['form'] for t in token_list])
                not_altered.append(f"O\t{original_sentence}")
                continue

            altered = False  

            for idx in range(1, len(token_list)-1):
                token = token_list[idx]

                if token['upos'] in ('NOUN', 'VERB'):
                    repl_token = get_replaceable_n(token['form'], token['upos'])
                    if repl_token is None:
                        repl_token = token['form']
                    new_sentence = [t['form'] if i != idx else repl_token for i, t in enumerate(token_list)]
                    altered_sents.append(f"C\t{' '.join(new_sentence)}")
                    altered = True
                    altered_count += 1
                    break


            if not altered:
                original_sentence = " ".join([t['form'] for t in token_list])
                not_altered.append(f"O\t{original_sentence}")

        all_data=altered_sents+not_altered
        random.shuffle(all_data)
        
        return all_data
#########################------------COORDINATION INVERSION------------#############################################
    def coordination_inversion(self, file_path):

            def has_single_cconj(token_lists):

                def token_check(token):
                    try:
                        if token.token['upos'] == 'CCONJ':
                            return True
                    except:
                        return False
                    return False


                def find_cconj(node):
                    result = []
                    if token_check(node):
                        result.append(node)
                    for child in node.children:
                        result.extend(find_cconj(child))
                    return result 
                
                data = []
                for token_list in token_lists:
                    count = 0

                    head=token_list.to_tree()
                    if token_check(head.token):
                        count += 1

                    cconj_nodes = find_cconj(head)
                    if cconj_nodes:
                        count += len(cconj_nodes)
                            
                    if count == 1:
                        data.append(token_list)

                return data
                    
            def verb_num(token_list):

                def token_check(token):
                    try:
                        if token.token['upos'] in ['VERB', 'AUX']:
                            return True
                    except:
                        return False
                    return False
                
                def find_verb(node):
                    result = []
                    if token_check(node):
                        result.append(node)
                    for child in node.children:
                        result.extend(find_verb(child))
                    return result
                
                verb_num = 0
                head = token_list.to_tree()
                if token_check(head):
                    verb_num += 1

                verb_nodes = find_verb(head)
                if verb_nodes:
                    verb_num += len(verb_nodes)
                
                return verb_num


            def cconj_verb_relation(token_list):
                
                def token_check(token):
                    try:
                        if token.token['upos'] in ['CCONJ']:
                            return True
                    except:
                        return False
                    return False
                
                def find_cconj(node):
                    result = []
                    if token_check(node):
                        result.append(node.token)
                    for child in node.children:
                        result.extend(find_cconj(child))
                    return result

                head = token_list.to_tree()
                cconj_node = find_cconj(head)
                for conj in cconj_node:
                    head_index = conj['head'] - 1
                    if token_list[head_index]['upos'] in ["VERB", "AUX"]:
                        return True

                return False


            def coordinate_clause_check(token_list):
                first_clause = []
                second_clause = []
                cc_token = None
                cc_token_head = None

                for token in token_list:
                    if token['upos'] == 'CCONJ':
                        cc_token = token['id']
                        cc_token_head = token['head']

            
                for idx in range(cc_token-1):
                    first_clause.append(token_list[idx]['form'])
                for idx in range(cc_token, len(token_list)):
                    second_clause.append(token_list[idx]['form'])

                return cc_token, first_clause, second_clause


            def clause_len_check(first_clause, second_clause):

                if round(len(first_clause)/ len(second_clause)) == 1:
                    return True
                else:
                    return False

            token_lists = self.read_conllu_file(file_path, parsing="PARSE")
            data = has_single_cconj(token_lists)

            all_data = []
            for tkl in data:
                if verb_num(tkl) >= 2 and cconj_verb_relation(tkl):
                    cc_token, first_clause, second_clause = coordinate_clause_check(tkl)
                    if clause_len_check(first_clause, second_clause):
                        all_data.append(tkl)

            count = len(all_data) // 2
            final_data = """"""
            for tkl in all_data[:count]:
                cc_token, first_clause, second_clause = coordinate_clause_check(tkl)
                first_word = second_clause[0].title()
                end_punctuation = second_clause[-1]
                word_after_cc = first_clause[0].lower()
                inverted_1_clause = []
                inverted_2_clause = []

                inverted_2_clause.append(word_after_cc)
                for idx in range(1, len(first_clause)):
                    inverted_2_clause.append(first_clause[idx])
                inverted_2_clause.append(end_punctuation)

                inverted_1_clause.append(first_word)
                for idx in range(1, len(second_clause)-1):
                    inverted_1_clause.append(second_clause[idx])

                sentence = f"I\t{' '.join(inverted_1_clause)} {tkl[cc_token - 1]['form']} {' '.join(inverted_2_clause)}\n"
                final_data += sentence
            
            for tkl in all_data[count:]:
                tokens = []
                for token in tkl:
                    tokens.append(token['form'])
                
                sentence = f"O\t{' '.join(tokens)}\n"
                final_data += sentence

            data = final_data.strip().split("\n")

            
            data = [line for line in data if line.strip()]

            
            if not data:
                return []

            random.shuffle(data)
            return data

#########################-------------OUTPUT FUNCTION--------------###############################
    def process(self):
        """
        Function that processes all the tasks and gives the desired output.
        """
        languages=list(self.file_mapping.keys())
        all_data={}
        for lang in languages:
            paths=self.file_mapping[lang]   #{'eng':{'sent_length:['6/thfjdfjdh/n'],'tree_depth:[]},'french':{'sent_length:['6/thfjdfjdh/n'],'tree_depth:[]}}
            all_data[lang]= {
                'sentence_length': self.sent_len(paths[0]) + self.sent_len(paths[1]), #binning, sort ,skew
                'tree_depth': self.tree_depth(paths[0]) + self.tree_depth(paths[1]), #binning ,sort, skew
                
                'subj_number': self.remove_skew(self.subj_num(paths[0]) + self.subj_num(paths[1])),#skew
                'obj_number': self.remove_skew(self.obj_num(paths[0]) + self.obj_num(paths[1])),#skew
                'past_present': self.remove_skew(self.tense_extract(paths[0]) + self.tense_extract(paths[1])), #skew
                
                'bigram_shift': self.bigram_shift(paths[0]) + self.bigram_shift(paths[1]),
                'word_content': self.word_content(self.read_conllu_file(paths[0], parsing="PARSE") + self.read_conllu_file(paths[1], parsing="PARSE")),
                'odd_man_out': self.odd_man_out(paths[0]) + self.odd_man_out(paths[1]),
                'coordination_inversion': self.coordination_inversion(paths[0]) + self.coordination_inversion(paths[1])
                }
        
           
        data_paths={}
        languages=list(all_data.keys())
        task_names=list(all_data[languages[0]].keys())
        
        for lang in languages:
            for task in task_names:
                if task in ['sentence_length','tree_depth']:
                    train_data,test_data,val_data=self.train_test_val_split(all_data[lang][task])
                    train_data, test_data, val_data = self.sorting(train_data), self.sorting(test_data), self.sorting(val_data)
                
                else:
                    train_data, test_data, val_data = self.train_test_val_split(all_data[lang][task])
                    
                    
                path=self.write(train_data+val_data+test_data,lang,task)
                data_paths[lang] = path  #{'english': '/english','french':'/french'} ->output
                
        return data_paths
    