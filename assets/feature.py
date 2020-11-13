import argparse
from collections import Counter

class WordMapper():
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = []
        self.vocab_size = None
    
    def load_dict(self,dict_file):
        with open(dict_file,mode='r') as f:
            for line in f.readlines():
                word = line.strip().split(' ')[0]
                idx = line.strip().split(' ')[1]

                self.word_to_idx[word] = int(idx)
                self.idx_to_word[int(idx)] = word
                self.vocab.append(word)
        
        self.vocab_size = len(self.vocab)

class Preprocessor():
    def __init__(self,feature_flag,word_mapper):
        self.feature_flag = int(feature_flag)
        self.word_mapper = word_mapper
    
    def process(self,input_file,output_file):
        with open(output_file,mode='w') as f1:
            with open(input_file,mode='r') as f2:
                if self.feature_flag == 1:
                    for line in f2.readlines():
                        new_line = ''
                        label,words_counter = self.parse_line(line)
                        new_line += label
                        for pair in words_counter:
                            word = pair[0]
                            freq = pair[1]

                            try:
                                idx = self.word_mapper.word_to_idx[word]
                                new_line += '\t' + str(idx) + ':1'
                            except KeyError:
                                continue
                        new_line += '\n'
                        f1.write(new_line)
                else:
                    for line in f2.readlines():
                        new_line = ''
                        label,words_counter = self.parse_line(line)
                        new_line += label
                        for pair in words_counter:
                            word = pair[0]
                            freq = pair[1]

                            if freq >= 4:
                                continue
                            else:
                                try:
                                    idx = self.word_mapper.word_to_idx[word]
                                    new_line += '\t' + str(idx) + ':1'
                                except KeyError:
                                    continue
                        new_line += '\n'
                        f1.write(new_line)

                    
    def parse_line(self,line):
        label = line.strip().split('\t')[0]
        review_text = line.strip().split('\t')[1]
        words = review_text.split(' ')
        words_counter = Counter(words).most_common()

        return label,words_counter


def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('Sentiment Polarity Analyzer - Feature Engineering')
    parser.add_argument('train_input',help='path to the training input .tsv file')
    parser.add_argument('validation_input',help='path to the validation input .tsv file')
    parser.add_argument('test_input',help='path to the test input .tsv file')
    parser.add_argument('dict_input',help='path to the dictionary input .txt file')
    parser.add_argument('formatted_train_out',help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument('formatted_validation_out',help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument('formatted_test_out',help='path to output .tsv file to which the feature extractions on the test data should be written')
    parser.add_argument('feature_flag',help='integer taking value 1 or 2 that specifies whether to construct the Model 1 feature set or the Model 2 feature set')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    word_mapper = WordMapper()
    word_mapper.load_dict(args.dict_input)

    preprocessor = Preprocessor(args.feature_flag,word_mapper)

    preprocessor.process(args.train_input,args.formatted_train_out)
    preprocessor.process(args.validation_input,args.formatted_validation_out)
    preprocessor.process(args.test_input,args.formatted_test_out)