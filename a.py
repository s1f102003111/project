import lzma
import sys
import glob

def load_xz_ngram_file(n=2):
    ngram_dict = {}
    ngram_freq = {}
    files = glob.glob("/path/to/nwc2010-ngrams/word/over999/{}gms/*.xz".format(n))
    for file in files:
        print(file)
        with lzma.open(file, "rt") as f:
            for line in f.readlines():
                words, freq = line.strip().split("\t")
                if "</S>" in words or words.endswith("「") or words.endswith("【"):
                    # 文末記号とカッコは除外
                    continue

                freq = int(freq)
                word1, word2 = words.rsplit(" ", maxsplit=1)
                if ngram_dict.get(word1) is None or (ngram_dict.get(word1) and ngram_freq[word1] < freq):
                    ngram_dict[word1] = word2
                    ngram_freq[word1] = freq
    print("load {} words".format(len(ngram_dict)))
    return ngram_dict

def get_next_words(word, ngram, cnt=10):
    target_word = word
    result = [word]
    for _ in range(cnt):
        if ngram.get(target_word) is None:
            print("not get {}".format(target_word))
            return result
        else:
            next_word = ngram[target_word]
            result.append(next_word)
            if len(target_word.split(" ")) > 1:
                target_word = target_word.split(" ", maxsplit=1)[-1] + " " + next_word
            else:
                target_word = next_word
    return result


def interface(n=2, cnt=10):
    ngram = load_xz_ngram_file(n=n)
    print("To finish, type </S>")
    while True:
        word = input('Enter {} word: '.format(n-1))
        if word == '</S>':
            print('FINISH')
            break
        else:
            if len(word.split(" ")) != n-1:
                print("Wrong input: {} words".format(len(word.split(" "))))
            else:
                res = get_next_words(word, ngram, cnt)
                print(" ".join(res))


if __name__ == '__main__':
    argv = sys.argv[1:]
    if argv:
        if len(argv) >1 and argv[1].isnumeric():
            interface(int(argv[0]), int(argv[1]))
        else:
            interface(int(argv[0]))
    else:
        print("Usage: python ngram_generate.py n [generate_num]")