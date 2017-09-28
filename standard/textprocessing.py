from collections import Counter

def create_lookup_tables(text):
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 0)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

def token_lookup():
    tlookup = {
        '.': '||Period||',
        ',': '||Comma||',
        '"': '||Quotation_Mark||',
        ';': '||Semicolon||',
        '!': '||Exclamation_Mark||',
        '?': '||Question_Mark||',
        '(': '||Left_Parentheses||',
        ')': '||Right_Parentheses||',
        '--': '||Dash||',
        '\n': '||Return||'
        }
    return tlookup

def preprocess(text):
    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]

    return token_dict, vocab_to_int, int_to_vocab, int_text
