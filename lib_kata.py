import numpy
import jax.numpy as jnp

def load_dataset_and_tokenize():
    # load dataset, and tokenize
    with open("shakespeare.txt", "r", encoding="latin-1") as f:
        text = f.read()
    
    # but not with the GPT-4 tokenizer, it is too large for this experiment. Do something quite a bit smaller for now.
    # shakespeare_encoded = jnp.array(enc.encode(text))
    
    text_numpy = numpy.array([x for x in text.lower()])
    chars, counts = numpy.unique(text_numpy, return_counts=True)
    count_sorter = numpy.argsort(counts)[::-1] # inverse frequency sort
    vocabulary = chars[count_sorter]
    vocabulary_freqs = counts[count_sorter]
    print(f'vocabulary: \n{",".join(vocabulary)} \n size: {len(vocabulary)} chars')
    tokenizer_stoi = {ch: i for i, ch in enumerate(vocabulary)}
    tokenizer_itos = {i: ch for i, ch in enumerate(vocabulary)}
    text_encoder = lambda X: jnp.array([tokenizer_stoi[x] for x in X], dtype=jnp.int32)
    text_decoder = lambda X: "".join([tokenizer_itos[int(x)] for x in X])
    # invertability test
    print(text_decoder(text_encoder(text_numpy[0:100])))
    text_encoded = text_encoder(text_numpy)
    
    return text_encoded, text_encoder, text_decoder, len(vocabulary)
    

def make_training_Xy(data, context_size=8):
    data_cpu = numpy.array(data)
    _X=[data_cpu[n:n+context_size] for n in range(data_cpu.shape[0]-context_size)]
    _y=[data_cpu[n+context_size] for n in range(data_cpu.shape[0]-context_size)]
    return _X, _y

def preview_Xy(X,y, text_decoder, count=8, offset=0):
    for idx in range(count):
        ptr = idx + offset
        print(X[ptr], " -> " , y[ptr], " | ",  text_decoder(X[ptr]), "_", text_decoder(numpy.array([y[ptr]])))