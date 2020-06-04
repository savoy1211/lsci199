import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from preprocess.dataPreprocess import preprocess_en as preprocess
import csv


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Entropy(X) = H[X] = - \sum_i p(x_i) \log p(x_i)
# Entropy Reduction 1 = H[X_i] - H[X_{i+1}] <- we can get this
# Entropy Reduction 2 = H[all continuations | sentence up to i] - H[all continuations | sentence up to i+1]


def prob(sentence, with_delimiters=True):
    if with_delimiters:
        sentence_tokens = [tokenizer.bos_token_id] + tokenizer.encode(sentence) + [tokenizer.eos_token_id]
    else:
        sentence_tokens = tokenizer.encode(sentence)
    sentence_tensor = torch.tensor([sentence_tokens])
    if torch.cuda.is_available():
        sentence_tensor = sentence_tensor.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        predictions = model(sentence_tensor)
        probabilities = torch.log_softmax(predictions[0], -1)
        # print(probabilities[0].shape)
        model_probs = probabilities[0, :, tuple(sentence_tokens)].diag(int(with_delimiters))
        entropy = - (probabilities[0,:,:].exp() * probabilities[0,:,:]).sum(-1)[1:]

    # entropy_reduction = []
    # for idx, item in enumerate(entropy):
    #     if idx < len(entropy) - 1:
    #         entropy_reduction.append(entropy[idx].item() - entropy[idx + 1].item())

    mProb = []
    # mProb.append(('<BOS>', 0.0))
    for token, prob in zip(sentence_tokens[1:], model_probs):
        if tokenizer.decode(token) != '<|endoftext|>':
            mProb.append((tokenizer.decode(token).strip(' '), prob.item()))
    mProb.append(('<EOS>', 0.0))

    return mProb


if __name__ == '__main__':
    if len(sys.argv) > 1:
        sys.exit(print(prob(sys.argv[1])))
    else:
        sentence = "It's raining cats and dogs."
        result = prob(sentence)
        print(result)
        print(len(result))
