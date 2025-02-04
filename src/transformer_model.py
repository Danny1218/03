from transformers import GPT2LMHeadModel
from transformer_criteria import get_config


def create_model():
    return GPT2LMHeadModel(get_config())


if __name__ == '__main__':
    model = create_model()
    print('Model has', sum(p.numel() for p in model.parameters()), 'parameters') 