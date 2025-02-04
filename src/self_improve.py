if __package__ is None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from src.training_loop import self_improve, preprocess, _tokenizer
else:
    from .training_loop import self_improve, preprocess, _tokenizer

if __name__ == '__main__':
    import sys
    prompt_text = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else input('Enter prompt: ')
    prompt = preprocess(prompt_text)
    improved = self_improve(prompt)
    print(f"Improved prompt: {_tokenizer.decode(improved[0], skip_special_tokens=True)}") 