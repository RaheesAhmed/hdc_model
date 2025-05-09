"""
Simple test script for HDC language model.
"""

import numpy as np
from hdc_model.models.language_model import NGramLanguageModel

def main():
    print("Testing HDC language model...")
    
    # Sample corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a stitch in time saves nine",
        "all that glitters is not gold",
        "actions speak louder than words",
        "the early bird catches the worm",
    ]
    
    print("Creating language model...")
    model = NGramLanguageModel(n=3, dimensions=10000, vector_type='binary')
    
    print("Training language model...")
    model.fit(corpus)
    
    print("Model trained successfully!")
    
    # Test prediction
    context = ["the", "quick"]
    print(f"\nPredicting next word for context: {context}")
    try:
        predictions = model.predict_next(context, top_k=3)
        print("Top 3 predictions:")
        for token, score in predictions:
            print(f"  {token}: {score:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    # Test generation
    print("\nGenerating text...")
    try:
        generated = model.generate(context, length=5)
        print(f"Generated text: {' '.join(generated)}")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
