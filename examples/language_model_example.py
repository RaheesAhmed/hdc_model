"""
Example usage of the HDC language model.

This script demonstrates how to use the HDC language model for text generation.
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc_model.models.language_model import NGramLanguageModel, ContextualLanguageModel


def main():
    # Sample corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "a stitch in time saves nine",
        "all that glitters is not gold",
        "actions speak louder than words",
        "the early bird catches the worm",
        "practice makes perfect",
        "look before you leap",
        "the pen is mightier than the sword",
        "when in rome do as the romans do",
        "the cat sat on the mat",
        "the dog barked at the cat",
        "the bird flew over the tree",
        "the fish swam in the pond",
        "the sun shines bright in the sky",
        "the moon glows at night",
        "the stars twinkle in the dark",
        "the rain falls from the clouds",
        "the snow covers the ground",
        "the wind blows through the trees",
        "the river flows to the sea",
    ]
    
    print("Training N-Gram Language Model...")
    
    # Create and train the n-gram model
    ngram_model = NGramLanguageModel(n=3, dimensions=10000, vector_type='binary')
    ngram_model.fit(corpus)
    
    # Generate text with the n-gram model
    seed = ["the", "quick"]
    generated = ngram_model.generate(seed, length=10)
    
    print(f"N-Gram Model Seed: {' '.join(seed)}")
    print(f"N-Gram Model Generated: {' '.join(generated)}")
    
    # Calculate perplexity
    test_text = "the quick brown fox jumps over the lazy cat"
    perplexity = ngram_model.perplexity(test_text)
    print(f"N-Gram Model Perplexity: {perplexity:.2f}")
    
    print("\nTraining Contextual Language Model...")
    
    # Create and train the contextual model
    contextual_model = ContextualLanguageModel(context_size=3, dimensions=10000, vector_type='binary')
    contextual_model.fit(corpus)
    
    # Generate text with the contextual model
    seed = ["the", "cat", "sat"]
    generated = contextual_model.generate(seed, length=10)
    
    print(f"Contextual Model Seed: {' '.join(seed)}")
    print(f"Contextual Model Generated: {' '.join(generated)}")
    
    # Predict next tokens
    context = ["the", "quick", "brown"]
    predictions = contextual_model.predict_next(context, top_k=3)
    
    print(f"\nContext: {' '.join(context)}")
    print("Top 3 predictions:")
    for token, score in predictions:
        print(f"  {token}: {score:.4f}")


if __name__ == "__main__":
    main()
