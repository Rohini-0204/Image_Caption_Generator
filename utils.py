import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image):
    """
    Resizes and prepares the image for the VGG16 model.
    """
    # Resize to VGG16 input size
    img = image.resize((224, 224))
    # Convert to array
    img = img_to_array(img)
    # Reshape for model (batch dimension)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # Preprocess (subtract mean RGB)
    img = preprocess_input(img)
    return img

def idx_to_word(integer, tokenizer):
    """
    Converts a predicted integer ID back into a word.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image_features, tokenizer, max_length=35):
    """
    Generates a caption one word at a time using Greedy Search.
    """
    in_text = 'startseq'
    
    for i in range(max_length):
        # Encode the current input string to a sequence of integers
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence (MUST match the 'post' padding used in training!)
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        # Predict the next word probabilities
        yhat = model.predict([image_features, sequence], verbose=0)
        # Get the index with the highest probability
        yhat = np.argmax(yhat)
        
        # Convert index back to word
        word = idx_to_word(yhat, tokenizer)
        
        # Stop if no word found or reached end of caption
        if word is None or word == 'endseq':
            break
            
        # Append the word for the next iteration
        in_text += " " + word
        
    # Clean the final output
    final_caption = in_text.replace('startseq', '').strip()
    return final_caption