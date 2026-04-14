**PROBLEM STATEMENT:**
People write reviews, comments, feedback
Businesses need to understand sentiment automatically
Manual analysis is slow

**Solution:**
A real-time sentiment analysis system using ML

User → Streamlit UI → FastAPI → ML Model → Response → UI

**app.py**
FastAPI handles API requests
Model loads once at startup
Two endpoints:
Single prediction
Batch prediction

Input → Clean → Tokenize → Vector → Predict

**text_cleaner.py**
The same preprocessing used during training is reused during prediction to maintain accuracy.
steps includes :
  Lowercase
  Remove noise
  Normalize text
"Don't like this!!!" → "do not like this"

**Word2Vec Concept**
  Words are converted into vectors (numbers)
  Similar words have similar vectors

      Each word → vector
    All vectors → averaged
    Final vector → model input

**streamlit_app.py**
Streamlit provides interactive UI
Sends API requests to backend
Displays predictions
**features:**
Single input analysis
Batch processing
Analytics dashboard
History tracking



**Why we used Word2Vec instead of BOW?**
Word2Vec captures the meaning and relationships between words, 
while Bag of Words only counts word frequency without understanding context.

**Bag of Words (BoW)**
Converts text into word counts
**Ignores:**
Word meaning
Word order
Context
**Example:**
"I love this product"
"I like this product"

**Word2Vec (W2V)**
Converts each word into a vector (numbers)
**Captures:**
Semantic meaning
Word relationships
Context similarity
**Example:**
Vector("king") - Vector("man") + Vector("woman") ≈ Vector("queen")
  It Understands relationsip, not just the words!.





  Architecture:


<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/10ce3248-e1bb-4a6a-add5-0334e1a3a6ba" />













  
