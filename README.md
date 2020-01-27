# Long Short-Term Memory Neural Network for Sentiment Analysis

A machine learning model that learns to label phrases taken from movie reviews on Rotten Tomatoes on a scale of five values: negative (0), somewhat negative (1), neutral (2), somewhat positive (3), and positive (4).

## Datasets

The dataset I used parsed sentences from the Rotten Tomatoes dataset into many phrases with the Stanford parser. Each single data point consists of a PhraseId, SentenceId, Phrase, and Sentiment label. There are 156,060 unique data points in total. Repeated phrases, such as short and/or common words, are included only once in the dataset.

My pre-processing steps included converting each phrase to lower-case, removing any punctuation, and getting rid of any phrases with a length of zero. I then encode the phrases such that each word corresponds to an index. The less common the word, the higher the index. Additionally, I make each phrase the same length by adding zeros to the beginning of the encoding if the length of the phrase is less than the length of the longest phrase in the dataset.

I split the data into three sets such that 70% of the data points are for use in training, 15% for validation, and another 15% for testing.

## Model

I approached this task by implementing a Long Short-Term Memory recurrent neural network. 

For each phrase in the training data, I run the forward pass which creates a 50 dimensional word embedding for the input phrase with PyTorch’s Embedding module. Then it runs the LSTM, which takes word embeddings as inputs and outputs hidden states with dimensionality 50. Next, it maps from the hidden state space to the sentiment score space using PyTorch’s Linear module, which applies a linear transformation to the incoming data. After that, it applies a softmax followed by a logarithm to the sentiment score space, with the dimension along which this is computed being 1. Finally, it returns the last element of the output of the log_softmax function.

I used Negative Log Likelihood for my loss function and Stochastic Gradient Descent as my optimizer.

## Conclusions

After optimizing the LSTM, the combination of hyperparameters that yielded the highest validation accuracy was an embedding dimension of 50, hidden dimension of 50, learning rate of 0.1, 7 epochs, and batch size of 50. Making predictions on the test set using these hyperparameter values resulted in a test loss of 0.0098, and test accuracy of 0.481735.

## References

Inspiration for this project and the dataset I used came from [this](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) Kaggle competition.

