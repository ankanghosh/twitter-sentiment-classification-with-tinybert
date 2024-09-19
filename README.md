# twitter-sentiment-classification-with-tinybert
Classification of the sentiment of tweets in the Sentiment140 dataset using a light-weight, optimized, compressed, and distilled version of the Bidirectional Encoder Representations from Transformers (BERT) model, known as TinyBERT. TinyBERT offers a good balance between performance and efficiency.

# About the project
Classification of the sentiment (positive / negative) of tweets in the Sentiment140 dataset using TinyBERT.
This is a Google Colab notebook to analyze and classify the sentiment of tweets in the Sentiment140 dataset. The data is first cleaned, normalized, and preprocessed. TinyBERT is used to implement the model for the binary classification of text. Training is performed across Tensor Processing Unit (TPU) cores using TensorFlow's distributed computing setup for TPUs. The model performs fairly well and achieves an accuracy of about 83%.

# Tools Used
Tools and libraries used in this project include TensorFlow, Keras, Transformers, pandas, csv, NumPy, scikit-learn, Matplotlib, and seaborn.

# Who can benefit from the project?
Anyone can use the project to get started with the basics of sentiment analysis or binary classification using text data and TensorFlow.

# Getting Started
Anyone interested in getting started with Machine Learning, Deep Learning, or Natural Language Processing, specifically, sentiment analysis or binary classification using text data and TensorFlow and BERT / Transformer networks, can clone or download the project to get started.

# References
I have leveraged ChatGPT for guidance in terms of some concepts and to clarify doubts, both theoretical and code-based. However, I did not use ChatGPT to generate code. The most important points of reference for the project are as follows.
1. TensorFlow tutorial about fine-tuning BERT to perform sentiment analysis. Link [here](https://www.tensorflow.org/text/tutorials/classify_text_with_bert).
2. TensorFlow tutorial about using basic text classification to perform sentiment analysis. Link [here](https://www.tensorflow.org/tutorials/keras/text_classification).

# Additional Notes
1. The dataset is available [here](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip). If the dataset is taken down in the future, please feel free to reach out to me at ankanatwork@gmail.com if you would like to learn more about the data. However, I may not be able to share the dataset with you due to licensing restrictions.
2. The project is a basic one in nature and is not currently being maintained.
3. [Here](https://researchguy.in/twitter-sentiment-classification-using-tensorflow-and-tinybert/) is the blog post covering this work.
