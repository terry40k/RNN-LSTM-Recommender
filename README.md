# RNN LSTM Recommender
This repository contains code for training an RNN LSTM architecture using one of PyTorch's core libraries (torch) to build a recommender system.  Our use case was for a furniture rental company dataset, but it also works for other types of recommendations.  

# Prerequisites
Ensure you have the following libraries installed:

- Python 3.7 or later
- PyTorch

Other Python libraries:
- NumPy
- Pandas

# Dataset
The notebook utilizes a custom dataset with 13 features.

# Code Structure
# Data Loading and Preprocessing
- Reads the input CSV, sorts by account_number and agreement_date, and converts model_number into numeric IDs (item2id)
- Splits data into training, validation, and test sets based on quantiles of agreement_date
- Groups items by user to create customer level sequences that are further filtered (no sequences of length < 2)

# DataLoader Sequence
- Transforms each customer's sequence into (input_seq, target) samples and predicts the next item in the sequence via the build_rnn_samples function
- We create an RNNDataset class and use PyTorch's DataLoader for the samples and enable batching and shuffling

# RNN LSTM Model Definition
- We create an RNNModel class using PyTorch's neural network module (nn.Module) that uses an embedding layer
- We have a forward pass function that has the LSTM's final hidden state through a fully connected layer to predict logits.  We later convert those logits for probabilities and use them for ranking metrics

# Training Loop
- We use train_rnn for our loop over the training batches, performing forward/backward passes and use an Adam optimizer to minimize cross-entropy loss

# Evaluation Functions
- We use the evaluate_rnn function to compute training and validation losses without updating model parameters for each epoch
- The final epoch prints the testing loss

# Recommendation Logic
- An item_info dictionary that maps each item ID to a tuple (model_number, model_description_1, model_description_2) with an rnn_recommended_items function that retrieves the top-K predictions and returns descriptions for each recommended item

# Ranking Metrics
- We use evaluate_rnn_ranking to calculates top-K metrics like Hit Rate, MRR, and NDCG
- Hit Rate measures the proportion of customers who receive at least one relevant recommendation within the top K items recommended by the model.  For our model, we want to see at least one relevant item within the top 5 recommendations.  The higher the number, the higher percentage of customers that see a relevant item within the top 5
- MRR measures how quickly the first relevant item appears in our recommended list.  The higher the number, the more effective our model’s at placing the most appealing or relevant items near the top 5 recommendations
- NDCG not only measures the relevance of the item but also assesses the position of where that item is in the recommended list.  A higher score means the model’s suggesting the more relevant items higher up the list.  This is particularly useful for cases where the overall ranking quality of multiple relevant items is important.

# Hyperparameters
- I probably should've defined the hyperparameters up front so you spend less time editing different places of the code.  For now,  look for the below variables
- Feel free to adjust these parameters to your liking.  You may want to experiment using a larger batch size (e.g. 64 or 128) and increase your embedding and hidden state dimensions (e.g. 128, 256, or even 512) so you have a better and larger representation of your data.  Useful if you have the hardware and/or compute units for GPU acceleration.  You may also start with a lower epoch threshold as it can take time to train the model, especially when we increase dimensions and the number of LSTM layers

- ```max_seq_len``` (this is the maximum number of items in a sequence)
- ```embed_dim``` (how large the representation is for each sequence)
- ```hidden_dim``` (how large the memory cell’s hidden state while processing each sequence)
- ```dropout``` (used to randomly drop a percentage of cells; useful for the network to generalize better and reduces chances of overfitting against the training data)
- ```num_layers``` (the number of LSTM layers stacked on top of each other; allows the network to learn more complex patterns than a single layer)
- ```batch_size``` (the number of pairs used in the sequence)
- You can also adjust the number of epochs from this code:
```sh
for epoch in range(10):
    train_loss = train_rnn(rnn_model, rnn_train_loader, rnn_optimizer, rnn_criterion)
    val_loss = evaluate_rnn(rnn_model, rnn_val_loader, rnn_criterion)
    print(f"[RNN] Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

# Optimizer Settings
- Uses the Adam optimizer with a learning rate of ```1e-4```
- You may want to experiment with different learning rates (e.g. the default 1e-3) until you find a good convergence point
