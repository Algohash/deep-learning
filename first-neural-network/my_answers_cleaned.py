import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Defining activation function
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        #print("N_records: ", n_records)
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # print(X, y)
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch - one row of features i.e. single data point in feature space.

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # Hidden inputs takes the input data points X and multiple that with the weights. Here dot products multiplies two matrices.
        hidden_inputs = np.dot(np.transpose(X), self.weights_input_to_hidden) # signals into hidden layer
        # The hidden_output is the output coming out from the activation function. For the hidden layers the activation
        # function is a sigmoid.
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # The output from the hidden layer flows into the output layer. hence the hidden outputs are multipled with the weights
        final_inputs = np.dot(np.transpose(hidden_outputs), self.weights_hidden_to_output) # signals into final output layer
        # The output from the output layer has not activation funciton (as per the requirement). Then its a simple regression. Input just 
        # flows through to the output.
        final_outputs = final_inputs # signals from final output layer

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #print("Inside backprop")
        #### Implement the backward pass hervalue with your calculations.
        # Output layer error is the difference between desired target and actual output.
        error = (y-final_outputs)        
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        # Since the output layer does not have a sigmoid activation, the sigmod is not differenciated to get the deltaE.
        # Thus the output_error_term is same as the error itself. If the activation is sigmoid, then the derivative of 
        # the sigmoid function has to scaled in.
        output_error_term = error 
        #print("output_error_term: ",  output_error_term)
        
        # TODO: Calculate the hidden layer's contribution to the error
        # Hidden layer error as back propagated from the output layer, then the error propagates from the output.
        # So the hidden layer error  is the weight term or output_error_term
        hidden_error = output_error_term*np.transpose(self.weights_hidden_to_output)
        
        # Since the hidden layer has sigmoid activation - the derivative of the sigmoid is scaled in to get the error_term.
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)        
     
        # The delta weights sums up all the weighted errors for all the input data points. This then scaled later by the 
        # number of data points to get the average error.
        delta_weights_i_h += np.multiply(np.transpose(np.matrix(X)), hidden_error_term)
        delta_weights_h_o += np.multiply(np.transpose(np.matrix(hidden_outputs)), output_error_term) 
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
            This is called once for every epoch.
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # The weights are then updated based on the learning rate, number or records and deltaE.
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step


    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 3500
learning_rate = 0.5
hidden_nodes = 20
output_nodes = 1
