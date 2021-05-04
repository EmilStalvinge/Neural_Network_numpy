# Neural_Network_numpy
A Neural network from scratch with numpy. 



\section{Mathematical exercises}

The chosen loss function Li for a multi-output classification problem is the cross-entropy loss which, for
numerical issues when z  0, is computed together with the softmax function. The cost J, for an M-class problem, is computed by summing the loss Li over all the training points xi

\begin{equation}
    L_i = ln(\sum_{l=1}^M e^{z_{il}}) - \sum_{m=1}^M \tilde{y}_{im}z_{im}
\end{equation}

\begin{equation}
J = \frac{1}{n}\sum_{i=1}^n L_i
\end{equation}

\begin{equation}
  \tilde{y}_{im}=\begin{cases}
    1, & \text{if $y_i=m$}.\\
    0, & \text{if $y_i \ne m$}.
  \end{cases}
  \text{      for m = 1,...,M}
\end{equation}

\textit{\textbf{Exercise 1.1}}

The derivative of the weights with respect to the cost for softmax activation combined with cross-entropy loss in eq.1

\begin{equation}
    \frac{\delta J}{\delta w_{mj}} = \frac{\delta J}{\delta z_{im}} \frac{\delta z_{im}}{\delta w_{mj}}
\end{equation}

\begin{equation}
\frac{\delta J}{\delta z_{im}} = \frac{1}{\sum_{l=1}^M e^{z_{il}}} e^{z_{il}}  - \tilde{y}_{im} \\
=\begin{cases}
    \frac{e^{z_{il}}}{\sum_{l=1}^M e^{z_{il}}} -1, & \text{if $y_i = m$}.\\
    \frac{e^{z_{il}}}{\sum_{l=1}^M e^{z_{il}}}, & \text{if $y_i \ne m$}.
  \end{cases}
\end{equation}

\begin{equation}
\frac{\delta z_{im}}{\delta w_{mj}} = x_i
\end{equation}


The derivative of the bias with respect to the cost for softmax activation combined with cross-entropy loss in eq.1

\begin{equation}
    \frac{\delta J}{\delta b_{m}} = \frac{\delta J}{\delta z_{im}} \frac{\delta z_{im}}{\delta b_{m}}
\end{equation}

\begin{equation}
\frac{\delta z_{im}}{\delta b_{m}} = 1
\end{equation}



\newpage


\section{Code exercises}
\textit{\textbf{Exercise 1.2}}


\subsection{initialize the fully connected layer (dense layer)}



\begin{figure}[H]
 \centering
    \includegraphics[width=0.6\textwidth]{img/dense.png}
    \caption{dense layer}
    \label{}
\end{figure}

For a whole batch the input matrix $X$ is one sample per row $X_{sample, feature}$
 
\begin{equation}  
\begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \end{bmatrix}
\end{equation}

The weight matrix $w$ is transposed to have one output neuron per column $w_{output, input}$
 
\begin{equation}  
\begin{bmatrix} w_{11} & w_{21} \\ w_{12} & w_{21} \\ w_{13} & w_{23} \end{bmatrix}
\end{equation}

\subsubsection{Forward function}

The outputs $z$ is calculated in the forward function with matrix multiplication as 
 
$$X*w+b = z$$

Each row of z corresponds to one sample, the number of rows is equal to the batch size


\begin{equation} 
\begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \end{bmatrix}
\begin{bmatrix} w_{11} & w_{21} \\ w_{12} & w_{21} \\ w_{13} & w_{23} \end{bmatrix} + 
\begin{bmatrix} b_1 & b_2 \\ b_1 & b_2 \end{bmatrix} = 
\begin{bmatrix} x_{11}w_{11}+x_{12}w_{12}+x_{13}w_{13} + b_1
                & x_{11}w_{21}+x_{12}w_{22}+x_{13}w_{23} + b_2 \\
                  x_{21}w_{11}+x_{22}w_{12}+x_{23}w_{13} + b_1
                & x_{21}w_{21}+x_{22}w_{22}+x_{23}w_{23} + b_2 
\end{bmatrix} \\

\text{= np.dot(inputs, self.weights) + self.biases}
\end{equation}


\subsubsection{Backward function}

The derivative the weights respect the cost function 

$$ \frac{\delta J}{\delta w_i} = \frac{\delta J}{\delta z_i} \frac{\delta z_i}{\delta w_{ij}}  = \frac{\delta J}{\delta z_i}  X_j$$

where $\frac{\delta J}{\delta z_i}$ is the gradient that is comming from the right

And the derivative of a weight $w$ respect to the $z$ $\frac{\delta z_i}{\delta w_{ij}}$  is just the corresponding $X$ to that weight

\begin{equation} 
\frac{\delta J}{\delta w_i} = X^T \frac{\delta J}{\delta z_i}
\end{equation}

\begin{equation}
\begin{bmatrix} x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23}  \end{bmatrix} ^T
\begin{bmatrix} dz_{11} & dz_{12} \\ dz_{21} & dz_{22} \\ dz_{31} & dz_{32} \end{bmatrix} =
\begin{bmatrix} x_{11}dz_{11}+x_{21}dz_{21}
                & x_{11}dz_{12}+x_{12}dz_{22} \\
                  x_{12}dz_{11}+x_{12}dz_{12}
                & x_{12}dz_{12}+x_{22}dz_{22} 
\end{bmatrix} \\

\text{= np.dot(self.inputs.T, d\_outputs)}
\end{equation}

The derivative of the biases respect the cost function 

$$ \frac{\delta J}{\delta b} = \frac{\delta J}{\delta z_i}  \frac{\delta z_i}{\delta b}  = \frac{\delta J}{\delta z_i}  1$$

where $\frac{\delta J}{\delta z_i}$ is the gradient that is comming from the right

And the derivative of a weight $b$ respect to the $z$ $\frac{\delta z_i}{\delta w_{ij}}$  is just 1

\begin{equation}
\begin{bmatrix} 1 & 1 \end{bmatrix} 
\begin{bmatrix} dz_{11} & dz_{12} \\ dz_{21} & dz_{22}\end{bmatrix} =
\begin{bmatrix} dz_{11}+dz_{21} & dz_{12}dz_{22}\end{bmatrix} \\

\text{= np.sum(d\_outputs, axis=0)}
\end{equation}

\textit{\textbf{In the code:}} d\_outputs = $\frac{\delta J}{\delta z_i}$ and inputs = X

\begin{minted}[frame=lines,framesep=2mm,linenos]{python}
class Dense_Layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.01 * np.random.randn(n_inputs,n_outputs) #or rand?
        self.biases = np.zeros((1, n_outputs))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases # b is added to each row (broadcasting)
    def backward(self, d_outputs):
        self.dweights = np.dot(self.inputs.T, d_outputs)
        self.dbiases = np.sum(d_outputs, axis=0, keepdims=True)
        self.d_inputs = np.dot(d_outputs, self.weights.T) 
\end{minted}








\subsection{ReLU activation function)}

\subsubsection{Forward}

\begin{figure}[H]
 \centering
    \includegraphics[width=0.6\textwidth]{img/relu.png}
    \caption{relu}
    \label{}
\end{figure}

\subsubsection{Backward}

The drivative of the ReLU is 0 if the input was less then 0:   $\text{self.d\_inputs[self.inputs <= 0] = 0}$

else it's just 1 (the same as the output):  $\text{self.d\_inputs = d_outputs.copy()}$

\begin{minted}[frame=lines,framesep=2mm,linenos]{python}
class ReLU_Activation:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self, d_outputs):
        self.d_inputs = d_outputs.copy()
        self.d_inputs[self.inputs <= 0] = 0
\end{minted}

\newpage


\subsection{Sigmoid activation function)}

\subsubsection{Forward}

\begin{figure}[H]
 \centering
    \includegraphics[width=0.6\textwidth]{img/sigmoid.png}
    \caption{sigmoid}
    \label{}
\end{figure}

\subsubsection{Backward}



The derivative of the sigmoid is $$\frac{d\sigma}{dx} = \sigma(x)(1-\sigma(x))$$

The derivative with respect to the cost function:
$$ \frac{\delta J}{\delta x} = \frac{\delta J}{\delta \sigma}  \frac{d\sigma}{dx} = \frac{\delta J}{\delta \sigma} \sigma(x)(1-\sigma(x))$$

in the code $\frac{\delta J}{\delta \sigma}$ =  d\_outputs

$$\text{self.d\_inputs = d\_outputs * (1 - self.output) * self.output}$$

\begin{minted}[frame=lines,framesep=2mm,linenos]{python}
class Sigmoid_Activation:
    # Forward pass
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        # Backward pass
    def backward(self, d_outputs):
        # Derivative - calculates from output of the sigmoid function
        self.d_inputs = d_outputs * (1 - self.output) * self.output
\end{minted}





\subsection{Softmax activation function)}

\subsubsection{Forward}

The softmax activation is to make the outputs into probilities between 0 and 1. The Sum of the probablities is also 1.

\begin{figure}[H]
 \centering
    \includegraphics[width=0.4\textwidth]{img/softmax.png}
    \caption{softmax}
    \label{}
\end{figure}


\begin{minted}[frame=lines,framesep=2mm,linenos]{python}
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # minus det största värdet för numerical stability
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities  
\end{minted}

\newpage




\subsection{Cross-entropy loss (log loss)}

\subsubsection{Forward}

$$ L =  -ln(g_y)$$ 

where $g_y$ is the softmax output for the prediction value of the true label

Cross entropy loss is used to get smooth curve with a high loss for a bad prediction.

\begin{figure}[H]
 \centering
    \includegraphics[width=0.6\textwidth]{img/cross_entropy.png}
    \caption{cross-entropy}
    \label{}
\end{figure}


\begin{minted}[frame=lines,framesep=2mm,linenos]{python}
class Crossentropy_Loss:      
    def forward(self, y_pred, y_true):
                
        # Number of samples in a batch
        samples = len(y_pred)
        prob_for_true = y_pred[range(samples),y_true]
        
        # Losses
        negative_log_likelihoods = -np.log(prob_for_true)
        return negative_log_likelihoods

    def batch_loss(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        batch_loss = np.mean(sample_losses)
        # Return loss
        return batch_loss
\end{minted}

\newpage







\subsection{Combined Softmax and cross-entropy loss derivation}

together they form a much simpler derivative that is faster to compute

\begin{figure}[H]
 \centering
    \includegraphics[width=1\textwidth]{img/softmax_crossentropy.png}
    \caption{softmax and cross-entropy}
    \label{}
\end{figure}

\begin{figure}[H]
 \centering
    \includegraphics[width=0.6\textwidth]{img/softmax_crossentropy_derivation.png}
    \caption{softmax and cross-entropy derivation}
    \label{}
\end{figure}


\begin{minted}[frame=lines,framesep=2mm,linenos]{python}
class Activation_Softmax_Loss_CategoricalCrossentropy:    
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Crossentropy_Loss()
        # Forward pass
    def forward(self, inputs, y_true):
        # Output layer’s activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.batch_loss(self.output, y_true)

    def backward(self, d_outputs, y_true):
        # Number of samples
        samples = len(d_outputs)
        # Copy so we can safely modify
        self.d_inputs = d_outputs.copy()
        
        # Calculate gradient
        self.d_inputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples
\end{minted}
