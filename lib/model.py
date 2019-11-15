# inbuilt lib imports:
from typing import Dict
import math

# external libs
import tensorflow as tf
from tensorflow.keras import models, layers

# project imports


class CubicActivation(layers.Layer):
    """
    Cubic activation as described in the paper.
    """

    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        vector : ``tf.Tensor``
            hidden vector of dimension (batch_size, hidden_dim)

        Returns tensor after applying the activation
        """
        # TODO(Students) Start

        return tf.pow(vector, 3)
        # raise NotImplementedError
        # TODO(Students) End


class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        # Trainable Variables
        # TODO(Students) Start
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.num_transitions = num_transitions

        # Initiliation between (-1e2, 1e2) as mentioned in paper.
        self.embeddings = tf.Variable(
            tf.random.uniform(
                [self.vocab_size, self.embedding_dim],
                minval=-0.01, maxval=0.01, dtype=tf.float32
            ),
            trainable=trainable_embeddings
        )

        # weight_hidden = hidden_dim, num_tokens*embedding_dim
        self.weight_hidden = tf.Variable(
            tf.compat.v2.random.normal(
                # [self.hidden_dim, self.num_tokens * self.embedding_dim],
                [self.num_tokens * self.embedding_dim, self.hidden_dim],
                mean=0,
                stddev=1.0 / math.sqrt(self.num_transitions)
            ),
            trainable=trainable_embeddings
        )

        # num_transitions , hidden_dim
        self.weight_output = tf.Variable(
            tf.compat.v2.random.normal(
                [self.hidden_dim, self.num_transitions, ],
                mean=0,
                stddev=1.0 / math.sqrt(self.num_transitions)
            ),
            trainable=trainable_embeddings
        )

        # hidden_dim, 1
        self.bias = tf.Variable(
            tf.zeros([self.hidden_dim, 1]),
            trainable=trainable_embeddings
        )

        # hidden_dim, 2
        '''
        self.biases2 = tf.Variable(
            tf.zeros([self.hidden_dim, 1])
        )
        '''

        # TODO(Students) End


    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start
        embeddings_inputs = tf.nn.embedding_lookup(self.embeddings, inputs)
        embeddings_inputs = tf.reshape(
            embeddings_inputs, [
                embeddings_inputs.shape[0],
                embeddings_inputs.shape[1] * embeddings_inputs.shape[2]
            ]
        )

        # hidden_layer = tf.add(tf.matmul(self.weight_hidden, tf.transpose(embeddings_inputs) ), self.biases)

        hidden_layer = tf.add(
            tf.matmul(embeddings_inputs, self.weight_hidden),
            # tf.matmul(self.weight_hidden, embeddings_inputs),
            tf.transpose(self.bias)
        )

        logits = self._activation(hidden_layer)
        logits = tf.matmul(logits, self.weight_output)

        # TODO(Students) End
        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict


    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:
        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """
        # TODO(Students) Start

        '''
        logits -= tf.reduce_max(logits, 1, keepdims=True)
        logits = tf.exp(logits)

        logits = tf.where(
            # labels>-1,
            tf.equal(labels, -1)
            logits,
            tf.equal(labels, -1),
            tf.zeros_like(logits),
        )

        sum_logits = tf.reduce_sum(
            logits,
            keepdims=True,
            axis=1
        )

        logits = tf.math.divide_no_nan(logits, sum_logits)

        # logits = tf.where(
        #     tf.equal(logits, 0),
        #     tf.zeros_like(logits),
        #     tf.math.log(tf.clip_by_value(logits, 1e-10, 1.0))
        # )

        loss = (-1) * tf.reduce_sum(
            tf.multiply(labels, logits),
            axis=1
        )

        # loss = tf.math.log(ce_loss)

        '''

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits( 
            logits=logits,
            labels=tf.argmax(tf.transpose(labels))
        )
        # loss = tf.nn.softmax_cross_entropy_with_logits((labels >- 1) * labels, logits)

        regularised_loss = tf.nn.l2_loss(self.weight_hidden) + \
            tf.nn.l2_loss(self.weight_output) + \
            tf.nn.l2_loss(self.bias) + \
            tf.nn.l2_loss(self.embeddings)
        
        # +  tf.nn.l2_loss(self.biases2)+
        regularization = (self._regularization_lambda) * regularised_loss

        # TODO(Students) End

        return tf.reduce_mean(tf.add(loss, regularization))
