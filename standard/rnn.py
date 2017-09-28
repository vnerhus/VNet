import tensorflow as tf
import numpy as np
import textprocessing as tp
import time
import sys
from select import select

## Standard LSTM, built for the purpose of text generation. ##
class lstm():

    ## Class initialization ##
    def __init__(self, text):

        # Hyperparameters stored in dictionary
        self.hyperparameters = {
                        "num_epochs": 200,   # Number of training epochs
                        "learning_rate": 0.001, # Learning rate
                        "batch_size": 256,      # Size of each batch
                        "sequence_length": 32,  # Length of sequence
                        "embed_dim": 256,       # Embedding dimension size
                        "lstm_size": 256,       # Lstm size
                        "stack_size": 1,        # Number of stacked LSTM-cells
                        }
        self.token_dict, self.vocab_to_int, self.int_to_vocab, self.int_text = tp.preprocess(text)
        self.refreshHyperparameters()

        print("\n\n\n -- VNET V0.1 INITIALIZED -- ")

    ## Training of model ##
    def train(self, load=False, displayFrequency=8):
        print("\n ------------------\n | ## TRAINING ## |\n ------------------\n")
        print("  Hyperparameters:")
        [print("  {} : {}".format(value, key)) for value, key in self.hyperparameters.items()]
        print("\n")
        save_dir = './save'
        train_graph = tf.Graph()
        with train_graph.as_default():
            self.vocab_size = len(self.int_to_vocab)
            input_text, targets, lr = self.get_inputs()
            input_data_shape = tf.shape(input_text)
            cell, initial_state = self.get_init_cell()
            logits, final_state = self.build_nn(cell, input_text)

            # Probabilities for generating words
            probs = tf.nn.softmax(logits, name='probs')

            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([input_data_shape[0], input_data_shape[1]]))

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

        batches = self.get_batches(self.int_text)

        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            print("\n\n")
            stop = False
            for epoch_i in range(self.num_epochs):
                if(stop == True):
                    break
                state = sess.run(initial_state, {input_text: batches[0][0]})

                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        input_text: x,
                        targets: y,
                        initial_state: state,
                        lr: self.learning_rate}
                    train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                    # Show every <show_every_n_batches> batches
                    if(epoch_i * len(batches) + batch_i) % displayFrequency == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))

                if(epoch_i%5==0):
                    print("Press Enter to stop or wait 10 seconds...")
                    timeout = 10
                    rlist, _, _ = select([sys.stdin], [], [], timeout)

                    if(rlist):
                        print("Stopping...")
                        stop = True
                    else:
                        print("Timed out, continuing...")

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            print('Model Trained and Saved\n')

    ## Generation from select prime words ##
    def generate(self, prime_words, random=False, index=0, gen_length=200):
        save_dir = './save'
        loaded_graph = tf.Graph()

        if(random):
            prime_word = prime_words[random.randint(0, len(prime_words)-1)]
        else:
            prime_word = prime_words[index]

        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(save_dir + '.meta')
            loader.restore(sess, save_dir)

            # Get Tensors from loaded model
            input_text, initial_state, final_state, probs = self.get_tensors(loaded_graph)

            # Sentences generation setup
            gen_sentences = [prime_word + '']
            prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

            # Generate sentences
            for n in range(gen_length):
                # Dynamic Input
                dyn_input = [[self.vocab_to_int[word] for word in gen_sentences[-self.sequence_length:]]]
                dyn_seq_length = len(dyn_input[0])

                # Get Prediction
                probabilities, prev_state = sess.run([probs, final_state], {input_text: dyn_input, initial_state: prev_state})

                pred_word = self.pick_word(probabilities[dyn_seq_length-1], self.int_to_vocab)

                gen_sentences.append(pred_word)

            # Remove tokens
            tv_script = ' '.join(gen_sentences)
            for key, token in token_dict.items():
                ending = ' ' if key in ['\n', '(', '"'] else ''
                tv_script = tv_script.replace(' ' + token.lower(), key)
            tv_script = tv_script.replace('\n ', '\n')
            tv_script = tv_script.replace('( ', '(')

            print(tv_script)

    def setHyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.refreshHyperparameters()
        pass

    def refreshHyperparameters(self):
        self.num_epochs = self.hyperparameters["num_epochs"]
        self.learning_rate = self.hyperparameters["learning_rate"]
        self.batch_size = self.hyperparameters["batch_size"]
        self.sequence_length = self.hyperparameters["sequence_length"]
        self.embed_dim = self.hyperparameters["embed_dim"]
        self.lstm_size = self.hyperparameters["lstm_size"]
        self.stack_size = self.hyperparameters["stack_size"]

    def get_inputs(self):
        x = tf.placeholder(tf.int32, (None, None), name="input")
        y = tf.placeholder(tf.int32, (None, None), name="targets")
        learning_rate = tf.placeholder(tf.float32)

        return x, y, learning_rate

    def get_init_cell(self):
        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        lstmStack = tf.contrib.rnn.MultiRNNCell([lstmCell]*self.stack_size)
        initial_state = lstmStack.zero_state(self.batch_size, tf.float32)
        initial_state = tf.identity(initial_state, 'initial_state')
        return lstmStack, initial_state

    def get_embed(self, input_data):
        embedding = tf.Variable(tf.random_uniform((self.vocab_size, self.embed_dim), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, input_data)
        return embed

    def build_rnn(self, cell, embed):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
        final_state = tf.identity(final_state, 'final_state')
        return outputs, final_state

    def build_nn(self, cell, input_data):
        embed = self.get_embed(input_data)
        outputs, final_state = self.build_rnn(cell, embed)
        logits = tf.contrib.layers.fully_connected(outputs, self.vocab_size, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32))
        return logits, final_state

    def get_batches(self, int_text):
        n_batches = len(int_text)//(self.batch_size*self.sequence_length)
        xdata = np.array(int_text[: n_batches * self.batch_size * self.sequence_length])
        ydata = np.array(int_text[1: n_batches * self.batch_size * self.sequence_length + 1])
        x_batches = np.split(xdata.reshape(self.batch_size, -1), n_batches, 1)
        y_batches = np.split(ydata.reshape(self.batch_size, -1), n_batches, 1)
        return np.array(list(zip(x_batches, y_batches)))

    def get_tensors(self, loaded_graph):
        """
        Gets input, initial_state, final_state, and probabilities tensor from <loaded_graph>
        :param loaded_graph: TensorFlow graph loaded from file.
        :return Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
        """

        InputTensor = loaded_graph.get_tensor_by_name("input:0")
        InitialStateTensor= loaded_graph.get_tensor_by_name("initial_state:0")
        FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
        ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")
        return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor

    def pick_word(self, probabilities, int_to_vocab):
        """
        Picks the next word in the generated text.
        :param probabilities: Probabilities of the next word.
        :param int_to_vocab: Dictionary of word ids as the keys and words as the values.
        :return: String of the predicted word.
        """

        return np.random.choice(list(int_to_vocab.values()), p=probabilities)
