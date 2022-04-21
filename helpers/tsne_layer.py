import tensorflow as tf


class TSNE_Layer():
    def __init__(self, inc, outc, P):
        self.w = tf.Variable(tf.random_poisson(shape = [inc, outc], dtype = tf.float16, lam = 0.05, seed = 1))
        self.P = P
        self.m =  tf.Variable(tf.zeros_like(self.w))
        self.v = tf.Variable(tf.zeros_like(self.w))

    def get_weights(self):
        return self.w

    def tf_q_tsne(self, Y):
        distances = self.tf_neg_distance(Y)
        inv_distances = tf.pow(1. - distances, -1)
        inv_distances = tf.matrix_set_diag(inv_distances,
                                           tf.zeros([inv_distances.shape[0].value],
                                            dtype = tf.float16))
        return inv_distances / tf.reduce_sum(inv_distances), inv_distances

    def tf_tsne_grad(self, P, Q, W, inv):
        pq_diff = P - Q
        pq_expanded = tf.expand_dims(pq_diff, 2)
        y_diffs = tf.expand_dims(W, 1) - tf.expand_dims(W, 0)
        distances_expanded = tf.expand_dims(inv, 2)
        y_diffs_wt = y_diffs * distances_expanded
        grad = 4 * tf.reduce_sum(pq_expanded * y_diffs_wt, 1)
        return grad


    def feedforward(self):
        self.Q, self.inv_distances = self.tf_q_tsne(self.w)
        return self.Q

    def backprop(self):
        grad = self.tf_tsne_grad(self.P, self.Q, self.w, self.inv_distances)
        update_w = []
        update_w.append(tf.assign(self.m, self.m*beta1 + (1- beta1)*grad))
        update_w.append(tf.assign(self.v, self.v*beta2 + (1 - beta2) * (grad**2)))
        m_hat = self.m / (1 - beta1)
        v_hat = self.v / (1 - beta2)
        adam_middel = learning_rate / tf.sqrt()
