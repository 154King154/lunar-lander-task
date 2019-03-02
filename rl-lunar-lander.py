
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.saved_model import tag_constants
import numpy as np
#import matplotlib.pyplot as plt

try:
    xrange = xrange
except:
    xrange = range

env = gym.make('LunarLander-v2')

gamma = 0.99  # коэффициент дисконтирования


def discount_rewards(r):
    """ принимая на вход вектор выигришей,
    вернуть вектор дисконтированных выигрышей"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # Ниже инициализирована feed-forward часть нейросети.
        # Агент оценивает состояние среды и совершает действие
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size,
                                      biases_initializer=None, activation_fn=tf.nn.relu)
        hidden_2 = slim.fully_connected(hidden, h_size,
                                      biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden_2, a_size,
                                           activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)  # выбор действия

        # Следующие 6 строк устанавливают процедуру обучения.
        # Нейросеть принимает на вход выбранное действие
        # и соответствующий выигрыш,
        # чтобы оценить функцию потерь и обновить веса модели.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0,
                                tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                             self.indexes)
        # функция потерь
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) *
                                    self.reward_holder)

        tvars = tf.trainable_variables()
        self.exported = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,
                                                          tvars))


tf.reset_default_graph()  # Очищаем граф tensorflow

myAgent = agent(lr=1e-2, s_size=8, a_size=4, h_size=64)  # Инициализируем агента
saver = tf.train.Saver()


total_episodes = 5000  # Количество итераций обучения
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Запуск графа tensorflow
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "/home/alex/PycharmProjects/weights/model.ckpt")
    print("Model restored.")
    i = 0
    total_reward = []
    total_lenght = []

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while i < total_episodes:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # Выбрать действие на основе вероятностей, оцененных нейросетью
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)


            if i % 100 == 0: env.render()


            s1, r, d, _ = env.step(a)  # Получить награду за совершенное действие
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            if d == True:
                # Обновить нейросеть
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders,
                                                      gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_lenght.append(j)
                break

            # Обновить общий выигрыш
        print("---- Reward in", i, "session:", running_reward)
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]), ' --- Progress: ', i*100/total_episodes, '%')


        if i % 1000 == 0:
            save_path = saver.save(sess, "/home/alex/PycharmProjects/weights/model.ckpt")
            print("Model saved in path: %s" % save_path)
            print('At: ', i, ' from ', total_episodes)
        i += 1



    save_path = saver.save(sess, "/home/alex/PycharmProjects/weights/model.ckpt")
    print("Model saved in path: %s" % save_path)
