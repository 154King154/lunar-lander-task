import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.saved_model import tag_constants
import numpy as np

# Указываем environment для gym
env = gym.make('LunarLander-v2')

# Папка с файлами имеющими расширение .ckpt.meta или .ckpt.*
# Указание имени файла необходимо завершить на расширении .ckpt
WEIGHTS_PATH="/home/alex/PycharmProjects/weights/weights_learned_bck/model.ckpt"



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
        # Сам выбор действия
        self.chosen_action = tf.argmax(self.output, 1)

        # Следующие строки устанавливают процедуру обучения.
        # Нейросеть принимает на вход выбранное действие
        # и соответствующий выигрыш,
        # чтобы оценить функцию потерь и обновить веса модели.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0,
                                tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                             self.indexes)
        # Объявим значение (функцию) loss для потерь и расхождений
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
saver = tf.train.Saver() # Инициализируем модуль для импорта/экспорта весов (встр. tensorflow)


max_ep = 999        # Количество эпизодов
winned_eps=0        # Счетчик выигранных эпизодов
PREVENT_DEF=0
init = tf.global_variables_initializer()
total_reward = []
i = 1

with tf.Session() as sess:
    sess.run(init)
    # Попытка загрузить файл, если есть
    try:
        saver.restore(sess, WEIGHTS_PATH)
    except:
        sess.run(init)
        print("Model is empty. Stop testing empty model.")
    else:
        print("Model successfully restored.")

    print('Testing trained agent.')
    while i < 101:
        s = env.reset()
        running_reward = 0
        ep_history = []
        for j in range(max_ep):
            # Выбрать действие на основе вероятностей, оцененных нейросетью
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            if PREVENT_DEF > 0:
                s1, r, d, _ = env.step(0)  # Получить награду за совершенное действие
                PREVENT_DEF -= 1
                a = 0
            else:
                s1, r, d, _ = env.step(a)  # Получить награду за совершенное действие
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r

            #if i % 5 == 0: env.render() #Раскоментировать, чтобы видеть рендер каждой пятой игры
            if int(s[6]) == int(s[7]) == 1: PREVENT_DEF = 10
            if d == True:
                total_reward.append(running_reward)
                winned_eps += 1
                PREVENT_DEF = 0
                break


        print(np.mean(total_reward[-winned_eps:]), ' --- Progress: ', i,
                  '% --- Done param: ', len(total_reward), ' from ', i, 'episodes (', winned_eps, ' winned)')
        winned_eps = 0

        i += 1

    print("----- Average points in 100 episodes: ", np.mean(total_reward))
