# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import tensorflow as tf
import keras
# from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import random
import jieba

# 确保使用GPU进行训练
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 读取金庸小说的txt文件
data_path = "D:\\zongruntang\\nlp\\jyxstxtqj_downcc.com"
texts = []
if True:
    with open(os.path.join(data_path, '越女剑.txt'), 'r', encoding='ANSI') as file:
        texts.append(file.read())

# 合并所有文本
text = " ".join(texts)

# 去除英文字符和特殊标点符号，只保留中文字符和常见标点符号
text = re.sub(r'[a-zA-Z]', '', text)
text = re.sub(r'\s+', '', text)
text = re.sub(r'[^\u4e00-\u9fa5，。！？：“”《》‘’、\n]', '', text)
str = ''
for word in jieba.cut(text):
    str += word
    str += ' '

sentences = re.split(r'[。！？]', str)
sentences = [sentence for sentence in sentences if sentence]  # 去除空句子


# 分词和预处理
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# 创建输入序列和目标序列
input_sequences = []
for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    if len(token_list) > 0:  # 检查序列是否为空
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)


max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# # 检查是否有有效的输入序列
# if not input_sequences:
#     raise ValueError("没有有效的输入序列，请检查预处理步骤。")
#
# # 填充序列到固定长度
# max_sequence_len = 50  # 固定长度为50
# input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# 创建输入和标签
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# 构建seq2seq模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length=max_sequence_len-1),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 定义生成文本的函数
def generate_text(seed_text, next_words, max_sequence_len):
    cut__text = ''
    for _ in range(next_words):
        for word in jieba.cut(seed_text):
            cut__text += word
            cut__text += ' '
        token_list = tokenizer.texts_to_sequences([cut__text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        output_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + output_word
    return re.sub(r'\s+', '', seed_text)

# 定义种子文本列表
seed_texts = ["阿清竹棒一动，对手若不是手腕被戳，长剑脱手，便是","一名吴士兴犹未尽，长剑一挥，将一头山羊从头至臀","后来勾践不听文种、范蠡劝谏，兴兵和吴国交战，以石买为将，在钱塘江边","不料青衣剑士竟不挡架闪避，手腕抖动，噗的一声，","这时场中两名青衣剑士仍以守势缠住了一名锦衫剑士，另外","到第四天上，范蠡再要找她去会斗越国剑士时",""]

# 自定义回调函数，在每个epoch结束时生成文本并保存
class TextGenerationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        seed_text = random.choice(seed_texts)
        generated_text = generate_text(seed_text, 40, max_sequence_len)
        os.makedirs('./hzh', exist_ok=True)
        with open(os.path.join('./hzh', f'generated_text_epoch_{epoch + 1}.txt'), 'w', encoding='utf-8') as file:
            file.write(generated_text)

# 训练模型并保存历史记录
history = model.fit(xs, ys, epochs=100, verbose=1, callbacks=[TextGenerationCallback()])

# 保存模型
model.save("jy_nlp_seq2seq_model.keras")

# 绘制训练准确率的图像
def plot_accuracy(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练准确率')
    plt.legend()
    plt.grid(True)
    os.makedirs('./hzh', exist_ok=True)
    plt.savefig('./hzh/training_accuracy.png')
    # plt.show()

# 绘制训练损失的图像
def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.legend()
    plt.grid(True)
    os.makedirs('./hzh', exist_ok=True)
    plt.savefig('./hzh/training_loss.png')
    # plt.show()

# 调用绘制函数
plot_accuracy(history)
plot_loss(history)