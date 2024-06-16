# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import random
import jieba
# from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout
import tensorflow as tf
# from tensorflow.keras.layers import Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
# 确保使用GPU进行训练
physical_devices = tf.config.list_physical_devices('GPU')
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
cut_text = ''
for word in jieba.cut(text):
    cut_text += word
    cut_text += ' '

sentences = re.split(r'[。！？]', cut_text)
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

# 创建输入和标签
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# 参数定义
embed_dim = 128        # 嵌入维度
num_heads = 8          # 多头注意力的头数
ff_dim = 512           # 前馈网络的维度
dropout_rate = 0.1     # Dropout的概率

# 定义Transformer的编码器层
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 定义Transformer模型
class TransformerModel(tf.keras.models.Model):
    def __init__(self, max_sequence_len, total_words, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=total_words, output_dim=embed_dim, input_length=max_sequence_len-1)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dense1 = tf.keras.layers.Dense(20, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dense2 = tf.keras.layers.Dense(total_words, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.transformer_block(x, training=training)  # 这里需要显式传递training参数
        x = self.global_avg_pool(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.dense2(x)

# 创建模型实例
model = TransformerModel(max_sequence_len, total_words, embed_dim, num_heads, ff_dim, dropout_rate)


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
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        output_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + output_word
    return re.sub(r'\s+', '', seed_text)

# 定义种子文本列表
seed_texts = ["阿清竹棒一动，对手若不是手腕被戳，长剑脱手，便是","一名吴士兴犹未尽，长剑一挥，将一头山羊从头至臀","后来勾践不听文种、范蠡劝谏，兴兵和吴国交战，以石买为将，在钱塘江边","不料青衣剑士竟不挡架闪避，手腕抖动，噗的一声，","这时场中两名青衣剑士仍以守势缠住了一名锦衫剑士，另外","到第四天上，范蠡再要找她去会斗越国剑士时"]

# 自定义回调函数，在每个epoch结束时生成文本并保存
class TextGenerationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        seed_text = random.choice(seed_texts)
        generated_text = generate_text(seed_text, 40, max_sequence_len)
        os.makedirs('./train', exist_ok=True)
        with open(os.path.join('./train', f'generated_text_epoch_{epoch + 1}.txt'), 'w', encoding='utf-8') as file:
            file.write(generated_text)

# 训练模型并保存历史记录
history = model.fit(xs, ys, 32 ,epochs=1000, verbose=1, callbacks=[TextGenerationCallback()])

# 保存模型
model.save("jy_nlp_trans_model.keras")

# 绘制训练准确率的图像
def plot_accuracy(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练准确率')
    plt.legend()
    plt.grid(True)
    os.makedirs('./train', exist_ok=True)
    plt.savefig('./train/training_accuracy.png')
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
    os.makedirs('./train', exist_ok=True)
    plt.savefig('./train/training_loss.png')
    # plt.show()

# 调用绘制函数
plot_accuracy(history)
plot_loss(history)