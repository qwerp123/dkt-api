import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import traceback

# 禁用CUDA以强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.backends.cudnn.enabled = False

app = Flask(__name__)


def load_configs(model_dir):
    config_paths = {
        "model_config": "model_config.json",
        "train_config": "train_config.json",
        "dataset_info": "dataset_info.json"
    }

    configs = {}
    for key, filename in config_paths.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path) as f:
            configs[key] = json.load(f)

    return configs


class DKT(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout(0.2)

    def forward(self, q, r):
        # 确保所有索引都在有效范围内
        q = torch.clamp(q, 0, self.num_q - 1)
        r = torch.clamp(r, 0, 1)

        # 计算交互索引
        x = q + self.num_q * r
        x = torch.clamp(x, 0, self.num_q * 2 - 1)

        emb = self.interaction_emb(x)
        h, _ = self.lstm_layer(emb)
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        return torch.sigmoid(y)


def process_sequences(question_seq, answer_seq, seq_len, num_q):
    # 验证并清理输入序列
    cleaned_q = []
    cleaned_a = []

    for q, a in zip(question_seq, answer_seq):
        # 确保问题ID在有效范围内
        if isinstance(q, int) and 0 <= q < num_q:
            cleaned_q.append(q)
            # 确保答案为0或1
            cleaned_a.append(1 if a else 0)

    # 截断超长序列
    if len(cleaned_q) > seq_len:
        cleaned_q = cleaned_q[-seq_len:]
        cleaned_a = cleaned_a[-seq_len:]

    # 短序列填充
    if len(cleaned_q) < seq_len:
        pad_len = seq_len - len(cleaned_q)
        cleaned_q = [0] * pad_len + cleaned_q
        cleaned_a = [0] * pad_len + cleaned_a

    return cleaned_q, cleaned_a


def load_model(model_dir):
    device = torch.device("cpu")  # 强制使用CPU

    # 加载配置
    configs = load_configs(model_dir)
    num_q = configs["dataset_info"]["num_q"]
    seq_len = configs["train_config"]["seq_len"]

    # 初始化模型
    model = DKT(
        num_q=num_q,
        emb_size=configs["model_config"]["emb_size"],
        hidden_size=configs["model_config"]["hidden_size"]
    ).to(device)

    # 加载权重
    model_path = os.path.join(model_dir, "model.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 确保在CPU上加载模型
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    print(f"模型加载成功! 知识点数量: {num_q}, 序列长度: {seq_len}")
    return model, seq_len, num_q


# 类型转换函数 - 解决JSON序列化问题
def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


# 初始化模型
model_dir = "ckpts/dkt1"
model, seq_len, num_q = load_model(model_dir)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型未加载'}), 500


    data = request.json
    student_id = data.get('student_id', 'unknown')
    question_seq = data.get('question_seq', [])
    answer_seq = data.get('answer_seq', [])

    # 验证输入
    if not question_seq or not answer_seq:
        return jsonify({'error': '缺少问题序列或答案序列'}), 400

    if len(question_seq) != len(answer_seq):
        return jsonify({'error': '问题序列和答案序列长度不一致'}), 400

    # 处理序列
    processed_q, processed_a = process_sequences(
        question_seq, answer_seq, seq_len, num_q
    )

    # 转换为张量
    q_tensor = torch.tensor([processed_q], dtype=torch.long)
    r_tensor = torch.tensor([processed_a], dtype=torch.long)

    # 模型预测
    with torch.no_grad():
        knowledge_state = model(q_tensor, r_tensor)

    # 获取结果并转换为可序列化格式
    last_state = knowledge_state[0, -1, :].cpu().numpy()
    mastery_level = np.mean(last_state)

    # 找出最弱的3个知识点
    weak_indices = np.argsort(last_state)[:3]
    weak_scores = [float(last_state[i]) for i in weak_indices]

    # 构建响应数据
    response_data = {
        'student_id': student_id,
        'knowledge_state': [float(x) for x in last_state],
        'mastery_level': float(mastery_level),
        'weak_points': [int(i) for i in weak_indices],
        'weak_scores': weak_scores,
        'processed_question_seq': processed_q,
        'processed_answer_seq': processed_a
    }

    # 确保所有数据都是可序列化的
    serializable_data = {k: convert_to_serializable(v) for k, v in response_data.items()}

    return jsonify(serializable_data)




@app.route('/')
def health_check():
    return jsonify({
        'status': '运行中' if model else '未加载模型',
        'model': 'DKT',
        'num_skills': num_q,
        'sequence_length': seq_len,
        'endpoints': {
            'GET /': '服务状态检查',
            'POST /predict': '预测知识掌握状态'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)