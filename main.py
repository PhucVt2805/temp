import asyncio
from utils import split_sentence
from inference import ensemble_predict
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'Có cái đầu buồi'

def predict(models=[], text=''):
    base_model_names = []
    model_dirs = []
    model = ['distilbert/distilbert-base-multilingual-cased', 'huawei-noah/TinyBERT_General_4L_312D', 'sentence-transformers/all-MiniLM-L6-v2']
    dirs = dirs = ['results/'+ f for f in ['distilbert_distilbert-base-multilingual-cased', 'huawei-noah_TinyBERT_General_4L_312D', 'sentence-transformers_all-MiniLM-L6-v2']]
    if 'Distilbert' in models:
        base_model_names.append(model[0])
        model_dirs.append(dirs[0])
    if 'TinyBERT' in models:
        base_model_names.append(model[1])
        model_dirs.append(dirs[1])
    if 'MiniLM-L6' in models:
        base_model_names.append(model[2])
        model_dirs.append(dirs[2])
    print(f"Selected models: {base_model_names}, Directories: {model_dirs}")
    text_dict = asyncio.run(split_sentence(text))

    predictions, avg_probabilities = ensemble_predict(
        model_dirs=model_dirs,
        base_model_names=base_model_names,
        texts=text_dict['translated'],
        max_length=512,
        use_4bit_quantization=False,
        use_fl32=True
    )
    return (text_dict, predictions, avg_probabilities)

@app.route('/')
def home():
    return render_template(
        'index.html',
        navbar_height="10vh",
        horizontal_padding="4vw",
        button_radius="2em",
        title="CheckLLM - Nhận diện văn bản được tạo bằng AI"
    )

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()

    if not data:
        return jsonify({'status': 'Không nhận được dữ liệu'}), 400

    session['selected_models'] = data.get('model')
    session['input_text'] = data.get('text')
    session['mode'] = 'Toàn bộ văn bản' if data.get('mode') == 'All' else 'Từng đoạn'

    print(data)

    return jsonify({
        'status': 'Thành công'
    })

@app.route('/result')
def show_result():
    models = session.get('selected_models', [])
    text = session.get('input_text', '')
    mode = session.get('mode', '')
    text_dict, predictions, avg_probabilities = predict(models=models, text=text)
    return render_template('result.html',
        models=models,
        text=text,
        mode=mode,
        results={
            'text': text_dict['original'],
            'predictions': predictions,
            'avg_probabilities': avg_probabilities},
        title = 'Kết quả',
        navbar_height="10vh",
        horizontal_padding="4vw",
        button_radius="2em")

@app.route('/model')
def show_model():
    return render_template('model.html',
        title = "Thông tin về các mô hình",
        navbar_height="10vh",
        horizontal_padding="4vw",
        button_radius="2em")

@app.route('/dataset')
def show_dataset():
    return render_template('dataset.html',
        title = "Thông tin về tập dữ liệu",
        navbar_height="10vh",
        horizontal_padding="4vw",
        button_radius="2em")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999, debug=True)