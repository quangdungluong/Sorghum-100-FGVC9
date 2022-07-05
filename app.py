"""Flask app"""
import os
import pandas as pd

import torch
from flask import Flask, render_template, request, send_from_directory

from src.config import CFG
from src.data_preparation import create_labels_map
from src.main import predict, process_image
from src.model import SorghumModel

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
UPLOAD_FOLDER = 'uploads'

df = pd.read_csv('data/train_cultivar_mapping.csv')
labels_map = create_labels_map()
model = SorghumModel(CFG.backbone, CFG.embedding_size, CFG.num_classes)
model.load_state_dict(torch.load(CFG.model_path, map_location="cpu"))
model.eval()


def allowed_file(filename):
    """Check allow file"""
    return filename.split('.')[-1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    """Home"""
    return render_template('index.html', label='Hiiii', ground_truth='',
                            imagesource='/static/img/thumb.png', returnJson={})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Upload and process file"""
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image = process_image(
                img_path=file_path, transforms=CFG.data_transforms, display=False)
            output, probs = predict(model, image, labels_map)
        ## Get Ground truth label
        ground_truth = "None"
        name_path = file.filename.split('.')[0] + '.png'
        if name_path in df['image'].values:
            index = df[df['image'] == name_path].index
            ground_truth = df.iloc[index, 1].values[0]
        return_json = {}
        for i, prob in enumerate(probs):
            return_json[labels_map[i]] = round(prob*100,3)
        return_json = dict(sorted(return_json.items(), key=lambda item: item[1], reverse=True)[:6])
    return render_template('index.html', label=output, ground_truth=ground_truth,
                            imagesource=file_path, returnJson=return_json)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Save uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
