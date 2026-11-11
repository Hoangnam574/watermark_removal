import os
import sys
import uuid

print("Bat dau import cac thu vien...")

try:
    from flask import Flask, render_template, request, jsonify, send_file
    print("✓ Flask OK")
except ImportError as e:
    print(f"Loi import Flask: {e}")
    print("Chay: pip install flask")
    sys.exit(1)

try:
    from PIL import Image
    print("✓ Pillow OK")
except ImportError as e:
    print(f"Loi import Pillow: {e}")
    print("Chay: pip install Pillow")
    sys.exit(1)

try:
    import cv2
    print("✓ OpenCV OK")
except ImportError as e:
    print(f"Loi import OpenCV: {e}")
    print("Chay: pip install opencv-python")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"✓ TensorFlow OK (version {tf.__version__})")
except ImportError as e:
    print(f"Loi import TensorFlow: {e}")
    print("Chay: pip install tensorflow==1.15.0")
    sys.exit(1)

try:
    import neuralgym as ng
    print("✓ Neuralgym OK")
except ImportError as e:
    print(f"Loi import Neuralgym: {e}")
    print("Chay: pip install git+https://github.com/JiahuiYu/neuralgym")
    sys.exit(1)

import numpy as np
from preprocess_image import preprocess_image
from inpaint_model import InpaintCAModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['CHECKPOINT_DIR'] = 'model/'

# Tao thu muc neu chua ton tai
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Load config
FLAGS = ng.Config('inpaint.yml')

def remove_watermark(image_path, watermark_type, output_path):
    """Xu ly xoa watermark tu anh"""
    model = InpaintCAModel()
    image = Image.open(image_path)
    input_image = preprocess_image(image, watermark_type)
    
    tf.reset_default_graph()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    
    if input_image.shape != (0,):
        with tf.Session(config=sess_config) as sess:
            input_image_tf = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image_tf)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            
            # Load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(
                    app.config['CHECKPOINT_DIR'], from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            
            result = sess.run(output)
            cv2.imwrite(output_path, cv2.cvtColor(result[0][:, :, ::-1], cv2.COLOR_BGR2RGB))
            return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Khong tim thay file'}), 400
    
    file = request.files['file']
    watermark_type = request.form.get('watermark_type', 'istock')
    
    if file.filename == '':
        return jsonify({'error': 'Chua chon file'}), 400
    
    if file:
        # Tao ten file unique
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        input_filename = f"{file_id}_input{ext}"
        output_filename = f"{file_id}_output.png"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        file.save(input_path)
        
        try:
            success = remove_watermark(input_path, watermark_type, output_path)
            if success:
                return jsonify({
                    'success': True,
                    'original': f'/static/uploads/{input_filename}',
                    'processed': f'/static/outputs/{output_filename}',
                    'download': f'/download/{output_filename}'
                })
            else:
                return jsonify({'error': 'Khong the xu ly anh'}), 500
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(path, as_attachment=True, download_name='watermark_removed.png')

if __name__ == '__main__':
    print("=" * 50)
    print("Server dang khoi dong...")
    print("Truy cap: http://localhost:5000")
    print("Nhan Ctrl+C de dung server")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
if __name__ == '__main__':
    print("=" * 50)
    print("Server dang khoi dong...")
    print("Truy cap: http://localhost:5000")
    print("Hoac: http://127.0.0.1:5000")
    print("Nhan Ctrl+C de dung server")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
