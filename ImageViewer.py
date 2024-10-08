from flask import Flask, render_template_string, send_file
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
image_list = []

# Template for displaying images
html_template = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Image Viewer</title>
    <style>
      img {
        display: block;
        margin-bottom: 20px;
        width: 100%;  /* Adjust width as needed */
        max-width: 800px;  /* Adjust max width as needed */
      }
    </style>
  </head>
  <body>
    <div>
      {% for img in images %}
        <img src="{{ url_for('get_image', index=loop.index0) }}" alt="Image">
      {% endfor %}
    </div>
  </body>
</html>
'''

def numpy_to_pil_image(np_image):
    return Image.fromarray((np_image * 255).astype('uint8'))

def update_images(np_images):
    global image_list
    image_list = [numpy_to_pil_image(img) for img in np_images]

@app.route('/')
def index():
    return render_template_string(html_template, images=image_list)

@app.route('/image/<int:index>')
def get_image(index):
    if index < 0 or index >= len(image_list):
        return "Image not found", 404
    pil_image = image_list[index]
    img_io = BytesIO()
    pil_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')




if __name__ == '__main__':
    # Example usage: update_images with some sample numpy images
    sample_images = [np.random.rand(100, 100, 3) for _ in range(5)]
    update_images(sample_images)
    app.run(debug=True)
