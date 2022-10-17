# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# %% Model 불러오기
model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()
# %% Class labels 불러오기
labels = {}
with open("imagenet1000_clsidx_to_labels.txt") as f:
    for line in f:
        (key, val) = line.split(sep=":")
        labels[int(key)] = val[2:-3]
# %% Image 불러오기
IMAGE_PATH = '9.jpg'
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224,224))
img = tf.keras.preprocessing.image.img_to_array(img)
data = ([img],None)
index = np.argmax(model.predict(np.array([img])))
plt.imshow(img/255)
plt.axis('off')
plt.show()
# %% GradCAM
from tf_explain.core.grad_cam import GradCAM
method = "GradCAM"
explainer = GradCAM()
grid = explainer.explain(data, model, layer_name='block5_conv3', class_index=index)
# %% Integrated Gradients
from tf_explain.core.integrated_gradients import IntegratedGradients
method = "Integrated Gradients"
explainer = IntegratedGradients()
grid_temp = explainer.explain(data, model, class_index=index)
grid = np.array([grid_temp.T,grid_temp.T,grid_temp.T]).T
# %% Vanilla Gradients
from tf_explain.core.vanilla_gradients import VanillaGradients
method = "Vanilla Gradients"
explainer = VanillaGradients()
grid_temp = explainer.explain(data, model, class_index=index)
grid = np.array([grid_temp.T,grid_temp.T,grid_temp.T]).T
# %% Gradients*Inputs
from tf_explain.core.gradients_inputs import GradientsInputs
method = "Gradients*Inputs"
explainer = GradientsInputs()
grid_temp = explainer.explain(data, model, class_index=index)
grid = np.array([grid_temp.T,grid_temp.T,grid_temp.T]).T
# %% SmoothGrad
from tf_explain.core.smoothgrad import SmoothGrad
method = "SmoothGrad"
explainer = SmoothGrad()
grid_temp = explainer.explain(data, model, class_index=index)
grid = np.array([grid_temp.T,grid_temp.T,grid_temp.T]).T
# %% Save the XAI
explainer.save(grid, '.', 'explained.png')
# %% Prediction
plt.imshow(np.append(img, grid, axis=1)/255)
plt.axis('off')
plt.show()
print("Predicted :" , labels[index], "\n" + 
    "XAI Method:", method)
# %%
