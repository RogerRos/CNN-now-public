from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("/Users/rogerrosvidal/Desktop/Cotxes-Motos/IA(arquitectura,pesos)/model_IA-0.3")

img_path = "/Users/rogerrosvidal/Desktop/Cotxes-Motos/GUI(codi visualització)/FOTO"
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255

prediction = model.predict(img_array)
if prediction[0][0] < 0.5:
    print("-------------------------------------------------------")
    print("--------------------[ es un cotxe ]--------------------")
    print("-------------------------------------------------------")
else:
    print("-------------------------------------------------------")
    print("--------------------[ es una moto ]--------------------")
    print("-------------------------------------------------------")
