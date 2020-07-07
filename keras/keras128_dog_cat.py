from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

img_dog = load_img('./data/dog_cat/dog.jpg', target_size = (224, 224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size = (224, 224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size = (224, 224))
img_yang = load_img('./data/dog_cat/yang.jpg', target_size = (224, 224))

plt.imshow(img_yang)
plt.imshow(img_dog)
# plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

print("dog :", arr_dog)
print("dog type : ",type(arr_dog))  # dog type :  <class 'numpy.ndarray'>
print("dog.shape :", arr_dog.shape) # dog.shape : (224, 224, 3)

