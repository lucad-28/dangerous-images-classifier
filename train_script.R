library(reticulate)
# Replace with your virtual environment path
use_virtualenv("D:/Cursos/2024-II/Big Data - Tareas/Evidencia 1/sqlanaenv")
library(keras)

img_width <- 640
img_height <- 640
epochs <- 20
num_classes <- 2
train_dir <- "dataset/train"
validation_dir <- "dataset/validation"
class_names = c("cuchillos", "pistolas")

train_datagen <- image_data_generator(
  rescale = 1/255,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(img_width, img_height),
  batch_size = 32,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = c(img_width, img_height),
  batch_size = 32,
  class_mode = "categorical"
)

input <- layer_input(shape = c(img_width, img_height, 3))

output <- input %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

model <- keras_model(inputs = input, outputs = output)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  train_generator,
  steps_per_epoch = as.integer(train_generator$samples / train_generator$batch_size),
  epochs = epochs,
  validation_data = validation_generator,
  validation_steps = as.integer(validation_generator$samples / validation_generator$batch_size)
)

model %>% save_model_hdf5("modelo_objetos_peligrosos.h5")
scores <- model %>% evaluate(validation_generator)
cat('Precisión en validación:', scores[2], "\n")
