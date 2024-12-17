library(reticulate)
use_virtualenv("D:/Cursos/2024-II/Big Data - Tareas/Evidencia 1/cnns02env")
library(keras)
library(ggplot2)

img_width <- 640
img_height <- 640
epochs <- 20
num_classes <- 3
train_dir <- "dataset/train"
validation_dir <- "dataset/validation"
class_names = c("cuchillo", "pistola", "seguro")

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
  batch_size = 20,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = c(img_width, img_height),
  batch_size = 20,
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

# Lista para almacenar el historial de cada bloque
full_history <- list()

# Entrenamiento en bloques de 5 épocas
for (i in seq(1, epochs, by = 5)) {
  # Entrena por 5 épocas
  history <- model %>% fit(
    train_generator,
    steps_per_epoch = as.integer(train_generator$samples / train_generator$batch_size),
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = as.integer(validation_generator$samples / validation_generator$batch_size)
  )
  
  # Guarda las métricas del bloque actual en la lista
  full_history[[paste0("block_", i)]] <- history
  
  # Guarda el modelo al final del bloque
  save_model_hdf5(model, filepath = paste0("modelo_objetos_peligrosos_v3_epoch_", i+4, ".h5"))
  
  # Limpia memoria de tensores temporales
  gc()
}

# Al final del entrenamiento, puedes combinar o exportar full_history
saveRDS(full_history, file = "training_history_v3.rds")


#EVALUACION DEL MODELO
full_history <- readRDS("training_history_v3.rds")

losses <- unlist(lapply(full_history, function(h) h$metrics$loss))
val_losses <- unlist(lapply(full_history, function(h) h$metrics$val_loss))
accuracies <- unlist(lapply(full_history, function(h) h$metrics$accuracy))
val_accuracies <- unlist(lapply(full_history, function(h) h$metrics$val_accuracy))

# Crear un data frame para visualización
training_progress <- data.frame(
  epoch = seq_along(losses),
  loss = losses,
  val_loss = val_losses,
  accuracy = accuracies,
  val_accuracy = val_accuracies
)

# Gráfico de pérdida
ggplot(training_progress, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Entrenamiento")) +
  geom_line(aes(y = val_loss, color = "Validación")) +
  labs(title = "Pérdida durante el entrenamiento", y = "Pérdida", x = "Época") +
  scale_color_manual(values = c("Entrenamiento" = "blue", "Validación" = "red"), name="Dataset") +
  theme_minimal()

# Gráfico de precisión
ggplot(training_progress, aes(x = epoch)) +
  geom_line(aes(y = accuracy, color = "Entrenamiento")) +
  geom_line(aes(y = val_accuracy, color = "Validación")) +
  labs(title = "Precisión durante el entrenamiento", y = "Precisión", x = "Época") +
  scale_color_manual(values = c("Entrenamiento" = "blue", "Validación" = "red"), name="Dataset") +
  theme_minimal()


