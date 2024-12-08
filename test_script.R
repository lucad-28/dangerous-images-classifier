### Prueba

#model <- load_model_hdf5("modelo_objetos_peligrosos.h5")
#
#test_dir <- "dataset/test"
#test_files <- list.files(test_dir, full.names = TRUE)
#
#for (file_path in test_files) {
#  # Preprocesar la imagen
#  test_image <- image_load(path = file_path, target_size = c(img_width, img_height))
#  test_image_array <- image_to_array(test_image) / 255
#  test_image_array <- array_reshape(test_image_array, c(1, img_width, img_height, 3))
#  
#  # Realizar la predicción
#  prediction <- model %>% predict(test_image_array)
#  predicted_class <- which.max(prediction) - 1
#  
#  # Mostrar resultados
#  cat("Imagen:", basename(file_path), " - Clase predicha:", class_names[predicted_class + 1], "\n")
#  cat(prediction, "\n")
#}


model <- load_model_hdf5("modelo_objetos_peligrosos.h5")

test_generator <- flow_images_from_directory(
  directory = "dataset/test",
  generator = image_data_generator(rescale = 1/255),
  target_size = c(img_width, img_height),
  batch_size = 32,
  class_mode = "categorical",
  shuffle = FALSE
)

scores <- model %>% evaluate(test_generator)

cat("Pérdida en prueba:", scores[1], "\n")
cat("Precisión en prueba:", scores[2], "\n")
