# Instalar los paquetes necesarios
if (!requireNamespace("plumber", quietly = TRUE)) {
  install.packages("plumber")
}
if (!requireNamespace("tensorflow", quietly = TRUE)) {
  install.packages("tensorflow")
}
if (!requireNamespace("magick", quietly = TRUE)) {
  install.packages("magick")
}

library(plumber)
library(magick)
library(tensorflow)
library(keras)
library(jsonlite)

# Cargar el modelo preentrenado
model_path <- "modelo_objetos_peligrosos_v3_epoch_15.h5"  # Cambia por la ruta de tu modelo Keras
model <- keras::load_model_hdf5(model_path)

#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization")
  res$setHeader("Access-Control-Max-Age", "3600")
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  }
  plumber::forward()
}

#* Realiza predicción para un conjunto de imágenes
#* @post /predict_batch
#* @param files:[file] conjunto de archivos binarios en formato multipart
#* @serializer json
function(files) {
  results <- list()  # Lista para almacenar resultados de cada imagen
  
  for (i in seq_along(files)) {
    # Guardar la imagen temporalmente
    temp_file <- tempfile(fileext = ".jpg")  # Usa una extensión genérica
    writeBin(files[[i]], temp_file)
    
    # Leer y procesar la imagen con magick
    img <- tryCatch(
      {
        magick::image_read(temp_file)
      },
      error = function(e) {
        return(NULL)  # Si no se puede leer la imagen, continuar con las demás
      }
    )
    
    if (is.null(img)) {
      results[[i]] <- list(error = "No se pudo procesar la imagen")
      next
    }
    
    # Redimensionar la imagen
    img_resized <- magick::image_resize(img, "640x640")
    if (class(img_resized) != "magick-image") {
      stop("Error: La imagen redimensionada no es válida.")
    }
    
    # Convertir a RGB si es necesario
    img_resized <- magick::image_convert(img_resized, colorspace = "RGB")
    
    # Extraer datos de imagen
    img_data <- magick::image_data(img_resized, channels = "rgb")
    if (is.null(img_data)) {
      stop("Error: No se pudo extraer datos de la imagen.")
    }
    
    # Convertir a array numérico
    img_array <- as.integer(img_data) / 255.0  # Normalizar entre 0 y 1
    
    # Reestructurar dimensiones para el modelo
    img_array <- array(img_array, dim = c(640, 640, 3))
    img_array <- array_reshape(img_array, c(1, 640, 640, 3))  # Agregar dimensión batch
    
    
    
    # Realizar la predicción
    pred <- model %>% predict(img_array)
    print(pred)
    # Obtener la clase y probabilidad
    class_idx <- which.max(pred[1, ]) - 1  # Índice de la clase (R empieza en 1)
    probability <- pred[1, class_idx + 1]
    
    # Opcional: etiquetas de las clases
    class_labels <- c("cuchillo", "pistola", "seguro")  # Cambia según tu modelo
    predicted_class <- class_labels[class_idx + 1]
    
    # Guardar resultado
    results[[i]] <- list(
      predicted_class = predicted_class,
      probability = probability
    )
  }
  
  response <- list(
    message = "Predicciones realizadas",
    results = results
  )
  

  # Convertir la respuesta a JSON con auto_unbox
  response_json <- toJSON(response, auto_unbox = TRUE, pretty = TRUE)
  response_list <- fromJSON(response_json)
  # Devolver directamente el JSON serializado
  return(response_list)
}