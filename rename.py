import os

def rename_files_in_folder(folder_path):
    """
    Renombra todos los archivos en una carpeta, manteniendo la extensión,
    asignándoles nombres numéricos según el orden en que se encuentran.
    
    Args:
        folder_path (str): Ruta de la carpeta donde están los archivos.
    """
    try:
        # Obtén la lista de archivos en la carpeta
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        # Ordena los archivos alfabéticamente (opcional, puedes eliminarlo)
        files.sort()

        # Renombrar archivos
        for index, file_name in enumerate(files, start=1):
            old_path = os.path.join(folder_path, file_name)
            # Obtén la extensión del archivo
            file_extension = os.path.splitext(file_name)[1]
            # Nuevo nombre con formato numérico
            new_name = f"{index}{file_extension}"
            new_path = os.path.join(folder_path, new_name)
            
            # Renombrar el archivo
            os.rename(old_path, new_path)
            print(f"Renombrado: {file_name} -> {new_name}")

        print("\nTodos los archivos han sido renombrados correctamente.")
    
    except Exception as e:
        print(f"Error: {e}")

# Ruta de la carpeta que contiene los archivos
folder_path = r"D:\Cursos\2024-II\Big Data - Tareas\Evidencia 1\dataset\test"  # Cambia esta ruta

# Llama a la función
rename_files_in_folder(folder_path)