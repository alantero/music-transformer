import os
import pretty_midi
import concurrent.futures
from tqdm import tqdm
import pickle
import math


guitar_possibilities = ["guitar","petrucci", "Hard Rock", "classic clean", "kirk", "james", "rythim", "solo"]
bass_posibilities = ["bass", "myung"]

def process_midi_file(file_path):
    # Try to load the midi file
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
    except Exception:
        # If loading fails, return empty sets
        #print("Fil not Loaded", file_path)
        return set(), set()
    
    guitar_variants = set()
    bass_variants = set()
    
    for instrument in pm.instruments:
        if instrument.name:
            name_lower = instrument.name.lower()

            if any(keyword in name_lower for keyword in guitar_possibilities) and "bass guitar" not in name_lower:
                guitar_variants.add(instrument.name)
            # Verificar si alguna palabra clave de bajo está en el nombre
            if any(keyword in name_lower for keyword in bass_posibilities):
                bass_variants.add(instrument.name)


    return guitar_variants, bass_variants

def extract_variants(dataset_dir):
    # Collect all midi file paths
    file_paths = []
#    for d in dataset_dir:
#        for root, _, files in os.walk(d):#dataset_dir):
#            for file in files:
#                if file.lower().endswith(('.mid', '.midi')):
#                    file_paths.append(os.path.join(root, file))
 

    for d in dataset_dir:
        if d == "lmd_full/":
        


            total_midi_files = sum(1 for root, _, files in os.walk(d) 
                                   for file in files if file.lower().endswith(('.mid', '.midi')))

            print("Hola", total_midi_files)
            # Paso 2: Calcular cuántos archivos representan el 10%
            num_files_to_select = math.ceil(total_midi_files * 1)
            
            # Paso 3: Seleccionar solo el número necesario de archivos
            selected_files = []
            for root, _, files in os.walk(d):
                for file in files:
                    if file.lower().endswith(('.mid', '.midi')):
                        selected_files.append(os.path.join(root, file))
                        # Detenerse cuando se alcance el 10%
                        if len(selected_files) >= num_files_to_select:
                            break
                if len(selected_files) >= num_files_to_select:
                    break
            file_paths.extend(selected_files)

        else:
            # Para otros directorios, procesar todos los archivos MIDI como antes
            for root, _, files in os.walk(d):
                print(root)
                for file in files:
                    if file.lower().endswith(('.mid', '.midi')):
                        file_paths.append(os.path.join(root, file))


    guitar_variants = set()
    bass_variants = set()
   

    print(len(file_paths))
    # Process files in parallel and show progress with tqdm
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_midi_file, file_paths), total=len(file_paths)))
        for g_variants, b_variants in results:
            guitar_variants.update(g_variants)
            bass_variants.update(b_variants)
    
    return guitar_variants, bass_variants

if __name__ == '__main__':
    dataset_dir = ["/Users/agus/repositories/GuitarPro-to-Midi/midi","lmd_full/"]  # Set your dataset directory here
    #dataset_dir = ["lmd_full/"]  # Set your dataset directory here
    #dataset_dir = ["/Users/agus/repositories/GuitarPro-to-Midi/midi"]  # Set your dataset directory here
    guitar_vars, bass_vars = extract_variants(dataset_dir)
    
    print("Guitar Variants:")
    print(guitar_vars)
    print("Bass Variants:")
    print(bass_vars)
    with open('metal_variants.pkl', 'wb') as f:
        pickle.dump({'guitar_vars': guitar_vars, 'bass_vars': bass_vars}, f)
