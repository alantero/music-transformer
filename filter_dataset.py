import os
import shutil
import pretty_midi
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pickle
import csv
import re
import math

with open('metal_variants.pkl', 'rb') as f:
    variants = pickle.load(f)

guitar_variants = variants['guitar_vars']
bass_variants = variants['bass_vars']

#print(guitar_variants)

def file_has_instruments(filepath):
    """
    Returns True if the MIDI file at filepath contains at least one instrument
    whose name matches a guitar keyword and at least one matching a bass keyword.
    """
    try:
        pm = pretty_midi.PrettyMIDI(filepath)
    except Exception:
        return False

    found_guitar = False
    found_bass = False

    for inst in pm.instruments:
        if inst.name:
            name = inst.name.lower()
            if any(keyword in name for keyword in guitar_variants):
                found_guitar = True
            if any(keyword in name for keyword in bass_variants):
                found_bass = True
            if found_guitar and found_bass:
                return True
    return False

def get_unique_filepaths(dataset_dir):
    """
    Recursively traverse dataset_dir and return a list of unique MIDI file paths,
    where uniqueness is determined by the file's basename.
    for d in dataset_dir:
        for root, _, files in os.walk(d):#dataset_dir):
            for file in files:
                total = 0
                if file.lower().endswith(('.mid', '.midi')):
                    if file not in unique_files:
                        unique_files[file] = os.path.join(root, file)
                    total += 1

    """


    unique_files = {}
    
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
                        unique_files[file] = os.path.join(root, file)
                        if len(selected_files) >= num_files_to_select:
                            break
                if len(selected_files) >= num_files_to_select:
                    break
        else:
            # Para otros directorios, procesar todos los archivos MIDI como antes
            for root, _, files in os.walk(d):
                for file in files:
                    if file.lower().endswith(('.mid', '.midi')):
                        #unique_files.append(os.path.join(root, file))
                        unique_files[file] = os.path.join(root, file)

    return list(unique_files.values())



#def get_unique_filepaths(dataset_dir):
#    """
#    Recursively traverse dataset_dir and return a list of unique MIDI file paths.
#    Uniqueness is determined by a normalized basename that ignores trailing indexes.
#    """
#    unique_files = {}
#    for root, _, files in os.walk(dataset_dir):
#        for file in files:
#            if file.lower().endswith(('.mid', '.midi')):
#                norm = normalize_filename(file)
#                if norm not in unique_files:
#                    unique_files[norm] = os.path.join(root, file)
#    return list(unique_files.values())


def normalize_filename(filename):
    """
    Remove a trailing subscript from the file's basename.
    It removes an optional underscore or hyphen followed by digits from the basename.
    """
    base, ext = os.path.splitext(filename)
    normalized_base = re.sub(r'[-]?\d+$()', '', base)
    return normalized_base + ext.lower()

def filter_dataset_parallel(dataset_dir, output_csv, num_workers=8):
    """
    Traverse dataset_dir, filter unique MIDI files by checking if they contain both
    guitar and bass instruments, and save their full paths in a CSV file.
    """
    # Get unique file paths (based on filename)
    unique_filepaths = get_unique_filepaths(dataset_dir)#[0:10]
    total = len(unique_filepaths)
    print("Total Unique Files:", total)
    valid_files = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the check function over the file list in parallel and show progress
        results = list(tqdm(
            executor.map(file_has_instruments, unique_filepaths),
            total=total,
            desc="Filtering MIDI files"
        ))
    
    # Collect files that passed the filter
    for fp, is_valid in zip(unique_filepaths, results):
        if is_valid:
            valid_files.append(fp)
    
    # Save the valid file paths in a CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath"])  # header
        for fp in valid_files:
            writer.writerow([fp])
    
    print(f"Saved {len(valid_files)} valid MIDI file paths to {output_csv}")

if __name__ == '__main__':
    dataset_dir = ["/Users/agus/repositories/GuitarPro-to-Midi/midi","lmd_full/"]         # Directorio del dataset original
    output_csv = "valid_metal_full_files.csv"  # CSV de salida con los paths validos
    filter_dataset_parallel(dataset_dir, output_csv, num_workers=8)



