
"""
Copyright 2021 Aditya Gomatam.

This file is part of music-transformer (https://github.com/spectraldoy/music-transformer), my project to build and
train a Music Transformer. music-transformer is open-source software licensed under the terms of the GNU General
Public License v3.0. music-transformer is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. music-transformer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. A copy of this license can be found within the GitHub repository
for music-transformer, or at https://www.gnu.org/licenses/gpl-3.0.html.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from random import randint, sample
from sys import exit
from vocabulary import *
from tokenizer import *
import glob

import pretty_midi  # Reemplazamos mido por pretty_midi
from torch import LongTensor
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import numpy as np
import csv



with open('metal_variants.pkl', 'rb') as f:
    variants = pickle.load(f)

guitar_variants = variants['guitar_vars']
bass_variants = variants['bass_vars']


# Funciones auxiliares (sin cambios)
def merge_intervals(intervals, min_gap=0.01):
    """
    Fusiona intervalos superpuestos en segmentos continuos.
    Args:
        intervals (list of tuples): Lista de tuplas (start, end) con tiempos de inicio y fin.
        min_gap (float): Brecha mínima entre intervalos para considerarlos separados.
    Returns:
        Lista de tuplas (start, end) representando intervalos fusionados.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] - prev[1] <= min_gap:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def intersect_intervals(intervals1, intervals2):
    """
    Encuentra segmentos superpuestos entre dos listas de intervalos.
    Args:
        intervals1 (list of tuples): Intervalos de guitarra como (start, end).
        intervals2 (list of tuples): Intervalos de bajo como (start, end).
    Returns:
        Lista de tuplas (start, end) representando segmentos superpuestos.
    """
    intersections = []
    i, j = 0, 0
    while i < len(intervals1) and j < len(intervals2):
        a_start, a_end = intervals1[i]
        b_start, b_end = intervals2[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            intersections.append((start, end))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return intersections

# Funciones adaptadas para pretty_midi
def get_notes_from_track(instrument):
    """
    Extrae notas de un instrumento pretty_midi con tiempos en segundos.
    Args:
        instrument (pretty_midi.Instrument): Instrumento MIDI.
    Returns:
        Lista de tuplas (start, end, note, velocity) para cada nota.
    """
    notes = []
    for note in instrument.notes:
        notes.append((note.start, note.end, note.pitch, note.velocity))
    return notes

def get_note_intervals(instrument):
    """
    Obtiene intervalos de actividad de un instrumento pretty_midi.
    Args:
        instrument (pretty_midi.Instrument): Instrumento MIDI.
    Returns:
        Lista de tuplas (start, end) cuando hay notas activas.
    """
    return [(note.start, note.end) for note in instrument.notes]

def count_notes_in_interval(instrument, start, end):
    """
    Cuenta notas en un instrumento que se superponen con un intervalo.
    Args:
        instrument (pretty_midi.Instrument): Instrumento MIDI.
        start (float): Inicio del intervalo en segundos.
        end (float): Fin del intervalo en segundos.
    Returns:
        Número de notas que se superponen con [start, end).
    """
    return sum(1 for note in instrument.notes if note.start < end and note.end > start)

def create_filtered_track(instrument, start, end):
    """
    Crea un instrumento pretty_midi filtrado con notas dentro de un intervalo.
    Args:
        instrument (pretty_midi.Instrument): Instrumento original.
        start (float): Tiempo de inicio del intervalo en segundos.
        end (float): Tiempo de fin del intervalo en segundos.
    Returns:
        pretty_midi.Instrument: Instrumento filtrado con tiempos ajustados.
    filtered_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
    for note in instrument.notes:
        if note.start < end and note.end > start:
            clipped_start = max(note.start, start)
            clipped_end = min(note.end, end)
            filtered_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=clipped_start - start,  # Ajustar tiempos para empezar en 0
                end=clipped_end - start
            )
            filtered_instrument.notes.append(filtered_note)
    return filtered_instrument
    """
    filtered_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
    for note in instrument.notes:
        if note.start < end and note.end > start:  # Nota se solapa con la subventana
            # Ajustar tiempos relativos sin recortar
            filtered_start = max(0, note.start - start)  # No negativo
            filtered_end = min(end - start, note.end - start)  # No exceder subventana
            filtered_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=filtered_start,
                end=filtered_end
            )
            filtered_instrument.notes.append(filtered_note)
    return filtered_instrument



def get_instrument_tracks(midi_file, variants):
    """
    Identifica instrumentos de un tipo basado en nombres.
    Args:
        midi_file (pretty_midi.PrettyMIDI): Archivo MIDI.
        variants (list): Lista de cadenas para buscar en nombres de instrumentos.
    Returns:
        Lista de instrumentos coincidentes.
    """
    return [instr for instr in midi_file.instruments if any(variant in instr.name.lower() for variant in variants)]

def process_midi_file_old(filepath, min_seg_duration=0.1, max_seg_duration=5.0):
    """
    Procesa un archivo MIDI para extraer segmentos donde al menos una guitarra y un bajo tocan simultáneamente.
    Para cada segmento, selecciona la guitarra y el bajo con más notas y los tokeniza en:
    <guitar> <notas de guitarra> <sep> <bass> <notas de bajo>.

    Args:
        filepath (str): Ruta al archivo MIDI.
        min_seg_duration (float): Duración mínima (en segundos) para considerar un segmento.
        max_seg_duration (float): Duración máxima de las subventanas.
    Returns:
        Lista de tuplas: ((seg_start, seg_end), eventos_combinados, índices_combinados) para cada segmento válido.
        Retorna lista vacía si no hay actividad simultánea de guitarra y bajo.
    """
    token_sequences = []

    # Cargar el archivo MIDI con pretty_midi
    try:
        midi_file = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return token_sequences

    # Identificar instrumentos de guitarra y bajo
    guitar_instruments = get_instrument_tracks(midi_file, guitar_variants)
    bass_instruments = get_instrument_tracks(midi_file, bass_variants)

    if not guitar_instruments or not bass_instruments:
        return token_sequences

    # Obtener intervalos de actividad
    guitar_intervals = [merge_intervals(get_note_intervals(instr)) for instr in guitar_instruments]
    bass_intervals = [merge_intervals(get_note_intervals(instr)) for instr in bass_instruments]

    # Verificar si hay superposición entre alguna guitarra y algún bajo
    all_guitar_intervals = [interval for g_intervals in guitar_intervals for interval in g_intervals]
    all_bass_intervals = [interval for b_intervals in bass_intervals for interval in b_intervals]
    overlaps_any = intersect_intervals(sorted(all_guitar_intervals), sorted(all_bass_intervals))
    if not overlaps_any:
        return token_sequences

    # Procesar cada bajo y encontrar guitarras superpuestas
    for bass_instr in bass_instruments:
        bass_active = merge_intervals(get_note_intervals(bass_instr))
        if not bass_active:
            continue

        for guitar_instr in guitar_instruments:
            guitar_active = merge_intervals(get_note_intervals(guitar_instr))
            overlaps = intersect_intervals(guitar_active, bass_active)

            for seg_start, seg_end in overlaps:
                if seg_end - seg_start < min_seg_duration:
                    continue

                # Dividir el segmento en subventanas
                subseg_starts = list(range(int(seg_start), int(seg_end), int(max_seg_duration)))
                for sub_start in subseg_starts:
                    sub_start = seg_start
                    sub_end = seg_end
                    #sub_end = min(sub_start + max_seg_duration, seg_end)
                    #if sub_end - sub_start < min_seg_duration:
                    if sub_end - sub_start < min_seg_duration:
                        continue

                    # Encontrar instrumentos activos en esta subventana
                    active_guitars = [(instr, count_notes_in_interval(instr, sub_start, sub_end)) 
                                      for instr in guitar_instruments if count_notes_in_interval(instr, sub_start, sub_end) > 0]
                    active_basses = [(instr, count_notes_in_interval(instr, sub_start, sub_end)) 
                                     for instr in bass_instruments if count_notes_in_interval(instr, sub_start, sub_end) > 0]

                    if not active_guitars or not active_basses:
                        continue

                    # Seleccionar guitarra y bajo con más notas
                    selected_guitar = max(active_guitars, key=lambda x: x[1])[0]
                    selected_bass = max(active_basses, key=lambda x: x[1])[0]

                    # Crear instrumentos filtrados
                    guitar_filtered = create_filtered_track(selected_guitar, sub_start, sub_end)
                    bass_filtered = create_filtered_track(selected_bass, sub_start, sub_end)

                    # Crear objetos PrettyMIDI para tokenización
                    guitar_midi = pretty_midi.PrettyMIDI()
                    guitar_midi.instruments.append(guitar_filtered)
                    bass_midi = pretty_midi.PrettyMIDI()
                    bass_midi.instruments.append(bass_filtered)

                    # Tokenizar (asumiendo que midi_parser y events_to_indices están definidos)
                    g, guitar_events, _ = midi_parser(mid=guitar_midi, instrument="guitar")
                    b, bass_events, _ = midi_parser(mid=bass_midi, instrument="bass")


                    # Combinar eventos
                    #combined_events = ["<start>"] + guitar_events + ["<sep>"] + bass_events + ["<end>"]
                    combined_events = guitar_events + ["<sep>"] + bass_events
                    #print(combined_events)
                    combined_indices = events_to_indices(combined_events) 
                    token_sequences.append(combined_indices)

    return token_sequences




def process_midi_file_intermediate(filepath, min_seg_duration=0.1, max_seg_duration=8.0):
    """
    Procesa un archivo MIDI para extraer segmentos de duración max_seg_duration donde la guitarra y el bajo seleccionados
    (aquellos con el mayor número de notas en toda la canción) tienen actividad. Descarta segmentos sin notas de bajo.

    Args:
        filepath (str): Ruta al archivo MIDI.
        min_seg_duration (float): Duración mínima (en segundos) para considerar un segmento.
        max_seg_duration (float): Duración máxima de las ventanas.
    Returns:
        Lista de tuplas: ((seg_start, seg_end), eventos_combinados, índices_combinados) para cada segmento válido.
        Retorna lista vacía si no hay guitarras o bajos.
    """
    token_sequences = []

    # Cargar el archivo MIDI con pretty_midi
    try:
        midi_file = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return token_sequences

    # Identificar instrumentos de guitarra y bajo
    guitar_instruments = get_instrument_tracks(midi_file, guitar_variants)
    bass_instruments = get_instrument_tracks(midi_file, bass_variants)

    if not guitar_instruments or not bass_instruments:
        return token_sequences

    # Seleccionar la guitarra con más notas en toda la canción
    selected_guitar = max(guitar_instruments, key=lambda instr: len(instr.notes))
    
    # Seleccionar el bajo con más notas en toda la canción
    selected_bass = max(bass_instruments, key=lambda instr: len(instr.notes))

    # Obtener la duración total de la canción
    total_duration = midi_file.get_end_time()

    # Dividir en ventanas de max_seg_duration
    for start in range(0, int(total_duration), int(max_seg_duration)):
        end = min(start + max_seg_duration, total_duration)
        if end - start < min_seg_duration:
            continue

        # Verificar si hay notas de bajo en este segmento
        bass_notes_in_segment = [note for note in selected_bass.notes if note.start < end and note.end > start]
        if not bass_notes_in_segment:
            continue  # Descartar segmento sin notas de bajo

        # Filtrar notas de guitarra y bajo en este segmento
        guitar_filtered = create_filtered_track(selected_guitar, start, end)
        bass_filtered = create_filtered_track(selected_bass, start, end)

        # Crear objetos PrettyMIDI para tokenización
        guitar_midi = pretty_midi.PrettyMIDI()
        guitar_midi.instruments.append(guitar_filtered)
        bass_midi = pretty_midi.PrettyMIDI()
        bass_midi.instruments.append(bass_filtered)

        # Tokenizar (asumiendo que midi_parser y events_to_indices están definidos)
        _, guitar_events, _ = midi_parser(mid=guitar_midi, instrument="guitar")
        _, bass_events, _ = midi_parser(mid=bass_midi, instrument="bass")

        # Combinar eventos
        combined_events = guitar_events + ["<sep>"] + bass_events
        combined_indices = events_to_indices(combined_events)
        #token_sequences.append(((start, end), combined_events, combined_indices))
        token_sequences.append(combined_indices)
        print(len(combined_indices))
    return token_sequences



def process_midi_filei_semifinal(filepath, min_seg_duration=0.1, max_seg_duration=10.0, max_tokens=1000):
    """
    Procesa un archivo MIDI y genera secuencias de tokens para guitarra y bajo.

    Args:
        filepath (str): Ruta al archivo MIDI.
        min_seg_duration (float): Duración mínima (en segundos) para considerar un segmento.
        max_seg_duration (float): Duración máxima inicial de las ventanas (en segundos).
        max_tokens (int): Número máximo de tokens permitidos por secuencia.
    Returns:
        Lista de secuencias de índices de tokens.
    """
    token_sequences = []

    # Cargar el archivo MIDI con pretty_midi
    try:
        midi_file = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return token_sequences

    # Identificar instrumentos de guitarra y bajo
    guitar_instruments = get_instrument_tracks(midi_file, guitar_variants)
    bass_instruments = get_instrument_tracks(midi_file, bass_variants)

    if not guitar_instruments or not bass_instruments:
        return token_sequences

    # Seleccionar la guitarra y el bajo con más notas
    selected_guitar = max(guitar_instruments, key=lambda instr: len(instr.notes))
    selected_bass = max(bass_instruments, key=lambda instr: len(instr.notes))

    # Obtener la duración total de la canción
    total_duration = midi_file.get_end_time()

    # Función auxiliar para procesar un segmento y dividirlo si es necesario
    def process_segment(start, duration):
        end = min(start + duration, total_duration)
        if end - start < min_seg_duration:
            return []

        # Verificar si hay notas de bajo en este segmento
        bass_notes_in_segment = [note for note in selected_bass.notes if note.start < end and note.end > start]
        if not bass_notes_in_segment:
            return []

        # Filtrar notas de guitarra y bajo en este segmento
        guitar_filtered = create_filtered_track(selected_guitar, start, end)
        bass_filtered = create_filtered_track(selected_bass, start, end)

        # Crear objetos PrettyMIDI para tokenización
        guitar_midi = pretty_midi.PrettyMIDI()
        guitar_midi.instruments.append(guitar_filtered)
        bass_midi = pretty_midi.PrettyMIDI()
        bass_midi.instruments.append(bass_filtered)

        # Tokenizar
        _, guitar_events, _ = midi_parser(mid=guitar_midi, instrument="guitar")
        _, bass_events, _ = midi_parser(mid=bass_midi, instrument="bass")

        # Combinar eventos
        combined_events = guitar_events + ["<sep>"] + bass_events
        combined_indices = events_to_indices(combined_events)

        # Verificar longitud
        if len(combined_indices) > max_tokens:
            # Si es demasiado largo, dividir en dos subsegmentos
            half_duration = duration / 2
            sub_sequences = []
            sub_sequences.extend(process_segment(start, half_duration))
            #print(len(sub_sequences[-1]))
            sub_sequences.extend(process_segment(start + half_duration, half_duration))
            return sub_sequences
        else:
            return [combined_indices]

    # Dividir en ventanas de max_seg_duration
    for start in range(0, int(total_duration), int(max_seg_duration)):
        segment_sequences = process_segment(start, max_seg_duration)
        token_sequences.extend(segment_sequences)
        #print(len(segment_sequences[0]))
    return token_sequences



import pretty_midi
import math

def process_midi_file(filepath, min_seg_duration=0.1, max_seg_duration=10.0, max_tokens=1000):
    """
    Procesa un archivo MIDI y genera secuencias de tokens para guitarra y bajo.
    Asegura que al inicio las notas hayan comenzado y al final hayan terminado.

    Args:
        filepath (str): Ruta al archivo MIDI.
        min_seg_duration (float): Duración mínima (en segundos) para considerar un segmento.
        max_seg_duration (float): Duración máxima inicial de las ventanas (en segundos).
        max_tokens (int): Número máximo de tokens permitidos por secuencia.
    Returns:
        Lista de secuencias de índices de tokens.
    """
    token_sequences = []

    # Cargar el archivo MIDI con pretty_midi
    try:
        midi_file = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return token_sequences

    # Identificar instrumentos de guitarra y bajo
    guitar_instruments = get_instrument_tracks(midi_file, guitar_variants)
    bass_instruments = get_instrument_tracks(midi_file, bass_variants)

    if not guitar_instruments or not bass_instruments:
        return token_sequences

    # Seleccionar la guitarra y el bajo con más notas
    selected_guitar = max(guitar_instruments, key=lambda instr: len(instr.notes))
    selected_bass = max(bass_instruments, key=lambda instr: len(instr.notes))

    # Obtener la duración total de la canción
    total_duration = midi_file.get_end_time()

    # Función auxiliar para procesar un segmento
    def process_segment(start, duration):
        end = min(start + duration, total_duration)
        if end - start < min_seg_duration:
            return []

        # Filtrar notas de guitarra y bajo que hayan comenzado antes o en el inicio del segmento
        # y ajustar las que terminan después del segmento
        guitar_notes = [
            note for note in selected_guitar.notes 
            if note.start <= start and note.end > start
        ] + [
            note for note in selected_guitar.notes 
            if note.start > start and note.end <= end
        ]
        bass_notes = [
            note for note in selected_bass.notes 
            if note.start <= start and note.end > start
        ] + [
            note for note in selected_bass.notes 
            if note.start > start and note.end <= end
        ]

        # Verificar si hay notas válidas para ambos instrumentos
        if not guitar_notes or not bass_notes:
            return []

        # Crear instrumentos filtrados con notas ajustadas
        guitar_filtered = create_filtered_track(selected_guitar, start, end, guitar_notes)
        bass_filtered = create_filtered_track(selected_bass, start, end, bass_notes)

        # Crear objetos PrettyMIDI para tokenización
        guitar_midi = pretty_midi.PrettyMIDI()
        guitar_midi.instruments.append(guitar_filtered)
        bass_midi = pretty_midi.PrettyMIDI()
        bass_midi.instruments.append(bass_filtered)

        # Tokenizar
        _, guitar_events, _ = midi_parser(mid=guitar_midi, instrument="guitar")
        _, bass_events, _ = midi_parser(mid=bass_midi, instrument="bass")

        # Combinar eventos
        combined_events = guitar_events + ["<sep>"] + bass_events
        combined_indices = events_to_indices(combined_events)

        # Verificar longitud
        if len(combined_indices) > max_tokens:
            half_duration = duration / 2
            sub_sequences = []
            sub_sequences.extend(process_segment(start, half_duration))
            sub_sequences.extend(process_segment(start + half_duration, half_duration))
            return sub_sequences
        else:
            return [combined_indices]

    # Dividir en ventanas de max_seg_duration
    for start in range(0, int(total_duration), int(max_seg_duration)):
        segment_sequences = process_segment(start, max_seg_duration)
        token_sequences.extend(segment_sequences)

    return token_sequences

def create_filtered_track(instrument, start, end, notes):
    """
    Crea un instrumento filtrado ajustando notas al intervalo.
    - Si una nota comienza antes de 'start', ajusta su inicio a 'start'.
    - Si una nota termina después de 'end', la corta para que termine en 'end'.

    Args:
        instrument (pretty_midi.Instrument): Instrumento original.
        start (float): Tiempo de inicio del intervalo en segundos.
        end (float): Tiempo de fin del intervalo en segundos.
        notes (list): Lista de notas a ajustar.
    Returns:
        pretty_midi.Instrument: Instrumento filtrado con tiempos ajustados.
    """
    filtered_instrument = pretty_midi.Instrument(
        program=instrument.program, 
        is_drum=instrument.is_drum, 
        name=instrument.name
    )
    for note in notes:
        # Ajustar inicio y fin al intervalo
        adjusted_start = max(note.start, start)
        adjusted_end = min(note.end, end)
        
        # Ajustar tiempos relativos al inicio del segmento
        filtered_start = adjusted_start - start
        filtered_end = adjusted_end - start
        
        # Crear la nota ajustada
        filtered_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=filtered_start,
            end=filtered_end
        )
        filtered_instrument.notes.append(filtered_note)
    return filtered_instrument


# Resto del código (sin cambios)
def sample_end_data(seqs, lth, factor=6):
    data = []
    for seq in seqs:
        lower_bound = max(len(seq) - lth, 0)
        idx = randint(lower_bound, lower_bound + lth // factor)
        data.append(seq[idx:])
    return data

def sample_data(seqs, lth, factor=6):
    data = []
    for seq in seqs:
        length = randint(lth - lth // factor, lth + lth // factor)
        idx = randint(0, max(0, len(seq) - length))
        data.append(seq[idx:idx+length])
    return data

def aug(data, note_shifts=None, time_stretches=None, verbose=False):
    if note_shifts is None:
        note_shifts = torch.arange(-2, 3)
    if time_stretches is None:
        time_stretches = [1, 1.05, 1.1]
    if any([i <= 0 for i in time_stretches]):
        raise ValueError("time_stretches must all be positive")

    special_tokens = {guitar_token, sep_token, bass_token, start_token, end_token, pad_token}

    note_shifted_data = []
    for seq in data:
        for shift in note_shifts:
            _shift = shift.item() if hasattr(shift, 'item') else shift
            note_shifted_seq = []
            for idx in seq:
                if idx not in special_tokens:
                    if (0 < idx <= note_on_events and 0 < idx + _shift <= note_on_events) or \
                       (note_on_events < idx <= note_events and note_on_events < idx + _shift <= note_events):
                        note_shifted_seq.append(idx + _shift)
                    else:
                        note_shifted_seq.append(idx)
                else:
                    note_shifted_seq.append(idx)
            note_shifted_seq = torch.LongTensor(note_shifted_seq)
            note_shifted_data.append(note_shifted_seq)

    time_stretched_data = []
    for seq in note_shifted_data:
        for time_stretch in time_stretches:
            time_stretched_seq = []
            delta_time = 0
            for idx in seq:
                idx = idx.item() if isinstance(idx, torch.Tensor) else idx  # Asegurarse de que idx sea escalar
                if idx not in special_tokens and note_events < idx <= note_events + time_shift_events:
                    time = idx - (note_events - 1)
                    time = time.item() if isinstance(time, torch.Tensor) else time  # Asegurarse de que time sea escalar
                    time_stretch = time_stretch.item() if isinstance(time_stretch, torch.Tensor) else time_stretch  # Asegurarse de que time_stretch sea escalar
                    delta_time += round(time * DIV * time_stretch)
                else:
                    if delta_time > 0:
                        time_to_events(delta_time, index_list=time_stretched_seq)
                        delta_time = 0
                    time_stretched_seq.append(idx)
            time_stretched_seq = torch.LongTensor(time_stretched_seq)
            time_stretched_data.append(time_stretched_seq)


    aug_data = [F.pad(F.pad(seq, (1, 0), value=start_token), (0, 1), value=end_token) for seq in time_stretched_data]
    aug_data = torch.nn.utils.rnn.pad_sequence(aug_data, padding_value=pad_token).transpose(-1, -2)
    return aug_data

def randomly_sample_aug_data(aug_data, k, augs=25):
    random_indices = sample(range(len(aug_data) // augs), k=k)
    out = torch.cat(
        [aug_data[i * augs:i * augs + augs] for i in random_indices],
        dim=0
    )
    return out

def process_file(file, max_tokens, verbose):
    try:
        print(file) if verbose else None
        segments = process_midi_file(file, max_tokens=max_tokens)
        if segments:
            #print(segments)
            return segments
    except (OSError, ValueError, EOFError) as ex:
        print(f"{type(ex).__name__} was raised: {ex}")
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocessing.py",
        description="Preprocess MIDI files into single tensor for ML"
    )
    parser.add_argument("source", help="source directory of MIDI files to preprocess")
    parser.add_argument("destination", help="destination path at which to save preprocessed data as a single tensor, "
                                            "including filename and extension")
    parser.add_argument("length", help="approximate sequence length to cut data into (length will be randomly sampled)",
                        type=int)
    parser.add_argument("-a", "--from-augmented-data", help="flag to specify whether or not the source contains "
                                                            "already augmented data", action="store_true")
    parser.add_argument("-t", "--transpositions", help="list of pitch transpositions to make in data augmentation",
                        nargs="+", type=int)
    parser.add_argument("-s", "--time-stretches", help="list of stretches in time to make in data augmentation",
                        nargs="+", type=float)
    parser.add_argument("-v", "--verbose", help="verbose output flag", action="store_true")
    args = parser.parse_args()

    if args.source[-1] != "/":
        args.source += "/"
    if not os.path.isdir(args.source):
        print("Error: source must be an existing directory")
        exit(1)
    if os.path.isdir(args.destination):
        if args.destination[-1] != "/":
            args.destination += "/"
        args.destination += "gnershk.pt"
    elif not (args.destination.endswith(".pt") or args.destination.endswith(".pth")):
        args.destination += ".pt"
    args.length = int(args.length)

    DATA = []
    PATH = args.source

    if not args.from_augmented_data:
        print("Translating midi files to event vocabulary (NOTE: may take a while)...") if args.verbose else None
        #midi_files = list(glob.iglob(PATH + '**/*.mid*', recursive=True))#[0:10]
        with open("valid_metal_full_files.csv", newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            midi_files = [row["filepath"] for row in reader]
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_file, file, args.length, args.verbose) for file in midi_files]
            for future in tqdm(as_completed(futures), total=len(midi_files), desc="Processing MIDI files"):
                result= future.result()
                if result:
                    DATA.extend(result)
    
    print("Randomly sampling and cutting data to length...") if args.verbose else None
    #DATA = sample_data(DATA, lth=args.length) + sample_end_data(DATA, lth=args.length)
    #print(DATA[:5])
    print("Dataset length sequences: ", len(DATA))
    print("Done!") if args.verbose else None



    #print(f"DATA después de sample_data (primera secuencia): {DATA[0]}")

    if not args.from_augmented_data:
        DATA = [F.pad(F.pad(torch.tensor(seq), (1, 0), value=start_token), (0, 1), value=end_token) for seq in DATA]


        #print(f"DATA después de padding (primera secuencia): {DATA[0]}")  # Imprime después del acolchado

        DATA = torch.nn.utils.rnn.pad_sequence(DATA, padding_value=pad_token).transpose(-1, -2)

        #print(f"DATA final (primera secuencia): {indices_to_events(DATA[0])}")



    #for d in DATA[-10:]:
    #    print(d)
    #    print("len",len(d))

    print("Number of tokens: ", len(DATA[-1]))

    if args.from_augmented_data:
        print("Augmenting data (NOTE: may take even longer)...") if args.verbose else None
        DATA = aug(DATA, note_shifts=args.transpositions, time_stretches=args.time_stretches,
                   verbose=(args.verbose >= 2))
        print("Done!") if args.verbose else None
   
    
    
    DATA = DATA[torch.randperm(DATA.shape[0])]
    print("Saving...") if args.verbose else None
    torch.save(DATA, args.destination)
    print("Done! Dataset saved at: ", args.destination)



