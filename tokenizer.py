import pretty_midi
from vocabulary import *
from torch import LongTensor




def midi_parser(fname=None, mid=None, instrument=None):
    # Validar entrada: solo uno de fname o mid
    if not ((fname is None) ^ (mid is None)):
        raise ValueError("Input one of fname or mid, not both or neither")

    # Cargar archivo MIDI
    if fname is not None:
        try:
            mid = pretty_midi.PrettyMIDI(fname)
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")

    # Inicializar listas y variables
    event_list = []
    index_list = []
    pedal_events = {}
    pedal_flag = False
    tempo = int(mid.get_tempo_changes()[1][0]) if mid.get_tempo_changes()[1].size > 0 else 120 

    # Si se especifica un instrumento, agregar su token
    #if instrument is not None:
    #    token = f"<{instrument.lower()}>"
    #    if token not in vocab:
    #        raise ValueError(f"Token {token} not found in vocabulary!")
    #    token_idx = vocab.index(token)
    #    event_list.append(token)
    #    index_list.append(token_idx)

    # Obtener eventos ordenados
    all_events = []
    for instr in mid.instruments:
        for note in instr.notes:
            all_events.append(('note_on', note.start, note.pitch, note.velocity))
            all_events.append(('note_off', note.end, note.pitch, 0))
        for cc in instr.control_changes:
            if cc.number == 64:  # Pedal sustain
                all_events.append(('control_change', cc.time, cc.value))

    all_events.sort(key=lambda x: x[1])

    # Procesar eventos
    current_time = 0
    for event_type, time, value, velocity in all_events:
        delta_time_ms = int(round((time - current_time) * 1000))  # Delta en milisegundos
        current_time = time

        if delta_time_ms > 0:
            time_to_events(delta_time_ms, event_list=event_list, index_list=index_list)

        if event_type == 'note_on':
            vel = velocity_to_bin(velocity)
            idx = value + 1  # +1 por <pad>
            event_list.append(vocab[note_on_events + note_off_events + time_shift_events + vel + 1])
            index_list.append(note_on_events + note_off_events + time_shift_events + vel + 1)
            event_list.append(vocab[idx])
            index_list.append(idx)
        elif event_type == 'note_off':
            note = value
            if pedal_flag:
                if note not in pedal_events:
                    pedal_events[note] = 0
                pedal_events[note] += 1
            else:
                idx = note_on_events + note + 1
                event_list.append(vocab[idx])
                index_list.append(idx)
        elif event_type == 'control_change':
            if value >= 64:  # Pedal down
                pedal_flag = True
            else:  # Pedal up
                if pedal_events:
                    pedal_flag = False
                    for note in pedal_events:
                        idx = note_on_events + note + 1
                        for _ in range(pedal_events[note]):
                            event_list.append(vocab[idx])
                            index_list.append(idx)
                    pedal_events = {}

    return LongTensor(index_list), event_list, tempo






def list_parser(index_list=None, event_list=None, fname="bloop", tempo=120, program=0):
    """
    Translates a set of events or indices in the Oore et. al, 2018 vocabulary into a midi file

    Args:
        index_list (list or torch.Tensor): list of indices in vocab OR
        event_list (list): list of events in vocab
        fname (str, optional): name for single track of midi file returned
        tempo (int, optional): tempo of midi file returned in µs / beat

    Returns:
        mid (pretty_midi.PrettyMIDI): single-track piano midi file translated from vocab
    """
    # Validar entrada: solo uno de index_list o event_list
    if not ((index_list is None) ^ (event_list is None)):
        raise ValueError("Input one of index_list or event_list, not both or neither")

    # Verificar y convertir event_list a index_list
    if event_list is not None:
        if not all(isinstance(i, str) for i in event_list):
            raise ValueError("All events in event_list must be str type")
        index_list = events_to_indices(event_list)
    else:
        if not all(isinstance(i.item() if hasattr(i, 'item') else i, int) for i in index_list):
            raise ValueError("All indices in index_list must be int type")

    # Crear objeto MIDI
    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=program, name=fname)  # Piano, canal 0

    # Variables para reconstrucción
    current_time = 0  # En segundos
    note_starts = {}  # Diccionario para rastrear inicios de notas
    vel = 0

    # Reconstruir el MIDI
    for idx in index_list:
        idx = idx.item() if hasattr(idx, 'item') else idx
        if idx <= 0:  # Ignorar <pad>
            continue
        idx -= 1  # Ajustar por <pad>

        # Note on
        if 0 <= idx < note_on_events:
            note = idx
            note_starts[note] = current_time
            vel = vel or 64  # Velocidad por defecto si no se especifica

        # Note off
        elif note_on_events <= idx < note_on_events + note_off_events:
            note = idx - note_on_events
            if note in note_starts:
                start_time = note_starts.pop(note)
                piano.notes.append(pretty_midi.Note(
                    velocity=vel, pitch=note, start=start_time, end=current_time
                ))

        elif note_on_events + note_off_events <= idx < note_on_events + note_off_events + time_shift_events:
            cut_time = idx - (note_on_events + note_off_events - 1)
            current_time += (cut_time * DIV) / 1000  # Incremento en segundos

        # Velocity
        elif note_on_events + note_off_events + time_shift_events <= idx < total_midi_events:
            vel = bin_to_velocity(idx - (note_on_events + note_off_events + time_shift_events))

    # Agregar notas pendientes
    for note, start_time in note_starts.items():
        piano.notes.append(pretty_midi.Note(
            velocity=vel or 64, pitch=note, start=start_time, end=current_time
        ))

    # Agregar instrumento y retornar
    mid.instruments.append(piano)
    return mid
