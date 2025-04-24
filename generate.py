import torch
import pretty_midi
import time
import argparse
from masking import *
from tokenizer import *
from vocabulary import *


#device = "cpu"

def load_model(filepath, compile=False):
    from model import MusicTransformer
    from hparams import hparams, device
    file = torch.load(filepath, map_location=device)
    if "hparams" not in file:
        file["hparams"] = hparams
    model = MusicTransformer(**file["hparams"]).to(device)
    #model.load_state_dict(file["state_dict"], strict=False)
    if compile:
        model = torch.compile(model)
    model.eval()
    return model

def greedy_decode_conditional(model, inp, mode="categorical", temperature=1.0, k=None):
    inp = events_to_indices(inp)

    if inp[0] != start_token:
        inp = [start_token] + inp
    if inp[1] != guitar_token:
        inp.insert(1, guitar_token) 
    if inp[-1] != sep_token:
        inp = inp + [sep_token]# + [bass_token]

    inp = torch.tensor(inp, dtype=torch.int64, device=device).unsqueeze(0)

    #print("inp",inp)
    n = inp.dim() + 2
    if not callable(temperature):
        temperature__ = temperature
        def temperature(x): return temperature__
    if k is not None and not callable(k):
        k__ = k
        def k(x): return k__
    torch.set_float32_matmul_precision("high")
    try:
        with torch.no_grad():
            while True:
                predictions = model(inp, mask=create_mask(inp, n))
                #print("pred1",predictions)
                predictions /= temperature(inp[-1].shape[-1])
                if mode == "argmax":
                    prediction = torch.argmax(predictions[..., -1, :], dim=-1)
                elif k is not None:
                    top_k_preds = torch.topk(predictions[..., -1, :], k(inp[-1].shape[-1]), dim=-1)
                    predicted_idx = torch.distributions.Categorical(logits=top_k_preds.values[..., -1, :]).sample()
                    prediction = top_k_preds.indices[..., predicted_idx]
                elif mode == "categorical":
                    prediction = torch.distributions.Categorical(logits=predictions[..., -1, :]).sample()
                else:
                    raise ValueError("Invalid mode or top k passed in")

                #print("pred",prediction)

                if prediction == end_token:
                    return inp.squeeze()
                inp = torch.cat([inp, prediction.view(1, 1)], dim=-1)

    #except (KeyboardInterrupt, RuntimeError):
    except (RuntimeError):
        pass
    return inp.squeeze()



def greedy_decode(model, guitar_seq, max_tokens=512, mode="categorical", temperature=1.0, k=None):
    # Convertir la secuencia de guitarra a índices
    guitar_seq = events_to_indices(guitar_seq)
    if guitar_seq[0] != start_token:
        guitar_seq = [start_token] + guitar_seq
    if guitar_seq[-1] != end_token:
        guitar_seq = guitar_seq + [end_token]
    guitar_seq = torch.tensor(guitar_seq, dtype=torch.int64, device=device).unsqueeze(0)

    # Inicializar la secuencia de bajo con <start>
    bass_seq = torch.tensor([start_token], dtype=torch.int64, device=device).unsqueeze(0)

    # Generar la secuencia de bajo token por token
    for _ in range(max_tokens):
        # Generar máscaras: no causal para guitarra, causal para bajo
        guitar_mask = create_mask(guitar_seq, n=guitar_seq.dim() + 2, is_causal=False)
        bass_mask = create_mask(bass_seq, n=bass_seq.dim() + 2, is_causal=True)

        # Pasar por el modelo
        with torch.no_grad():
            predictions = model(guitar_seq, bass_seq, guitar_mask, bass_mask)

        # Obtener el último token predicho
        last_pred = predictions[:, -1, :]

        # Aplicar temperature
        last_pred /= temperature

        if mode == "argmax":
            prediction = torch.argmax(last_pred, dim=-1)

        elif k is not None:
            top_k_preds = torch.topk(last_pred, k, dim=-1)
            prediction = torch.distributions.Categorical(logits=top_k_preds.values).sample()
            prediction = top_k_preds.indices[0, prediction]

        elif mode == "categorical":
            prediction = torch.distributions.Categorical(logits=last_pred).sample()
        else:
            raise ValueError("Invalid mode")

        # Añadir el token predicho a la secuencia de bajo
        bass_seq = torch.cat([bass_seq, prediction.view(1, 1)], dim=-1)

        # Si se predice <end>, detener
        if prediction.item() == end_token:
            break

    return bass_seq.squeeze()



def audiate_conditional(token_ids, save_path="gneurshk.mid", tempo=512820, verbose=False):
    if not save_path.endswith(".mid"):
        save_path += ".mid"
    print(f"Saving midi file at {save_path}...") if verbose else None
    idx_guitar = (token_ids==guitar_token).nonzero(as_tuple=True)[0].item()
    idx_bass = (token_ids==bass_token).nonzero(as_tuple=True)[0].item()### Should be bass token

    token_ids_guitar = token_ids[idx_guitar+1 :idx_bass-1]
    token_ids_bass = token_ids[idx_bass+1 :]


    mid_guitar = list_parser(token_ids_guitar, tempo=tempo, program = 25)
    mid_bass = list_parser(token_ids_bass, tempo=tempo, program = 33)


    for note in mid_guitar.instruments[0].notes:
        print(f"Nota {note.pitch}: start={note.start}, end={note.end}")

    for note in mid_bass.instruments[0].notes:
        print(f"Bajo Nota {note.pitch}: start={note.start}, end={note.end}")

    mid_guitar.write("guitar.mid")

    ### New joint midi
    tempo_times, tempo_values = mid_guitar.get_tempo_changes()

    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo_values[0])
    
    mid.instruments.append(mid_guitar.instruments[0])
    mid.instruments.append(mid_bass.instruments[0])

    mid.write(save_path)
    print("Done")
    return


def audiate(guitar_seq, bass_seq, save_path="gneurshk.mid", tempo=512820, verbose=False):
    if not save_path.endswith(".mid"):
        save_path += ".mid"
    print(f"Saving midi file at {save_path}...") if verbose else None

    # Convertir secuencias a eventos
    #guitar_seq = guitar_seq.tolist()
    #bass_seq = bass_seq.tolist()
    #guitar_events = indices_to_events(guitar_seq)
    #bass_events = indices_to_events(bass_seq)


    #print(guitar_events)
    #print(bass_events)

    # Crear objetos PrettyMIDI para guitarra y bajo
    mid_guitar = list_parser(guitar_seq, tempo=tempo, program=25)  # Guitarra acústica
    mid_bass = list_parser(bass_seq, tempo=tempo, program=33)     # Bajo eléctrico

    # Combinar ambos instrumentos en un solo archivo MIDI
    mid = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    mid.instruments.append(mid_guitar.instruments[0])
    mid.instruments.append(mid_bass.instruments[0])

    mid.write(save_path)
    print("Done")
    return





def generate_conditional(model_, inp, save_path="./bloop.mid", mode="categorical", temperature=1.0, k=None, tempo=120, verbose=False, max_tokens=512):
    print("Greedy decoding...") if verbose else None
    start = time.time()
    token_ids = greedy_decode(model=model_, inp=inp, mode=mode, temperature=temperature, k=k, max_tokens=max_tokens)
    end = time.time()
    print(f"Generated {len(token_ids)} tokens. Time taken: {round(end - start, 2)} secs.") if verbose else None
    print(token_ids)
    print(indices_to_events(token_ids))
    return audiate(token_ids=token_ids, save_path=save_path, tempo=tempo, verbose=verbose)


def generate(model_, guitar_input, save_path="./bloop.mid", mode="categorical", temperature=1.0, k=None, tempo=512820, verbose=False, max_tokens=512):
    print("Greedy decoding...") if verbose else None
    start = time.time()
    bass_seq = greedy_decode(model=model_, guitar_seq=guitar_input, mode=mode, temperature=temperature, k=k, max_tokens=max_tokens)

    guitar_seq = events_to_indices(["<start>"] + guitar_input + ["<end>"])
    print(bass_seq)
    end = time.time()
    print(f"Generated {len(bass_seq)} tokens. Time taken: {round(end - start, 2)} secs.") if verbose else None
    return audiate(guitar_seq, bass_seq, save_path=save_path, tempo=tempo, verbose=verbose)



if __name__ == "__main__":
    from hparams import hparams

    def check_positive(x):
        if x is None:
            return x
        x = int(x)
        if x <= 0:
            raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
        return x

    parser = argparse.ArgumentParser(description="Generate midi audio with a Music Transformer!")
    parser.add_argument("path_to_model", help="path to .pt file with model state dict and hyperparameters", type=str)
    parser.add_argument("save_path", help="path to save the generated midi file", type=str)
    parser.add_argument("-c", "--compile", help="compile model for speed", action="store_true")
    parser.add_argument("-m", "--mode", help="decode sampling mode: 'categorical' or 'argmax'", type=str)
    parser.add_argument("-k", "--top-k", help="top k samples for decoding", type=check_positive)
    parser.add_argument("-t", "--temperature", help="temperature for sampling", type=float)
    parser.add_argument("-tm", "--tempo", help="tempo in BPM", type=check_positive)
    parser.add_argument("-i", "--midi-prompt", help="MIDI file to continue", type=str)
    parser.add_argument("-it", "--midi-prompt-tokens", help="number of tokens from MIDI prompt", type=int)
    parser.add_argument("-ins", "--instrument", help="initial instrument: 'guitar' or 'bass'", type=str, default="guitar")
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")

    args = parser.parse_args()

    temperature_ = float(args.temperature) if args.temperature else 1.0
    mode_ = args.mode if args.mode else "categorical"
    k_ = int(args.top_k) if args.top_k else None
    tempo_ = int(60 * 1e6 / int(args.tempo)) if args.tempo else 512820

    if args.midi_prompt:
        midi_parser_output = midi_parser(args.midi_prompt)
        tempo_ = midi_parser_output[2]
        guitar_input = (midi_parser_output[1])[0:args.midi_prompt_tokens] if args.midi_prompt_tokens else midi_parser_output[1]
    else:
        raise ValueError("No midi sequence recibed.")

    music_transformer = load_model(args.path_to_model, args.compile)
    print(guitar_input)
    generate(model_=music_transformer, guitar_input=guitar_input, save_path=args.save_path,
             temperature=temperature_, mode=mode_, k=k_, tempo=tempo_, verbose=args.verbose, max_tokens = args.midi_prompt_tokens)
