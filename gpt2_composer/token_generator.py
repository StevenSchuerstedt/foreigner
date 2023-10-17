import numpy as np
import note_seq

from .constants import *

def token_sequence_to_note_sequence(token_sequence, use_program=True, use_drums=True, instrument_mapper=None, only_piano=False):

    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()

    note_sequence = empty_note_sequence()

    # Render all notes.
    current_program = 1
    current_is_drum = False
    current_instrument = 0
    track_count = 0
    for token_index, token in enumerate(token_sequence):

        if token == "PIECE_START":
            pass
        elif token == "PIECE_END":
            break
        elif token == "TRACK_START":
            current_bar_index = 0
            track_count += 1
            pass
        elif token == "TRACK_END":
            pass
        elif token == "KEYS_START":
            pass
        elif token == "KEYS_END":
            pass
        elif token.startswith("KEY="):
            pass
        elif token.startswith("INST"):
            instrument = token.split("=")[-1]
            if instrument != "DRUMS" and use_program:
                if instrument_mapper is not None:
                    if instrument in instrument_mapper:
                        instrument = instrument_mapper[instrument]
                current_program = int(instrument)
                current_instrument = track_count
                current_is_drum = False
            if instrument == "DRUMS" and use_drums:
                current_instrument = 0
                current_program = 0
                current_is_drum = True
        elif token == "BAR_START":
            current_time = current_bar_index * BAR_LENGTH_120BPM
            current_notes = {}
        elif token == "BAR_END":
            current_bar_index += 1
            pass
        elif token.startswith("NOTE_ON"):
            pitch = int(token.split("=")[-1])
            note = note_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + 4 * NOTE_LENGTH_16TH_120BPM
            note.pitch = pitch
            note.instrument = current_instrument
            note.program = current_program
            note.velocity = 80
            note.is_drum = current_is_drum
            current_notes[pitch] = note
        elif token.startswith("NOTE_OFF"):
            pitch = int(token.split("=")[-1])
            if pitch in current_notes:
                note = current_notes[pitch]
                note.end_time = current_time
        elif token.startswith("TIME_DELTA"):
            delta = float(token.split("=")[-1]) / STEPS_PER_QUARTER * 4 * NOTE_LENGTH_16TH_120BPM
            current_time += delta
        elif token.startswith("DENSITY="):
            pass
        elif token == "[PAD]":
            pass
        else:
            #print(f"Ignored token {token}.")
            pass

    # Make the instruments right.
    instruments_drums = []
    for note in note_sequence.notes:
        pair = [note.program, note.is_drum]
        if pair not in instruments_drums:
            instruments_drums += [pair]
        note.instrument = instruments_drums.index(pair)

    if only_piano:
        for note in note_sequence.notes:
            if not note.is_drum:
                note.instrument = 0
                note.program = 0

    return note_sequence


def empty_note_sequence(qpm=120.0, total_time=0.0):
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    note_sequence.total_time = total_time
    return note_sequence


def note_sequence_to_token_sequence(note_sequence, note_densities=None, order=[0]):
    token_sequence = []
    token_sequence.append("PIECE_START")
    # token_sequence.append("STYLE=JSFAKES")
    # token_sequence.append("GENRE=JSFAKES")

    # handle tracks
    tracks = get_tracks_from_note_sequence(note_sequence, order)
    for track_id in tracks:
        track = tracks[track_id]
        if(len(track) == 0):
            continue
        token_sequence.append("TRACK_START")
        token_sequence.append(f"INST={track[0].program}")
        if note_densities:
            notes_per_bar = len(track) / 4
            note_density = np.where(
                np.array(note_densities[str(track[0].program)]) >= notes_per_bar)[0]
            note_density = note_density[0] if len(note_density) > 0 else 9
            token_sequence.append(f"DENSITY={int(note_density)}")
        # handle bars
        bars = get_bars_from_track(track)
        for bar_index, bar in enumerate(bars):
            token_sequence.append("BAR_START")
            token_sequence.extend(bar_notes_to_token_sequence(bar, bar_index))
            token_sequence.append("BAR_END")
            

        token_sequence.append("TRACK_END")

    # we do not use a PIECE END token, as we can simply sample until we reach the nth TRACK END token if we wish to generate n tracks.
    # token_sequence.append("PIECE_END")
    return token_sequence


def get_tracks_from_note_sequence(note_sequence, order=[0]):
    tracks = {}
    for o in order:
        tracks[o] = []

    for note in note_sequence.notes:
        if note.instrument not in tracks:
            tracks[note.instrument] = []
        tracks[note.instrument].append(note)
    return tracks


def get_bars_from_track(track):
    bars = []
    num_bars = count_bars(track)
    for bar in range(num_bars):
        bars.append(
            list(filter(lambda n: n.quantized_start_step // STEPS_PER_BAR == bar, track)))
    return bars


def count_bars(track):
    track.sort(key=lambda note: (
        note.quantized_start_step, note.quantized_end_step))
    return int(np.ceil(track[-1].quantized_end_step / STEPS_PER_BAR))


def bar_notes_to_token_sequence(bar, bar_index):
    token_sequence = []
    current_notes = {}
    cursor = bar_index * STEPS_PER_BAR
    for note in bar:
        while len(current_notes.keys()) > 0:
            ending_pitch = None
            for pitch in current_notes:
                if current_notes[pitch].quantized_end_step <= note.quantized_start_step:
                    if ending_pitch:
                        if current_notes[pitch].quantized_end_step < current_notes[ending_pitch].quantized_end_step:
                            ending_pitch = pitch
                    else:
                        ending_pitch = pitch

            if ending_pitch:
                ending_note = current_notes[ending_pitch]
                time_delta = ending_note.quantized_end_step - cursor
                if time_delta > 0:
                    token_sequence.append(f"TIME_DELTA={time_delta}")
                token_sequence.append(f"NOTE_OFF={ending_pitch}")
                cursor = ending_note.quantized_end_step
                del current_notes[ending_pitch]
            else:
                break

        time_delta = note.quantized_start_step - cursor
        if time_delta > 0:
            token_sequence.append(f"TIME_DELTA={time_delta}")
            cursor = note.quantized_start_step
        
        # ending_pitches = []
        # for pitch in current_notes:
        #     token_sequence.append(f"NOTE_OFF={pitch}")
        #     ending_pitches.append(pitch)
        # for pitch in ending_pitches:
        #     del current_notes[pitch]

        current_notes[note.pitch] = note
        token_sequence.append(f"NOTE_ON={note.pitch}")

    # handle ends of last note
    while len(current_notes.keys()) > 0:
        ending_pitch = None
        for pitch in current_notes:
            if current_notes[pitch].quantized_end_step <= (bar_index + 1) * STEPS_PER_BAR:
                if ending_pitch:
                    if current_notes[pitch].quantized_end_step < current_notes[ending_pitch].quantized_end_step:
                        ending_pitch = pitch
                else:
                    ending_pitch = pitch

        if ending_pitch:
            ending_note = current_notes[ending_pitch]
            time_delta = ending_note.quantized_end_step - cursor
            if time_delta > 0:
                token_sequence.append(f"TIME_DELTA={time_delta}")
            token_sequence.append(f"NOTE_OFF={ending_pitch}")
            cursor = ending_note.quantized_end_step
            del current_notes[ending_pitch]
        else:
            break

    # fill gap to bar end
    time_delta = (bar_index + 1) * STEPS_PER_BAR - cursor
    if time_delta > 0:
        token_sequence.append(f"TIME_DELTA={time_delta}")

    # end all still sounding notes
    for pitch in current_notes:
        token_sequence.append(f"NOTE_OFF={pitch}")

    return token_sequence
