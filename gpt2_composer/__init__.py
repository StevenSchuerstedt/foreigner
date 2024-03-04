from copy import deepcopy

import pandas as pd

from .token_generator import *
from .constants import *
#from .model import *


def extract_4bar_sections(fn_mid):
    note_seq_examples = []

    # load sequence
    
    try:
        seq = note_seq.midi_file_to_note_sequence(fn_mid)
    except note_seq.MIDIConversionError:
        print("error")
        return note_seq_examples

    #seq = note_seq.midi_file_to_note_sequence(fn_mid)
    


    # split on tempo and time signature changes
    sub_seqs = note_seq.sequences_lib.split_note_sequence_on_time_changes(seq)
   
    # iterate sub_seqs
    for sub_seq in sub_seqs:
        # quantize sequence
        seq_quant = note_seq.quantize_note_sequence(
            sub_seq, steps_per_quarter=STEPS_PER_QUARTER)
        
        # split to 4bars parts
        seq_bars = split_note_seq_nbars(seq_quant)

        for seq_bar in seq_bars:
            # clean seq bar
            seq_bar_cleaned = seq_bar
            # seq_bar_cleaned = clean_note_sequence_pop909(seq_bar)

            # # skip examples with to few notes per instrument
            # if len(list(filter(lambda note: note.instrument == 0, seq_bar_cleaned.notes))) < 2:
            #     continue
            # if len(list(filter(lambda note: note.instrument == 2, seq_bar_cleaned.notes))) < 2:
            #     continue
            # if fn_chords:
            #     if len(list(filter(lambda note: note.instrument == 3, seq_bar_cleaned.notes))) < 2:
            #         continue

            note_seq_examples.append(seq_bar_cleaned)

    return note_seq_examples


def split_note_seq_nbars(seq_quant, num_bars=4, bpm=120, offset_bar=0):
    seq_bars = []
    total_bars = seq_quant.total_quantized_steps // (4 * STEPS_PER_QUARTER)
    for bar in range(offset_bar, total_bars, num_bars):
        seq_quant_nbars = deepcopy(seq_quant)
        # update total duration
        seq_quant_nbars.total_quantized_steps = num_bars * STEPS_PER_BAR
        seq_quant_nbars.total_time = seq_quant_nbars.total_quantized_steps / \
            STEPS_PER_QUARTER / (bpm / 60)
        # set fixed tempo
        seq_quant_nbars.tempos[0].qpm = 120
        # clear all control sequences
        while len(seq_quant_nbars.control_changes) > 0:
            seq_quant_nbars.control_changes.pop()

        notes_to_remove = []
        bars_start = bar * STEPS_PER_BAR
        bars_end = (bar+num_bars) * STEPS_PER_BAR
        for note in seq_quant_nbars.notes:
            # is not in bars
            if note.quantized_start_step >= bars_start and note.quantized_start_step < bars_end:
                # truncate note if longer then range
                if note.quantized_end_step > bars_end:
                    note.quantized_end_step = bars_end

                # remove offset
                note.quantized_start_step -= bars_start
                note.quantized_end_step -= bars_start

                # update sec times
                note.start_time = note.quantized_start_step / \
                    STEPS_PER_QUARTER / (bpm / 60)
                note.end_time = note.quantized_end_step / \
                    STEPS_PER_QUARTER / (bpm / 60)

                # prune too short notes
                if note.quantized_end_step - note.quantized_start_step <= 1:
                    notes_to_remove.append(note)
            else:
                notes_to_remove.append(note)

        for note in notes_to_remove:
            seq_quant_nbars.notes.remove(note)

        seq_bars.append(seq_quant_nbars)

    return seq_bars

