import numpy as np
import torch
from tqdm import tqdm
from midiutil.MidiFile import MIDIFile

######################################################
###
### PREPROCESSING
###
######################################################

def make_multi_channel(x_list, y_list, n_channels, return_list=False):
    """Transform a 2D MFCC image into a multi-n_channel 3D
    image by concatenating n_channels consecutive frames.

    Arguments
    ---------
    x_list : list of np.array (MFCC)
    y_list: list of np.array (transcription)
    n_channels : int, size of the extra dimension
    return_list : bool, if True, keeps the list structure, if False concatenate.

    Returns
    -------
    (x.shape[0]-n_channels+1, n_channels, x.shape[1]) 3D np.array,
    (y.shape[0]-n_channels+1, y.shape[1]) 2D np.array.
    """
    x_3d = []
    y_3d = []

    for x in x_list:
        x_shifted = [x[k:-n_channels+1+k, :] for k in range(n_channels-1)]
        x_shifted.append(x[n_channels-1:, :])
        x_3d.append(np.dstack(x_shifted).transpose(0,2,1))

    for y in y_list:
        y_3d.append(y[n_channels//2:-(n_channels//2)])

    assert x_3d[0].shape[0] == y_3d[0].shape[0], 'First dim are not the same for input and output : {} vs {}'.format(x.shape[0], y.shape[0])

    if not return_list:
         x_3d = np.concatenate(x_3d, axis=0)
         y_3d = np.concatenate(y_3d, axis=0)

    return x_3d, y_3d


######################################################
###
### Decoding
###
######################################################

def greedy_decoding(x, acoustic_model, language_model):
    decoded = torch.zeros(x.shape[0], 88)
    for t in range(x.shape[0]):
        if t == 0:
            transition_proba = torch.ones(88)
        else:
            transition_proba = language_model(yt.unsqueeze(0)).squeeze().data

        yt = acoustic_model(torch.FloatTensor(x[t]).unsqueeze(0)).data
        decoded[t] = yt*transition_proba

    return decoded


######################################################
###
### Evaluation metrics
###
######################################################

def tp_fn_fp_tn(p_y_pred, y_gt, thres=0.5, average=None):
    """
    Arguments
    ---------
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1.
      average: None (element wise) | 'micro' (calculate metrics globally)
        | 'macro' (calculate metrics for each label then average).

    Returns
    -------
      tp, fn, fp, tn or list of tp, fn, fp, tn.
    """
    if p_y_pred.ndim == 1:
        y_pred = np.zeros_like(p_y_pred)
        y_pred[np.where(p_y_pred > thres)] = 1.
        tp = np.sum(y_pred + y_gt > 1.5)
        fn = np.sum(y_gt - y_pred > 0.5)
        fp = np.sum(y_pred - y_gt > 0.5)
        tn = np.sum(y_pred + y_gt < 0.5)
        return tp, fn, fp, tn
    elif p_y_pred.ndim == 2:
        tps, fns, fps, tns = [], [], [], []
        n_classes = p_y_pred.shape[1]
        for j1 in range(n_classes):
            (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred[:, j1], y_gt[:, j1], thres, None)
            tps.append(tp)
            fns.append(fn)
            fps.append(fp)
            tns.append(tn)
        if average is None:
            return tps, fns, fps, tns
        elif average == 'micro' or average == 'macro':
            return np.sum(tps), np.sum(fns), np.sum(fps), np.sum(tns)
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")

def prec_recall_fvalue(p_y_pred, y_gt, thres=0.5, average=None):
    """
    Arguments
    ---------
      p_y_pred: shape = (n_samples,) or (n_samples, n_classes)
      y_gt: shape = (n_samples,) or (n_samples, n_classes)
      thres: float between 0 and 1.
      average: None (element wise) | 'micro' (calculate metrics globally)
        | 'macro' (calculate metrics for each label then average).

    Returns
    -------
      prec, recall, acc, fvalue | list or prec, recall, acc, fvalue.
    """
    eps = 1e-10
    if p_y_pred.ndim == 1:
        (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred, y_gt, thres, average=None)
        prec = tp / max(float(tp + fp), eps)
        recall = tp / max(float(tp + fn), eps)
        acc = tp / max(float(tp + fp + fn), eps)
        fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
        return prec, recall, acc, fvalue
    elif p_y_pred.ndim == 2:
        n_classes = p_y_pred.shape[1]
        if average is None or average == 'macro':
            precs, recalls, accs, fvalues = [], [], [], []
            for j1 in range(n_classes):
                (prec, recall, acc, fvalue) = prec_recall_fvalue(
                    p_y_pred[:, j1], y_gt[:, j1], thres, average=None
                )
                precs.append(prec)
                recalls.append(recall)
                accs.append(acc)
                fvalues.append(fvalue)
            if average is None:
                return precs, recalls, accs, fvalues
            elif average == 'macro':
                return np.mean(precs), np.mean(recalls), np.mean(acc), np.mean(fvalues)

        elif average == 'micro':
            (prec, recall, acc, fvalue) = prec_recall_fvalue(
                p_y_pred.flatten(), y_gt.flatten(), thres, average=None
            )
            return prec, recall, acc, fvalue
        else:
            raise Exception("Incorrect average arg!")
    else:
        raise Exception("Incorrect dimension!")


def batch_metrics(loader, model, thres=0.5, average='micro'):
    """
    Arguments
    ---------
        loader: torch DataLoader.
        model: torch nn.Module.
        thres: float between 0 and 1.
        average: 'micro' (calculate metrics globally)
          | 'macro' (calculate metrics for each label then average).
    Returns
    -------
        acc, fvalue.
    """
    # y_pred = (torch.rand(N, 88)>0.5).type(torch.float)
    accs = []
    fvalues = []
    model.eval()
    with tqdm(total=len(loader)) as pbar:
        for x, y in loader:
            y_pred = model(x)
            prec, recall, acc, fvalue = prec_recall_fvalue(
                y_pred.data.T, y.data.cpu().numpy().T, 0.5, 'micro'
            )
            accs.append(acc)
            fvalues.append(fvalue)
            pbar.update(1)
    return np.mean(accs), np.mean(fvalues)


def batch_metrics_greedy_decoding(loader, acoustic_model, language_model, thres=0.5, average='micro'):
    """
    Arguments
    ---------
        loader: torch DataLoader.
        model: torch nn.Module.
        thres: float between 0 and 1.
        average: 'micro' (calculate metrics globally)
          | 'macro' (calculate metrics for each label then average).
    Returns
    -------
        acc, fvalue.
    """
    accs = []
    fvalues = []
    acoustic_model.eval()
    language_model.eval()
    with tqdm(total=len(loader)) as pbar:
        for x, y in loader:
            y_pred = greedy_decoding(x, acoustic_model, language_model)
            prec, recall, acc, fvalue = prec_recall_fvalue(
                y_pred.data.T, y.data.cpu().numpy().T, thres, 'micro'
            )
            accs.append(acc)
            fvalues.append(fvalue)
            pbar.update(1)
    return np.mean(accs), np.mean(fvalues)


######################################################
###
### MIDI
###
######################################################

def prob_to_midi_roll(x, threshold):
    """Threshold input probability to binary, then convert piano roll (n_time, 88)
    to midi roll (n_time, 108).

    Arguments
    ---------
      x: (n_time, n_pitch)
    """
    pitch_bgn = 21
    x_bin = np.zeros_like(x)
    x_bin[np.where(x >= threshold)] = 1
    n_time = x.shape[0]
    out = np.zeros((n_time, 128))
    out[:, pitch_bgn : pitch_bgn + 88] = x_bin
    return out


def write_midi_roll_to_midi(x, out_path):
    """Write out midi_roll to midi file.

    Arguments
    ---------
      x: (n_time, n_pitch), midi roll.
      out_path: string, path to write out the midi.
    """
    sample_rate = 16000
    n_step = 512
    step_sec = float(n_step) / sample_rate

    def _get_bgn_fin_pairs(ary):
        pairs = []
        bgn_fr, fin_fr = -1, -1
        for i2 in range(1, len(ary)):
            if ary[i2-1] == 0 and ary[i2] == 0:
                pass
            elif ary[i2-1] == 0 and ary[i2] == 1:
                bgn_fr = i2
            elif ary[i2-1] == 1 and ary[i2] == 0:
                fin_fr = i2
                if fin_fr > bgn_fr:
                    pairs.append((bgn_fr, fin_fr))
            elif ary[i2-1] == 1 and ary[i2] == 1:
                pass
            else:
                raise Exception("Input must be binary matrix!")

        return pairs

    # Get (pitch, bgn_frame, fin_frame) triple.
    triples = []
    (n_time, n_pitch) = x.shape
    for i1 in range(n_pitch):
        ary = x[:, i1]
        pairs_per_pitch = _get_bgn_fin_pairs(ary)
        if pairs_per_pitch:
            triples_per_pitch = [(i1,) + pair for pair in pairs_per_pitch]
            triples += triples_per_pitch

    # Sort by begin frame.
    triples = sorted(triples, key=lambda x: x[1])

    # Write out midi.
    MyMIDI = MIDIFile(1)    # Create the MIDIFile Object with 1 track
    track = 0
    time = 0
    tempo = 120
    beat_per_sec = 60. / float(tempo)
    MyMIDI.addTrackName(track, time, "Sample Track")  # Add track name
    MyMIDI.addTempo(track, time, tempo)   # Add track tempo

    for triple in triples:
        (midi_pitch, bgn_fr, fin_fr) = triple
        bgn_beat = bgn_fr * step_sec / float(beat_per_sec)
        fin_beat = fin_fr * step_sec / float(beat_per_sec)
        dur_beat = fin_beat - bgn_beat
        MyMIDI.addNote(track=0,     # The track to which the note is added.
                    channel=0,   # the MIDI channel to assign to the note. [Integer, 0-15]
                    pitch=midi_pitch,    # the MIDI pitch number [Integer, 0-127].
                    time=bgn_beat,      # the time (in beats) at which the note sounds [Float].
                    duration=dur_beat,  # the duration of the note (in beats) [Float].
                    volume=100)  # the volume (velocity) of the note. [Integer, 0-127].
    out_file = open(out_path, 'wb')
    MyMIDI.writeFile(out_file)
    out_file.close()
