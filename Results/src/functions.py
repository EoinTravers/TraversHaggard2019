# coding: utf-8
import os
import sys
from glob import glob
import mne
from mne import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
import pandas as pd
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.externals import joblib
import statsmodels.formula.api as smf
from scipy import signal

import eegf

million = 1000000.


## Some preprocessing functions
def load_subject_csv(subject):
    dat_cols = ['participant', 'cb', 'condition',
                'block_nr', 'block_half', 'trial_nr', 'v_win', 'p_win', 'action',
                'response', 'rt', 'outcome', 'score_delta', 'score', 'visible']
    fn = glob('data/csv/%i*.csv' % subject)
    if len(fn) != 1:
        print('Something wrong reading files.', fn)
        raise Exception('Something wrong reading files.')
    fn = fn[0]
    data = pd.read_csv(fn)[dat_cols]
    visible = data[data['visible']==1].copy()
    m = smf.logit('response ~ v_win * p_win', data=visible).fit()
    pred = m.predict(data)
    data['predicted_response'] = np.where(data['visible']==1, pred, np.nan)
    data['predicted_action'] = np.where(data['condition']==0,
                                        data['predicted_response'], 1-data['predicted_response'])
    data['difficulty'] = .5 - np.abs(data['predicted_response'] - .5)
    data['difficult'] = data['difficulty'] > data['difficulty'].median()
    return data

def exclude_dropped_metadata(behaviour, epochs):
    dl = [v[0]  if (len(v) > 0) else 'ok' for v in epochs.drop_log]
    dl = np.array(dl)
    dl = dl[dl != 'IGNORED']
    data = behaviour.copy()
    data['droplog'] = dl
    data = data[data['droplog'] == 'ok']
    data.index = list(range(len(data)))
    return data

def get_gfp_peaks(erp, lp=4):
    gfp = erp.data.var(0)
    # gfp = np.concatenate([[np.nan], np.diff(gfp)])
    gfp_smooth = eegf.butter_lowpass_filter(gfp, lp, 250)
    peaks = signal.find_peaks(gfp_smooth)[0]
    times = erp.times[peaks]
    return times

def varimax(components, method='varimax', eps=1e-6, itermax=100,
            return_rotation_matrix=False):
    """Return rotated components."""
    if (method == 'varimax'):
        gamma = 1.0
    elif (method == 'quartimax'):
        gamma = 0.0

    nrow, ncol = components.shape
    rotation_matrix = np.eye(ncol)
    var = 0
    for _i in range(itermax):
        comp_rot = np.dot(components, rotation_matrix)
        tmp = np.diag((comp_rot ** 2).sum(axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(
            np.dot(components.T, comp_rot ** 3 - np.dot(comp_rot, tmp)))
        rotation_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var != 0 and (var_new < var * (1 + eps)):
            break
        var = var_new
    if return_rotation_matrix:
        return rotation_matrix
    else:
        return np.dot(components, rotation_matrix).T


def func_by_subject(epochs, func, exclude=[]):
    '''func must take an axis argument.'''
    data = epochs.metadata
    subjects = data['participant'].unique()
    subjects = [s for s in subjects if s not in exclude]
    res = []
    for s in subjects:
        X = epochs['participant == %i' % s].get_data()
        x = func(X, axis=0)
        res.append(x)
    return np.array(res)

def mean_by_subject(epochs, exclude=[]):
    return func_by_subject(epochs, np.mean, exclude=exclude)

def std_by_subject(epochs, exclude=[]):
    return func_by_subject(epochs, np.std, exclude=exclude)

## PCA Analyses

def topomap(w, info, axes=None, show=False):
    a = np.abs(w).max()
    return mne.viz.plot_topomap(w, info, axes=axes, show=show)


def do_rt_comparison(epochs, ch, drop_direction, bins=6, se=False):
    '''drop direction = 1 for trial-locked epochs, -1 for response-locked'''
    E = epochs.copy()
    qs = pd.qcut(E.metadata['rt'], bins)
    t0 = E.time_as_index(0)[0]
    for i, q in enumerate(np.sort(qs.unique())):
        ix = qs == q
        e = E[ix]
        lbl = '%.2f < RT < %.2f' % (q.left, q.right)
        X = e.get_data()[:, ch] * million
        if drop_direction==1:
            ti = e.time_as_index(q.right)[0]
            if ti > 0:
                X[:, ti:] = np.nan
        else:
            ti = e.time_as_index(-q.right)[0]
            if ti > 0:
                X[:, :ti] = np.nan
        if se:
            eegf.plot_mean_sem(X, e.times, label=lbl)
        else:
            plt.plot(e.times, X.mean(0), label=lbl)
    plt.vlines(0, linestyle='dashed', *plt.ylim())

def get_condition_epochs(epochs, crop=None, baseline=None):
    E = epochs.copy()
    if baseline is not None:
        E = E.apply_baseline(baseline)
    if crop is not None:
        E = E.crop(*crop)
    easy = E['(visible==True) & (difficult==False)']
    hard = E['(visible==True) & (difficult==True)']
    guess = E['(visible==False)']
    return easy, hard, guess

def do_threeway_comparison(epochs, ch, 
                           by_subject=True,
                           agg_func=mean_by_subject,
                           crop=None, baseline=None, 
                           title=None, ax=None, legend=True,
                           neg_up=True):
    # labels = ['Easy choice', 'Difficult choice', 'Guess']
    if ax is not None:
        plt.sca(ax)
    es = get_condition_epochs(epochs, crop=crop, baseline=baseline)
    labs = ['Easy', 'Difficult', 'Guess']
    for e, l in zip(es, labs):
        if by_subject:
            X = agg_func(e)           
        else:
            X = e.get_data()
        if type(ch) == list and len(ch) == 2:
            X = million * (X[:, ch[0]] - X[:, ch[1]])
        else:
            X = X[:, ch] * million
        eegf.plot_mean_sem(X, e.times, label=l)
        if neg_up:
            eegf.flipy()
    if legend:
        plt.legend()

def do_twoway_comparison(epochs, ch, variable, labels,
                         crop=None, baseline=None, title=None,
                         by_subject=True, neg_up=True):
    E = epochs.copy()
    if crop is not None:
        E = E.crop(*crop)
    if baseline is not None:
        E = E.apply_baseline(baseline)
    e0 = E['%s == 0' % variable]
    e1 = E['%s == 1' % variable]
    es = [e0, e1]
    for e, l in zip(es, labels):
        if by_subject:
            X = mean_by_subject(e)[:, ch] * million
        else:
            X = e.get_data()[:, ch] * million
        eegf.plot_mean_sem(X, e.times, label=l)
        if neg_up:
            eegf.flipy()
    plt.legend()
        
    
# def do_threeway_comparison(epochs, ch, crop=None, baseline=None, title=None, ax=None):
#     E = epochs.copy()
#     if baseline is not None:
#         E = E.apply_baseline(baseline)
#     if crop is not None:
#         E = E.crop(*crop)
#     if ax is not None:
#         plt.sca(ax)
#     easy = E['(visible==True) & (difficult==False)']
#     hard = E['(visible==True) & (difficult==True)']
#     guess = E['(visible==False)']
#     # labels = ['Easy choice', 'Difficult choice', 'Guess']
#     es = [easy, hard, guess]
#     labs = ['Easy', 'Hard', 'Guess']
#     for e, l in zip(es, labs):
#         X = e.get_data()[:, ch] * million
#         eegf.plot_mean_sem(X, e.times, label=l)
#     plt.legend()
    



def component_plot(epochs, plotfunc, which, fn=None, **plotfuncargs):
    '''I don't remember what this doesn...'''
    print((', '.join(['{}={!r}'.format(k, v) for k, v in list(plotfuncargs.items())])))
    w = {'onset': w1, 'action': w2}[which]
    plt.figure(figsize=(w, 4))
    plotfunc(epochs, **plotfuncargs)
    plt.title(comp_names[c])
    plt.hlines(0, linestyle='dotted', *plt.xlim())
    if which == 'onset':
        plt.xticks([0, .5, 1], ['0', '.5', '1'])
        plt.xlim(t1_0, t1_1) # 1.3 s in 6 cm = 6.15 cm per sec
        plt.xlabel('Time from stimulus onset (s)')
        plt.legend(loc='upper right', prop={'size': 12})
    elif which == 'action':
        plt.xlim(-2, .5)
        plt.title(comp_names[c])
        plt.xticks([-2, -1, 0])
        plt.xlabel('Time to action (s)')
        plt.legend(loc='lower left', prop={'size': 12})
    plt.tight_layout()
    if fn is not None:
        plt.savefig('figures/%s.png' % fn)
        plt.savefig('figures/%s.svg' % fn)
    plt.show()

def raw_by_subject(epochs, ch=9, yl=200, show_mean=True, show_raw=True):
    fig, axes = plt.subplots(figsize=(20, 20), ncols=4, nrows=5)
    axes = iter(np.concatenate(axes))
    times = epochs.times
    subjects = epochs.metadata['participant'].unique()
    for subject in subjects:
        ax = next(axes)
        X = epochs['participant == %i' % subject].get_data() * million
        rawplot(X, times=times, ch=ch, yl=yl, show_raw=show_raw, show_mean=show_mean, ax=ax)
        ax.set_title(subject)
    plt.tight_layout()
    plt.show()

def rawplot(X, times, ch=9, yl=200, show_raw=True, show_mean=True, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    if show_raw:
        for i in range(X.shape[0]):
            ax.plot(times, X[i, ch], alpha=.2, color='b')
    if show_mean:
        ax.plot(times, X[:, ch].mean(0), color='r')
    ax.set_ylim(-yl, yl)
    ax.invert_yaxis()
    
def find_outlier_trials(epochs, thresh=120. / million):
    X = epochs.get_data()[:, :32]
    aX = np.abs(X).max(2).max(1)
    return aX > thresh



def check_one_hot(x):
    return set(x) == {0, 1} and np.sum(x) == 1

def do_component(trial_e, response_e, ch, weights, info, title=None, neg_up=True):
    '''This wrapper produces a comprehensive set of ERP plots for each channel.
    
    Parameters
    ----------
    trial_e : mne.Epochs
        Stimulus-locked epochs
    response_e : mne.Epochs
        Response-locked epochs
    ch : int
        Channel index (0-31 in this case)
    weights : np.array
        Weight-vector for topomap data. Length must correspond to number of channels.
    info : mne.Info
        mne.Info object for either set of epochs. Used to plot topomap of weight vector.
    title : str or None
        Title of the figure.

    Returns
    -------
    fig: matplotlib.figure.Figure
        2 x 2 figure:
            Topomap of weight vector; Wait vs Act trials (stimulus locked);
            Easy vs Hard vs Guess (stimulus locked); E vs H vs G (response-locked)
    '''
    ## Note on variables:
    ## t{} = stimulus (trial) locked. r{} = response locked.
    ## {}X = EEg signal. {}D = metadata
    tX = trial_e.get_data()[:, ch] * million
    rX = response_e.get_data()[:, ch] * million
    tD = trial_e.metadata
    rD = response_e.metadata
    ## Setup figure
    fig, axes = plt.subplots(figsize=(12, 8), ncols=2, nrows=2)
    if title is not None:
        plt.suptitle(title)
    axes = np.concatenate(axes)
    ## Topomap
    plt.sca(axes[0])
    topomap(weights, info)
    ## Action vs Inaction
    plt.sca(axes[1])
    do_twoway_comparison(trial_e, ch, variable='action', labels=['Wait', 'Act'], by_subject=True, neg_up=neg_up)
    plt.hlines(0, -.5, 2)
    plt.vlines(0, linestyle='--', *plt.ylim())
    plt.xlim(-.5, 2)
    plt.title('Action/Inaction')
    ## Stim locked: Easy vs Hard vs Guess
    plt.sca(axes[2])
    do_threeway_comparison(trial_e, ch, crop=(-.5, 2.), baseline=(-.1, 0), by_subject=True, neg_up=neg_up)
    plt.hlines(0, -.5, 2)
    plt.vlines(0, linestyle='--', *plt.ylim())
    plt.xlim(-.5, 2)
    plt.title('Condition')
    ## Resp-locked: Easy vs Hard vs Guess
    plt.sca(axes[3])
    do_threeway_comparison(response_e, ch, crop=(-2, .5), baseline=(-2.1, -2.), by_subject=True, neg_up=neg_up)
    plt.hlines(0, -2, .5)
    plt.xlim(-2., .25)
    plt.vlines(0, linestyle='--', *plt.ylim())
    plt.title('Condition')
    plt.tight_layout()
    return fig
    
# do_twoway_comparison(trial_E, c, variable='action', labels=['Wait', 'Act'])

def export_to_df(epochs, chans, filename=None, hz=50, chan_labels=None):
    try:
        _ = iter(chans) # Check is it a list
    except TypeError:
        chans = [chans]
    if chan_labels is None:
        chan_labels = ['ch%i' % ch for ch in chans]
    export_epochs = epochs.copy().resample(50)
    if epochs.info['sfreq'] != hz:
        export_epochs = export_epochs.resample(hz)
    X = export_epochs.get_data()
    ## Metadata
    meta_d = export_epochs.metadata.copy().drop('droplog', axis=1)
    for c in meta_d.columns:
        if meta_d[c].dtype == bool:
            meta_d[c] = meta_d[c]*1
    ## Iterate over trials
    out_df = []
    for trial_i in range(len(export_epochs)):
        md = meta_d.iloc[trial_i]   
        trial_df = pd.DataFrame(export_epochs.times, columns=['time'])
        for ch_i, lab in zip(chans, chan_labels):
            trial_df[lab] = X[trial_i, ch_i]
        for k, v in md.items():
            trial_df[k] = v
        out_df.append(trial_df)
    out_df = pd.concat(out_df)
    if filename is not None:
        out_df.to_csv(filename, index=False)
    return out_df

def rotate_eeg(X, L):
    return np.stack([X[i].T.dot(L) for i in range(X.shape[0])], axis=0).swapaxes(2, 1)

def rotate_epochs(epochs, L):
    n_chans, n_to_retain = L.shape
    info = mne.create_info(n_to_retain, sfreq=epochs.info['sfreq'], ch_types='eeg')
    X = epochs.get_data()[:, :n_chans]
    pcaX = rotate_eeg(X, L)
    rot_epochs = mne.EpochsArray(pcaX, info, tmin=epochs.times[0])
    rot_epochs.metadata = epochs.metadata
    return rot_epochs

def correct_rotation_signs(rotmat, epochs, t0, t1):
    '''We want the PCA components to increase between t0 and t1, so flip
    components where this isn't this case.

    Note: Sign of PCA components is arbitrary - this is just for easier interpretation!
    '''
    # t0, t1 = epochs.time_as_index([t0, t1])
    X = epochs.copy().crop(t0, t1).get_data()[:, :32]
    X_pca =  rotate_eeg(X, rotmat)
    m_pca = X_pca.mean(0)
    comp_signs = np.sign(m_pca[:, -1] - m_pca[:, 0])
    return rotmat * comp_signs


def plot_weight_topomaps(weights, info, label='C'):
    '''weights: n_comp x 32'''
    n = weights.shape[0]
    fig = plt.figure(figsize=(n*1.5, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        topomap(weights[i], info, axes=ax, show=False)
        plt.title('%s%i' % (label, i+1))
    return fig


def do_pca_for_subject(trial_epochs_csd, response_epochs_csd, s, info):
    from matplotlib.gridspec import GridSpec
    print('# Participant %i' % s)
    r_ep = response_epochs_csd['participant == %i' % s]
    X = r_ep.copy().crop(-2., 0).get_data()[:, :32]
    trialX = trial_epochs_csd['participant == %i' % s]
    respX = r_ep.get_data()[:, :32]
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(nrows=4, ncols=12, figure=fig)
    fig.suptitle('Participant %i' % s)
    ## Do PCA
    covariance_csd = np.array([np.cov(X[i] - X[i].mean()) 
                                    for i in range(X.shape[0])])
    cov = covariance_csd.mean(0)
    # eig_vals, eig_vecs = np.linalg.eig(cov)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    eig_vals = eig_vals[::-1] ## Reverse order
    eig_vecs = eig_vecs[:, ::-1]
    ## Variance explained
    ve = eig_vals / eig_vals.sum()
    ax1 = fig.add_subplot(gs[0, 0:2])
    plt.sca(ax1)
    plt.plot(list(range(1, len(ve)+1)), ve*100, '-o')
    plt.hlines(ve.mean()*100, linestyle='dashed', *plt.xlim())
    plt.ylabel('% variance explained')
    plt.xlabel('Component')
    plt.xticks(list(range(1, 32, 2)))
    plt.xlim(0, 12)
    ## Correct signs
    t0, t1 = response_epochs_csd.time_as_index([-2, 0])
    respX_pca =  np.stack([respX[i].T.dot(eig_vecs)
                           for i in range(respX.shape[0])], axis=0).swapaxes(2, 1)
    X = respX_pca.mean(0)
    comp_signs = np.sign(X[:, t1] - X[:, t0])
    eig_vecs *= comp_signs
    ## Plot topography
    n = 9
    for i in range(n):
        ax = fig.add_subplot(gs[0, 2+i])
        topomap(eig_vecs[:, i], info, axes=ax)
        plt.title('C%i' % (i+1))
    # Get timecourse
    respX_pca =  np.stack([respX[i].T.dot(eig_vecs)
                           for i in range(respX.shape[0])], axis=0).swapaxes(2, 1) * 1000000
    trialX = trial_epochs_csd.get_data()[:, :32]
    trialX_pca =  np.stack([trialX[i].T.dot(eig_vecs)
                            for i in range(trialX.shape[0])], axis=0).swapaxes(2, 1) * 1000000
    ## PCA timecourse 1
    ax3 = fig.add_subplot(gs[1:2, :6])
    plt.sca(ax3)
    for i in range(9):
        eegf.plot_mean_sem(trialX_pca[:, i], trial_epochs_csd.times, label='C %i' % (i+1))
    plt.vlines(0, linestyle='--', *plt.ylim())
    plt.legend(loc='upper left', prop={'size': 10})
    plt.xlim(-.5, 2)
    plt.xlabel('Time from onset (s)')
    ## PCA timecourse 2
    ax4 = fig.add_subplot(gs[1:2, 6:])
    plt.sca(ax4)
    for i in range(9):
        eegf.plot_mean_sem(respX_pca[:, i], response_epochs_csd.times, label='C %i' % (i+1))
    plt.vlines(0, linestyle='--', *plt.ylim())
    plt.legend(loc='upper left', prop={'size': 10})
    plt.xlim(-2, .5)
    plt.xlabel('Time to action (s)')
    ## Rotate
    varimax_vectors = varimax(eig_vals[:n] * eig_vecs[:, :n], method='varimax').T
    ## Correct signs
    t0, t1 = response_epochs_csd.time_as_index([-2, 0])
    respX_vmax =  np.stack([respX[i].T.dot(varimax_vectors)
                            for i in range(respX.shape[0])], axis=0).swapaxes(2, 1)
    X = respX_vmax.mean(0)
    comp_signs = np.sign(X[:, t1] - X[:, t0])
    varimax_vectors *= comp_signs
    ## Topography
    for i in range(n):
        ax = fig.add_subplot(gs[2, 2+i])
        topomap(varimax_vectors[:, i], info, axes=ax, show=False)
        plt.title('VM%i' % (i+1))
    ## Varimax timecourses
    respX_vmax =  np.stack([respX[i].T.dot(varimax_vectors)
                            for i in range(respX.shape[0])], axis=0).swapaxes(2, 1)
    trialX_vmax =  np.stack([trialX[i].T.dot(varimax_vectors)
                            for i in range(trialX.shape[0])], axis=0).swapaxes(2, 1)
    ## Varimax timecourse 1
    ax3 = fig.add_subplot(gs[3, :6])
    plt.sca(ax3)
    for i in range(9):
        eegf.plot_mean_sem(trialX_vmax[:, i], trial_epochs_csd.times, label='C %i' % (i+1))
    plt.vlines(0, linestyle='--', *plt.ylim())
    plt.legend(loc='upper left', prop={'size': 10})
    plt.xlim(-.5, 2)
    plt.xlabel('Time from onset (s)')
    ## Varimax timecourse 2
    ax4 = fig.add_subplot(gs[3, 6:])
    plt.sca(ax4)
    for i in range(9):
        eegf.plot_mean_sem(respX_vmax[:, i], response_epochs_csd.times, label='C %i' % (i+1))
    plt.vlines(0, linestyle='--', *plt.ylim())
    plt.legend(loc='upper left', prop={'size': 10})
    plt.xlim(-2, .5)
    plt.xlabel('Time to action (s)')
    ## Finish
    # plt.tight_layout()
    return fig

def pca_component_plot(component_ix: int, rotation_matrix: np.array, 
                       stimXr: np.array, respXr: np.array,
                       stim_epochs: mne.Epochs, response_epochs: mne.Epochs):
    '''Make a pretty plot.
    
    Pars:
    - component_ix: ID of the component to plot
    - rotation_matrix
    - stimXr: Rotated stimulus-locked data
    - respXr: Rotated response-locked data
    - stim_epochs, response_epochs: The original epoch objects
    '''
    fig, axes = plt.subplots(1, 3, figsize=(18, 3), gridspec_kw={'width_ratios':[1, 4, 4]})
    topomap(rotation_matrix[:, component_ix], response_epochs.info, axes=axes[0])
    axes[0].set_title('Component %i' % (component_ix+1))
    axes = axes[1:]
    Xs = [stimXr, respXr]
    timeses = [stim_epochs.times, response_epochs.times]
    xlabs = ['Time from stimulus', 'Time to response']
    for ax, X, times, xlab in zip(axes, Xs, timeses, xlabs):
        plt.sca(ax)
        eegf.plot_mean_sem(X[:, component_ix] * million, times)
        plt.hlines(0, *plt.xlim(), linestyle='dashed')
        plt.vlines(0, *plt.ylim(), linestyle='dashed')
        plt.xlabel(xlab)
        ax.set_yticklabels([])
    plt.tight_layout()
    return fig
