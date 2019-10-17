#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.83.02), on March 14, 2018, at 12:16

...In reality, it was modified extesively from a basic Builder template, and so probably will no longer open in Builder.
 - Eoin

Notes
- A BET sticker is placed on the ↑ key, and ↓ on the down key.
- We used python2, and PsychoPy 1.85 to run this experiment. It is not
  compatible with python3.
- We use PyDAQmx to send triggers over a National Instruments DAQ box.
  daq.py, included in this directory, will try to use this module, and
  if it fails will just print to the console at times when a trigger
  would have been sent.

"""
## Imports
from __future__ import absolute_import, division
## This must happen here, as importing psychopy interferes with printing
print 'Enter participant number → [ENTER] → Order (0 or 1) → [ENTER].'
from psychopy import visual, parallel
from psychopy import locale_setup, sound, core, data, event, logging
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding
import pandas as pd
from time import time as now
from daq import *

## Ensure that relative paths start from the same directory as this script. Maybe not needed.
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

## Session information
dummy = 0 # Set to 1 to skip subject details while developing
if dummy:
    participant = 901
    cb = 1
else:
    # Both must be numeric!
    participant = input('Participant number?\n')
    cb = input('Order? (0 or 1)\n')

assert(cb in [0, 1]) , 'Order must be 0 or 1!'

expInfo = {
    'participant' : participant,
    'cb'   : cb,
    'date'        : data.getDateStr(),
    'score_delta' : 0,
    'expName'     : 'Roulette',
    'block_nr'    : 0,
    'block_half'  : 0,
    'trial_nr'    : 0
}

## Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data' + os.sep + '%s_%s' % (
    expInfo['participant'], expInfo['date'])
if not os.path.exists('data'):
    os.mkdir('data')
## Save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

## Set up triggers
task = setup_triggers()
triggers = {
    'exp_start'   : 1,
    'stim_start'  : 2,
    'response'    : 3,
    'stim_end'    : 4,
    'break_start' : 5,
    'break_end'   : 6,
    'exp_end'     : 7,
    'vol_start'   : 11,
    'vol_act'     : 12,
    'vol_end'     : 13
}
# Useful constants
black = (-1, -1, -1)
green = (-1, .5, -1)
red   = (1, -1, -1)
grey  = (.3, .3, .3)

key_map = {
    'bet' : 'up',
    'pass' : 'down'
}
action_keys = [key_map['bet'], key_map['pass']]

instructions = {
    'block0' : "press the BET button if you want to accept the gamble. To skip the gamble, don't press anything.",
    'block1' : "press the PASS button if you want to skip the gamble. To accept the gamble, don't press anything."
}

response_map = {'left':0, 'right':1}
# elif condition==2:
#     default_response, alternative_response = 2, -1

# Setup stimuli
design = pd.read_csv('design_data.csv').sample(frac=1)
design.index = range(len(design))
trials_per_block = 75
nblocks = int(len(design) / trials_per_block)
design['block'] = np.repeat(range(nblocks), trials_per_block)
break_every = 15

global_vars = {
    'punish'        : -20,
    'start_score'   : 100,
    'feedback_time' : 1,
    'iti' : [1., 1., 1.],
    'gamble_time': 4.995
};
expInfo['score'] = global_vars['start_score']

log_vars = [
    'date',
    'participant',
    'expName',
    'frameRate',
    'cb',
    'condition',
    'block_nr',
    'block_half',
    'trial_nr',
    'v_win',
    'p_win',
    'action',
    'response',
    'rt',
    'outcome',
    'score_delta',
    'score',
    'visible',
    'stim_id',
    'n',
    'timenow'
]

def flip_coin(p):
    return 1*(np.random.uniform() < p)

# Setup the Window
win = visual.Window(
    # winType = 'pygame',
    # size=[1600, 1200], fullscr=True, screen=0,
    size=[1200, 800], fullscr=False, screen=0,
    allowGUI=True, allowStencil=False,
    monitor='testMonitor',
    # color=[.75, .75, .75],
    color=[.5, .5, .5],
    colorSpace='rgb',
    blendMode='avg', useFBO=True,
    units='height')
expInfo['frameRate'] = win.getActualFrameRate()

# Setup basic instructions
instructions_text = visual.TextStim(
    win=win, name='instructions_text',
    text="Ready?\nPress SPACE to begin.",
    font='Arial', alignHoriz='center',
    pos=[0, 0], height=0.04, wrapWidth=.95, color='black',
    depth=-1.0);

# Basic trial routines
trial_clock = core.Clock()
circle_rad = .06

circle_red = visual.Circle(win=win, name='circle_red', radius = circle_rad, pos=[0,0],
                           depth = -2.0, fillColor=red, lineColor=black, edges=100)
circle_grey = visual.Circle(win=win, name='circle_grey', radius = circle_rad, pos=[0,0],
                            depth = -2.0, fillColor=grey, lineColor=black, edges=100)
circle_black = visual.Circle(win=win, name='circle_black', pos=[0,0],
                             radius = circle_rad,#+.002,
                             lineWidth = 10,
                             depth = -5.0, fillColor=None, lineColor=black, edges=100)
text_win_unknown = visual.TextStim(win=win, text='???',
                                   height=.05, pos=[0, 0], color=black)
fix_text = visual.TextStim(win=win, text='+', height=.05, pos=[0, 0], color=black)
# fix_textB = visual.TextStim(win=win, text='+', height=.05, pos=[0, 0], color=black, bold=1)
fix_textX = visual.TextStim(win=win, text='X', height=.05, pos=[0, 0], color=black, bold=1)


def make_trial_stimuli(p_win, win_amount):
    lose_amount = 10
    text_win = visual.TextStim(win=win, text='+%i' % win_amount,
                               height=.025, pos=[0,.0375], color=black)
    text_lose = visual.TextStim(win=win, text='-%i' % lose_amount,
                                height=.025, pos=[0,-.0375], color=black)
    win_angle = 360 * p_win
    arc_green = visual.ShapeStim(win=win, name='arc_green', pos=[0,0],
                                 vertices=generate_arc_vertices(win_angle),
                                 depth = -1.0, fillColor=green, lineColor=black)
    arc_grey = visual.ShapeStim(win=win, name='arc_grey', pos=[0,0],
                                 vertices=generate_arc_vertices(win_angle),
                                 depth = -1.0, fillColor=grey, lineColor=black)
    text_win_unknown = visual.TextStim(win=win, text='+??',
                                   height=.025, pos=[0, .03], color=black)
    return arc_green, text_win, text_lose, arc_grey

def generate_arc_vertices(angle, mid=90, radius = circle_rad):
    vertices = [[0,0]]
    a1 = mid + .5*angle
    a2 = mid - .5*angle
    for a in np.linspace(a1, a2, 20):
        r = np.deg2rad(a)
        v = [np.cos(r), np.sin(r)]
        vertices.append(v)
    return np.array(vertices) * radius

# text_score = visual.TextStim(win=win, text='100',
#                              height=.025, pos=[0, .15], color=black)
text_delta_score = visual.TextStim(win=win, text='+10',
                             height=.025, pos=[0, .1], color=black)

icon_x, icon_y = .1, -.15
icon_h = .03
# hand_y = icon_y - icon_h
# # hand_size = hand_w, hand_h = (93, 122)
# hand_size = hand_w, hand_h = np.array([93, 122]) / 93
# hand_size *= .03

pass_icon = visual.TextStim(win=win, text='Pass',
                            height=icon_h, pos=[-icon_x, icon_y], color=black)
pass_icon_box = visual.Rect(win=win, width=icon_h*2.5, height=icon_h*1.25,
                            pos=[-icon_x, icon_y], lineColor=black)
bet_icon = visual.TextStim(win=win, text='Bet',
                           height=icon_h, pos=[icon_x, icon_y], color=black)
bet_icon_box = visual.Rect(win=win, width=icon_h*2.5, height=icon_h*1.25,
                           pos=[icon_x, icon_y], lineColor=black)
# hand_icon = visual.ImageStim(win=win, image='imgs/hand.png', size=hand_size,
#                              pos=[icon_x, hand_y])

def merge_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def do_trial(gamble, expInfo):
    condition = expInfo['condition']
    default_response, alternative_response = expInfo['responses']
    action, response, rt = 0, default_response, -1
    
    arc_green, text_win, text_lose, arc_grey = make_trial_stimuli(
        p_win=gamble['p_win'], win_amount=gamble['v_win'])
    if gamble['visible']:
        trial_stimuli = [circle_red, arc_green, text_win, text_lose]
    else:
        trial_stimuli = [circle_grey, text_win_unknown]
    if condition == 1:
        trial_stimuli.append(circle_black)
    # Outcome
    outcome = flip_coin(gamble['p_win'])
    expInfo['outcome'] = outcome
    if outcome:
        outcome_stimuli = [circle_grey, arc_green, text_win, circle_black]
    else:
        outcome_stimuli = [circle_red, arc_grey, text_lose, circle_black]
    # ITI
    win.flip()
    core.wait(global_vars['iti'][0])
    fix_text.draw()
    win.flip()
    core.wait(global_vars['iti'][1]-.1)
    win.flip()
    core.wait(.1)
    fix_text.draw()
    win.flip()
    core.wait(global_vars['iti'][2])
    # Start trial
    for s in trial_stimuli:
        s.draw()
    win.flip()
    send_trigger(triggers['stim_start'], task)
    trial_clock.reset()
    # Wait...
    # keys = event.waitKeys(maxWait=global_vars['gamble_time'])
    keys = event.waitKeys(maxWait=global_vars['gamble_time'],
                          keyList=['q', action_keys[condition]])
    if keys is not None:
        # print keys
        if 'q' in keys:
            core.quit()
            return None
        rt = trial_clock.getTime()
        send_trigger(triggers['response'], task)
        print 'Response!'
        if action_keys[condition] in keys:
            action = 1
            response = alternative_response
            if condition == 0:
                trial_stimuli.append(circle_black)
            else:
                trial_stimuli = trial_stimuli[:-1]
        remaining_time = global_vars['gamble_time'] - rt
        for s in trial_stimuli:
            s.draw()
        # draw_button_icons(response, decided=False)
        win.flip()
        core.wait(remaining_time)
    decided = True
    if response==1:
        for s in outcome_stimuli:
            s.draw()
        score_delta = [-10, gamble['v_win']][outcome]
        expInfo['score'] += score_delta
        expInfo['score_delta'] = score_delta
        txt = '%i' % score_delta
        tc = [red, green][outcome]
        if score_delta > 0:
            txt = '+' + txt
        text_delta_score.text = txt
        text_delta_score.color = tc
        text_delta_score.draw()
    win.flip()
    send_trigger(triggers['stim_end'], task)
    core.wait(global_vars['feedback_time'])
    expInfo['timenow'] = now()
    expInfo = merge_dicts(expInfo, gamble)
    expInfo['action'] = action
    expInfo['response'] = response
    expInfo['rt'] = rt
    return expInfo

def do_block(block_df, expInfo):
    block_df = block_df.sample(frac=1).copy()
    if cb == 0:
        conditions = [0, 1]
    else:
        conditions = [1, 0]
    expInfo['block_half'] = 0
    for condition in conditions:
        expInfo['condition'] = condition
        if condition==0:
            expInfo['responses'] = [0, 1]
        elif condition==1:
            expInfo['responses'] = [1, 0]
        take_break(expInfo, newblock=True)
        expInfo['condition'] = condition
        for i, gamble in block_df.iterrows():
            expInfo = do_trial(gamble, expInfo)
            dataline = ','.join([str(expInfo[v]) for v in log_vars])
            datafile.write(dataline + '\n')
            datafile.flush()
            expInfo['trial_nr'] += 1
            if expInfo['trial_nr'] > 0 and expInfo['trial_nr'] % break_every == 0:
                take_break(expInfo)
        expInfo['block_half'] = 1
    expInfo['block_nr'] += 1
    return expInfo

# def show_block_instructions(condition):
#     txt = instructions['block%i' % condition]
#     instructions_text.text = 'For the next few rounds, ' + txt
#     instructions_text.draw()
#     win.flip()
#     keys = event.waitKeys(keyList=['space', 'q'])

break_text = "\
Time for a break. Please let the experimenter know you've reached this stage.\n\
\n\
When you're ready to go on, please read the following instructions carefully.\n\
Press SPACE to continue."
def take_break(expInfo, newblock=False):
    trial_nr, condition = expInfo['trial_nr'], expInfo['condition']
    if newblock:
        instructions_text.text = break_text
        prefix = 'For the next few rounds, '
        instructions_text.draw()
        win.flip()
        keys = event.waitKeys(keyList=['space'])
        if 'q' in keys:
            core.quit()
            return None
    else:
        prefix = 'Remember, '
    txtA = u"You've completed %i of %i rounds.\nYour balance is £%.2f.\n\n" % (
        trial_nr, len(design)*2, expInfo['score']/100.)
    txtB = prefix + instructions['block%i' % condition] + '\n\n'
    if trial_nr == 0:
        txtA = ''
    elif trial_nr % trials_per_block == 0 and newblock == False:
        txtB = ''
    if newblock:
        txt = txtB + txtA
    else:
        txt = txtA + txtB
    txt += u"Press SPACE to continue."
    pause = visual.TextStim(
        win=win, name='mini_pause',
        text=txt,
        pos=[0, 0], height=0.03, wrapWidth=.95, ori=0,
        color='black', colorSpace='rgb', depth=0.0)
    pause.draw()
    win.flip()
    send_trigger(triggers['break_start'], task)
    keys = event.waitKeys(keyList=['space'])
    if 'q' in keys:
        core.quit()
        return None
    # draw_button_icons(default_response)
    win.flip()
    send_trigger(triggers['break_end'], task)
    core.wait(1)
    
datafile = open(filename+'.csv', 'w')
datafile.write(','.join(log_vars) + '\n')
datafile.flush()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

# Show instructions
instructions_text.draw()
win.flip()
keys = event.waitKeys(keyList=['space'])
# while 1:
#     keys = event.waitKeys()
#     # keys = event.waitKeys(keyList=['space'])
#     print keys
#     if 'space' in keys:
#         break
send_trigger(triggers['exp_start'], task)
for ib, block in design.groupby('block'):
    expInfo = do_block(block, expInfo)
send_trigger(triggers['exp_end'], task)

vol_clock = core.Clock()

instructions_text.text = "\
You've reached the end of the game.\nThere's just one more thing to do..."
instructions_text.draw()
win.flip()
keys = event.waitKeys(keyList=['space'])

instructions_text.text = "\
Press SPACE to begin."
instructions_text.draw()
keys = event.waitKeys(keyList=['space'])
win.flip()

send_trigger(triggers['vol_start'], task)

fix_text.draw()
win.flip()
while True:
    keys = event.waitKeys(keyList=['space', 'q'])
    send_trigger(triggers['vol_act'], task)
    t_now = vol_clock.getTime()
    print t_now
    fix_textX.draw()
    win.flip()
    core.wait(.1)
    fix_text.draw()
    win.flip()
    if (t_now / 60) > 5:
        break
    elif 'q' in keys:
        core.quit()

send_trigger(triggers['vol_end'], task)
print 'Final score:', expInfo['score']
win.close()
core.quit()
