#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on February 26, 2025, at 16:28
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from core_code
#### Start of experiment

## Import packages
import numpy as np
import json
import os, shutil, subprocess
import atexit, signal
from datetime import datetime, timedelta
import pandas as pd
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'image_evolution'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '999',
    'run': '999',
    'condition': 'ratings',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'psychopy_output/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\dariu\\Documents\\PhD\\ImageGeneration\\repo\\behavioural_study\\clean_repo\\psychopy\\image_evolution.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1536, 864], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "start_text" ---
    intro_text = visual.TextStim(win=win, name='intro_text',
        text='Placeholder',
        font='Open Sans',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from core_code
    # Set global vars and process experiment parameters
    participant = int(expInfo['participant'])
    run = int(expInfo['run'])
    session = int(expInfo['session'])
    #condition = expInfo['condition']
    
    with open('../config.json') as f:
        config = json.load(f)
        
    data_dir = config["output_path"]
    output_path = os.path.join("..", data_dir, f"sub-{participant:02}", f"ses-{session:02}", f"run-{run}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise Exception("Target path for this participant, run, and session already exists!\nAborting...")
        
    max_iters = config["max_iters"]
    pop_size = config["pop_size"]
    mutation_size = config["mutation_size"]
    latent_size = config["latent_size"]
    
    instruction_text = config["instruction_text"]
    intro_text.text = instruction_text
    
    # Create output df
    latent_column_names = [f"x{i}" for i in range(latent_size)]
    df = pd.DataFrame(columns=[
        "participant",
        "session",
        "run",
        "generation",
        "img_num",
        "onset",
        "rating",
        "response_time",
        *latent_column_names])
    
    # Create timer
    timer = core.Clock()
    
    # --- Initialize components for Routine "generating" ---
    generating_fixation_cross = visual.ShapeStim(
        win=win, name='generating_fixation_cross', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "rating" ---
    slider = visual.Slider(win=win, name='slider',
        startValue=0, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='DarkGray', lineColor='Black', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    button = visual.ButtonStim(win, 
        text='Done', font='Arial',
        pos=(0, -0.25),
        letterHeight=0.05,
        size=(0.15, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='black', colorSpace='rgb',
        opacity=None,
        bold=False, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-2
    )
    button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "washout" ---
    washout_fixation_cross = visual.ShapeStim(
        win=win, name='washout_fixation_cross', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "set_image" ---
    set_image_fixation_cross = visual.ShapeStim(
        win=win, name='set_image_fixation_cross', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "save_data" ---
    data_fixation_cross = visual.ShapeStim(
        win=win, name='data_fixation_cross', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "end_text" ---
    outro = visual.TextStim(win=win, name='outro',
        text='Experiment complete.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "start_text" ---
    continueRoutine = True
    # update component parameters for each repeat
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    start_textComponents = [intro_text, key_resp]
    for thisComponent in start_textComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_text" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            key_resp.clock.reset()  # now t=0
        if key_resp.status == STARTED:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_textComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_text" ---
    for thisComponent in start_textComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from core_code
    # When trigger is received, set clock to zero
    timer.reset()
    
    # Record start time
    start_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
    with open(os.path.join(output_path, "run_start_time.txt"), 'w') as f:
        f.write(start_time)
    
    # Start collector subprocess
    collector = subprocess.Popen(['python', '../code/generator.py', output_path])
    atexit.register(os.kill, collector.pid, signal.CTRL_C_EVENT)
    
    
    # the Routine "start_text" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    generations = data.TrialHandler(nReps=max_iters, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='generations')
    thisExp.addLoop(generations)  # add the loop to the experiment
    thisGeneration = generations.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisGeneration.rgb)
    if thisGeneration != None:
        for paramName in thisGeneration:
            globals()[paramName] = thisGeneration[paramName]
    
    for thisGeneration in generations:
        currentLoop = generations
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisGeneration.rgb)
        if thisGeneration != None:
            for paramName in thisGeneration:
                globals()[paramName] = thisGeneration[paramName]
        
        # --- Prepare to start Routine "generating" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from generator_code
        # Set output path
        iteration = generations.thisN
        print(f"Beginning generation: {iteration}")
        
        this_gen_output_path = os.path.join(output_path, f"generation_{iteration:02}")
        while not os.path.exists(this_gen_output_path):
            core.wait(0.1)
        
        # keep track of which components have finished
        generatingComponents = [generating_fixation_cross]
        for thisComponent in generatingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "generating" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from generator_code
            # Wait for first image to be generated
            status_file = os.path.join(this_gen_output_path, "status.txt")
            
            while not os.path.isfile(status_file):
                core.wait(0.1)
                
            status = np.loadtxt(status_file, delimiter = ',')
            
            if status.shape[0] == 0:
                core.wait(0.1)
            elif status[0]:
                image_path = os.path.join(this_gen_output_path, "img_00.png")
                continueRoutine = False 
            else:
                core.wait(0.1)
            
            # *generating_fixation_cross* updates
            
            # if generating_fixation_cross is starting this frame...
            if generating_fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                generating_fixation_cross.frameNStart = frameN  # exact frame index
                generating_fixation_cross.tStart = t  # local t and not account for scr refresh
                generating_fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(generating_fixation_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'generating_fixation_cross.started')
                # update status
                generating_fixation_cross.status = STARTED
                generating_fixation_cross.setAutoDraw(True)
            
            # if generating_fixation_cross is active this frame...
            if generating_fixation_cross.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in generatingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "generating" ---
        for thisComponent in generatingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from generator_code
        # Set image path
        image_path = f"{this_gen_output_path}/img_00.png"
        
        # Initialize data arrays
        onset_times = np.zeros(pop_size)
        ratings = np.zeros(pop_size)
        response_times = np.zeros(pop_size)
        
        # the Routine "generating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=pop_size, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='trials')
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            image.setImage(image_path)
            # keep track of which components have finished
            trialComponents = [image, polygon]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + 3.0-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *polygon* updates
                
                # if polygon is starting this frame...
                if polygon.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon.frameNStart = frameN  # exact frame index
                    polygon.tStart = t  # local t and not account for scr refresh
                    polygon.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.started')
                    # update status
                    polygon.status = STARTED
                    polygon.setAutoDraw(True)
                
                # if polygon is active this frame...
                if polygon.status == STARTED:
                    # update params
                    pass
                
                # if polygon is stopping this frame...
                if polygon.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon.tStop = t  # not accounting for scr refresh
                        polygon.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon.stopped')
                        # update status
                        polygon.status = FINISHED
                        polygon.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "rating" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('rating.started', globalClock.getTime())
            # Run 'Begin Routine' code from rating_code
            ## Clear buffer
            #keys = event.getKeys(keyList=['1', '2', '3', '4', '6'])
            ## Define rating dict
            #rating_dict = {"1": 0, "2": 1, "3": 2, "4": 3, "6": 4}
            #
            slider.reset()
            # reset button to account for continued clicks & clear times on/off
            button.reset()
            # keep track of which components have finished
            ratingComponents = [slider, button]
            for thisComponent in ratingComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "rating" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from rating_code
                ## Check for keypress
                #keys = event.getKeys(keyList=['1', '2', '3', '4', '6'])
                #
                #if keys:
                #    ratings[trials.thisN] = rating_dict[keys[0]]
                #    response_times[trials.thisN] = timer.getTime()
                #    print(f"Rating is: {rating_dict[keys[0]]}")
                #    continueRoutine = False
                
                # *slider* updates
                
                # if slider is starting this frame...
                if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    slider.frameNStart = frameN  # exact frame index
                    slider.tStart = t  # local t and not account for scr refresh
                    slider.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider.started')
                    # update status
                    slider.status = STARTED
                    slider.setAutoDraw(True)
                
                # if slider is active this frame...
                if slider.status == STARTED:
                    # update params
                    pass
                # *button* updates
                
                # if button is starting this frame...
                if button.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    button.frameNStart = frameN  # exact frame index
                    button.tStart = t  # local t and not account for scr refresh
                    button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button.started')
                    # update status
                    button.status = STARTED
                    button.setAutoDraw(True)
                
                # if button is active this frame...
                if button.status == STARTED:
                    # update params
                    pass
                    # check whether button has been pressed
                    if button.isClicked:
                        if not button.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button.timesOn.append(button.buttonClock.getTime())
                            button.timesOff.append(button.buttonClock.getTime())
                        elif len(button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button.timesOff[-1] = button.buttonClock.getTime()
                        if not button.wasClicked:
                            # end routine when button is clicked
                            continueRoutine = False
                        if not button.wasClicked:
                            # run callback code when button is clicked
                            pass
                # take note of whether button was clicked, so that next frame we know if clicks are new
                button.wasClicked = button.isClicked and button.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in ratingComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "rating" ---
            for thisComponent in ratingComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('rating.stopped', globalClock.getTime())
            # Run 'End Routine' code from rating_code
            ## If no keys pressed, set data values as NA
            #if keys is None:
            #    ratings[trials.thisN] = np.NaN
            #    response_times[trials.thisN] = np.NaN
            #    print(f"Rating is: {pd.NA}")
            #
            #firstTrial = True
            
            ratings[trials.thisN] = slider.getRating()
            response_times[trials.thisN] = slider.getRT()
            trials.addData('slider.response', slider.getRating())
            trials.addData('slider.rt', slider.getRT())
            trials.addData('button.numClicks', button.numClicks)
            if button.numClicks:
               trials.addData('button.timesOn', button.timesOn)
               trials.addData('button.timesOff', button.timesOff)
            else:
               trials.addData('button.timesOn', "")
               trials.addData('button.timesOff', "")
            # the Routine "rating" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "washout" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('washout.started', globalClock.getTime())
            # keep track of which components have finished
            washoutComponents = [washout_fixation_cross]
            for thisComponent in washoutComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "washout" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *washout_fixation_cross* updates
                
                # if washout_fixation_cross is starting this frame...
                if washout_fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    washout_fixation_cross.frameNStart = frameN  # exact frame index
                    washout_fixation_cross.tStart = t  # local t and not account for scr refresh
                    washout_fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(washout_fixation_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'washout_fixation_cross.started')
                    # update status
                    washout_fixation_cross.status = STARTED
                    washout_fixation_cross.setAutoDraw(True)
                
                # if washout_fixation_cross is active this frame...
                if washout_fixation_cross.status == STARTED:
                    # update params
                    pass
                
                # if washout_fixation_cross is stopping this frame...
                if washout_fixation_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > washout_fixation_cross.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        washout_fixation_cross.tStop = t  # not accounting for scr refresh
                        washout_fixation_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'washout_fixation_cross.stopped')
                        # update status
                        washout_fixation_cross.status = FINISHED
                        washout_fixation_cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in washoutComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "washout" ---
            for thisComponent in washoutComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('washout.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "set_image" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('set_image.started', globalClock.getTime())
            # keep track of which components have finished
            set_imageComponents = [set_image_fixation_cross]
            for thisComponent in set_imageComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "set_image" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from set_next_image
                # Wait for next image to be generated
                if trials.thisN + 1 == pop_size:
                    print(f"Generation {iteration} finished.")
                    continueRoutine = False
                else:
                    status_file = os.path.join(this_gen_output_path, "status.txt")
                    while not os.path.isfile(status_file):
                        core.wait(0.1)
                        
                    status = np.loadtxt(status_file, delimiter = ',')
                    
                    if status.shape[0] == 0:
                        core.wait(0.1)
                    elif status[trials.thisN + 1]:
                        image_path = os.path.join(this_gen_output_path, f"img_{trials.thisN+1:02}.png")
                        continueRoutine = False  
                    else:
                        core.wait(0.1)
                
                
                
                # *set_image_fixation_cross* updates
                
                # if set_image_fixation_cross is starting this frame...
                if set_image_fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    set_image_fixation_cross.frameNStart = frameN  # exact frame index
                    set_image_fixation_cross.tStart = t  # local t and not account for scr refresh
                    set_image_fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(set_image_fixation_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'set_image_fixation_cross.started')
                    # update status
                    set_image_fixation_cross.status = STARTED
                    set_image_fixation_cross.setAutoDraw(True)
                
                # if set_image_fixation_cross is active this frame...
                if set_image_fixation_cross.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in set_imageComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "set_image" ---
            for thisComponent in set_imageComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('set_image.stopped', globalClock.getTime())
            # the Routine "set_image" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed pop_size repeats of 'trials'
        
        
        # --- Prepare to start Routine "save_data" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from data_code
        # Save onset times and ratings so that collector can begin processing fMRI data
        np.savetxt(os.path.join(this_gen_output_path, "onset_times.txt"), onset_times, delimiter=',')
        np.savetxt(os.path.join(this_gen_output_path, "ratings.txt"), ratings, delimiter=',')
        
        continueRoutine = False
        # keep track of which components have finished
        save_dataComponents = [data_fixation_cross]
        for thisComponent in save_dataComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "save_data" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *data_fixation_cross* updates
            
            # if data_fixation_cross is starting this frame...
            if data_fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                data_fixation_cross.frameNStart = frameN  # exact frame index
                data_fixation_cross.tStart = t  # local t and not account for scr refresh
                data_fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(data_fixation_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'data_fixation_cross.started')
                # update status
                data_fixation_cross.status = STARTED
                data_fixation_cross.setAutoDraw(True)
            
            # if data_fixation_cross is active this frame...
            if data_fixation_cross.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in save_dataComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "save_data" ---
        for thisComponent in save_dataComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # Run 'End Routine' code from data_code
        ## Save as dataframe
        print("Saving data...")
        df_info = pd.DataFrame(data = {
            "participant": np.repeat(participant, pop_size),
            "session": np.repeat(session, pop_size),
            "run": np.repeat(run, pop_size),
            "generation": np.repeat(generations.thisN, pop_size),
            "img_num": np.arange(pop_size),
            "onset": onset_times,
            "rating": ratings,
            "response_time": response_times
            })
        
        this_trial_latents = np.loadtxt(os.path.join(this_gen_output_path, "latents.txt"), delimiter=",")
        df_latents = pd.DataFrame(data = this_trial_latents, columns = latent_column_names)
        
        df_this_trial = pd.concat([df_info, df_latents], axis=1)
        df = pd.concat([df, df_this_trial], axis=0)
        df.to_csv(os.path.join(output_path, "all_data.csv"), sep=",")
        # the Routine "save_data" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed max_iters repeats of 'generations'
    
    
    # --- Prepare to start Routine "end_text" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    end_textComponents = [outro]
    for thisComponent in end_textComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_text" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *outro* updates
        
        # if outro is starting this frame...
        if outro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            outro.frameNStart = frameN  # exact frame index
            outro.tStart = t  # local t and not account for scr refresh
            outro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(outro, 'tStartRefresh')  # time at next scr refresh
            # update status
            outro.status = STARTED
            outro.setAutoDraw(True)
        
        # if outro is active this frame...
        if outro.status == STARTED:
            # update params
            pass
        
        # if outro is stopping this frame...
        if outro.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > outro.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                outro.tStop = t  # not accounting for scr refresh
                outro.frameNStop = frameN  # exact frame index
                # update status
                outro.status = FINISHED
                outro.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_textComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_text" ---
    for thisComponent in end_textComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='comma')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
