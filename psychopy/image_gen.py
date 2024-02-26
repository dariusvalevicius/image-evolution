#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on February 26, 2024, at 12:36
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

# Run 'Before Experiment' code from generate_images
#### Start of experiment

## Import packages
import torch
import numpy as np
from diffusers import StableUnCLIPImg2ImgPipeline
from sklearn.decomposition import PCA
import pickle as pk
import os, shutil
from datetime import datetime, timedelta
import pandas as pd
import subprocess
import atexit
import signal

## Classes
class Image(visual.ImageStim):
    # Embedding property
    embedding = None


## Functions
def generate_image(embedding, image_name):
    # Ensure correct size and datatype
    embedding = torch.tensor(np.reshape(embedding, (1,np.size(embedding))), dtype=torch.float16)
    embedding = embedding.to(device)
    # Generate and save image
    images = pipe(image_embeds=embedding, num_inference_steps=diffusion_steps).images
    images[0].save(image_name)
    
def new_generation(fitness, x, iteration):
    pop_size_local = pop_size - 1
    top_n = int(pop_size_local/2)
    
    # Get top vectors
    idx = np.argsort(fitness)[::-1]
    fitness_sorted = fitness[idx]
    x_sorted = x[idx, :]

    fitness_top = fitness_sorted[:top_n]
    x_top = x_sorted[:top_n, :]

    # Compute recombination probability weights
    median = np.median(fitness)
    fitness_relative = np.clip(fitness_top - median, 0, None)
    
    weights = np.zeros(np.size(fitness_relative))
    
    if np.sum(fitness_relative) > 0:
        weights = fitness_relative / np.sum(fitness_relative)
    else:
        weights = np.ones(np.size(fitness_relative)) / pop_size_local
    
    mean = np.sum((x_top.T * weights).T, axis=0)
    next_x = np.random.multivariate_normal(mean, mutation_size * cov, size=pop_size_local)
    
    return next_x
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'image_gen'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'dev_mode': '0',
    'max_iters': '10',
    'pop_size': '8',
    'scr_attached': '1',
    'diffusion_steps': '21',
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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\eclip\\Documents\\ImageGeneration\\image-evolution\\psychopy\\image_gen.py',
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
            size=[1920, 1080], fullscr=False, screen=0,
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
    win.mouseVisible = True
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
    intro = visual.TextStim(win=win, name='intro',
        text='Welcome to the experiment.\n\nYou will be presented with a series of images. After each image, rate the fear you experienced on the sliding scale.\n\nPress [SPACE] to start.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "block_start" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='You are about to begin a block. This will take 10-20 minutes.\n\nPress [SPACE] when you are ready to begin.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "clear_data" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "generate_text" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='Generating...',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "generating" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from generate_images
    ## Set to fullscreen
    win.fullscr = True
    win.flip()
    
    ## Set variables
    global model_path
    model_path = "../../stable-diffusion-2-1-unclip-small"
    
    global participant
    participant = int(expInfo['participant'])
    if not os.path.exists(f"data/{participant}"):
        os.makedirs(f"data/{participant}")
    
    global dev_mode
    dev_mode = int(expInfo['dev_mode'])
    #global conditions_selected
    #conditions_selected = expInfo['conditions_selected']
    global scr_attached
    scr_attached = int(expInfo['scr_attached'])
    global diffusion_steps
    diffusion_steps = int(expInfo['diffusion_steps'])
    
    global max_iters
    max_iters = int(expInfo['max_iters'])
    global pop_size
    pop_size = int(expInfo['pop_size']) + 1 # Add one dummy image
    global mutation_size
    mutation_size = 0.3
    global vec_size
    vec_size = 80
    global embedding_size
    embedding_size = 768
    global eda_samples
    eda_samples = 25
    
    # Load PCA and covariance matrix
    global pca
    pca = pk.load(open("pca.pkl",'rb')) 
    global cov
    cov = np.loadtxt("covariance_matrix.txt")
    
    
    ## Start osc_server subprocess
    process = subprocess.Popen(['python','osc_server.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    atexit.register(os.kill, process.pid, signal.CTRL_C_EVENT)
    
    ## Set unCLIP device
    if not dev_mode:
        global device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
    
        ## Load model
        global pipe
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variation="fp16")
        pipe = pipe.to(device)
        # pipe.enable_model_cpu_offload()
        # pipe.enable_vae_slicing()
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text_4 = visual.TextStim(win=win, name='text_4',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_8 = visual.TextStim(win=win, name='text_8',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "rating" ---
    slider = visual.Slider(win=win, name='slider',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=("0","","","","","5","","","","","10"), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=0.0,
        style='slider', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='DarkGray', lineColor='Black', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Please rate the fear you experienced.',
        font='Open Sans',
        pos=(0, 0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    button = visual.ButtonStim(win, 
        text='Done', font='Open Sans',
        pos=(0, -0.25),
        letterHeight=0.05,
        size=(0.25, 0.15), borderWidth=3.0,
        fillColor='lightgrey', borderColor='black',
        color='black', colorSpace='rgb',
        opacity=None,
        bold=False, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-4
    )
    button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "check_eda" ---
    check_eda_text = visual.TextStim(win=win, name='check_eda_text',
        text='There is a problem with the EDA connection.\n\nPlease alert the experimenter before continuing with the experiment.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "save_data" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text='Saving...',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "end_text" ---
    outro = visual.TextStim(win=win, name='outro',
        text='The experiment is now over.\n\nThank you for participating!',
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
    start_textComponents = [intro, key_resp]
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
        
        # *intro* updates
        
        # if intro is starting this frame...
        if intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro.frameNStart = frameN  # exact frame index
            intro.tStart = t  # local t and not account for scr refresh
            intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro, 'tStartRefresh')  # time at next scr refresh
            # update status
            intro.status = STARTED
            intro.setAutoDraw(True)
        
        # if intro is active this frame...
        if intro.status == STARTED:
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
    # the Routine "start_text" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    block = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions.csv'),
        seed=None, name='block')
    thisExp.addLoop(block)  # add the loop to the experiment
    thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in block:
        currentLoop = block
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
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "block_start" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('block_start.started', globalClock.getTime())
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # keep track of which components have finished
        block_startComponents = [text_9, key_resp_2]
        for thisComponent in block_startComponents:
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
        
        # --- Run Routine "block_start" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
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
            for thisComponent in block_startComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_start" ---
        for thisComponent in block_startComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('block_start.stopped', globalClock.getTime())
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        block.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            block.addData('key_resp_2.rt', key_resp_2.rt)
            block.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "block_start" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "clear_data" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        # Reset data for new loop
        
        all_embeddings = np.zeros((max_iters, pop_size, vec_size))
        all_ratings = np.zeros((max_iters, pop_size))
        all_scr = np.zeros((max_iters, pop_size))
        all_scr_means = np.zeros((max_iters, pop_size))
        scr_data = np.zeros((max_iters, pop_size, eda_samples))
        all_hr_means = np.zeros((max_iters, pop_size))
        all_temp_means = np.zeros((max_iters, pop_size))
        
        failed_trials = []
        # keep track of which components have finished
        clear_dataComponents = [text_5]
        for thisComponent in clear_dataComponents:
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
        
        # --- Run Routine "clear_data" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # if text_5 is stopping this frame...
            if text_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_5.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_5.tStop = t  # not accounting for scr refresh
                    text_5.frameNStop = frameN  # exact frame index
                    # update status
                    text_5.status = FINISHED
                    text_5.setAutoDraw(False)
            
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
            for thisComponent in clear_dataComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "clear_data" ---
        for thisComponent in clear_dataComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
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
            
            # set up handler to look after randomisation of conditions etc
            repeat_loop = data.TrialHandler(nReps=100.0, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='repeat_loop')
            thisExp.addLoop(repeat_loop)  # add the loop to the experiment
            thisRepeat_loop = repeat_loop.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisRepeat_loop.rgb)
            if thisRepeat_loop != None:
                for paramName in thisRepeat_loop:
                    globals()[paramName] = thisRepeat_loop[paramName]
            
            for thisRepeat_loop in repeat_loop:
                currentLoop = repeat_loop
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
                # abbreviate parameter names if possible (e.g. rgb = thisRepeat_loop.rgb)
                if thisRepeat_loop != None:
                    for paramName in thisRepeat_loop:
                        globals()[paramName] = thisRepeat_loop[paramName]
                
                # --- Prepare to start Routine "generate_text" ---
                continueRoutine = True
                # update component parameters for each repeat
                # keep track of which components have finished
                generate_textComponents = [text_6]
                for thisComponent in generate_textComponents:
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
                
                # --- Run Routine "generate_text" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.1:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text_6* updates
                    
                    # if text_6 is starting this frame...
                    if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_6.frameNStart = frameN  # exact frame index
                        text_6.tStart = t  # local t and not account for scr refresh
                        text_6.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        text_6.status = STARTED
                        text_6.setAutoDraw(True)
                    
                    # if text_6 is active this frame...
                    if text_6.status == STARTED:
                        # update params
                        pass
                    
                    # if text_6 is stopping this frame...
                    if text_6.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_6.tStartRefresh + 0.1-frameTolerance:
                            # keep track of stop time/frame for later
                            text_6.tStop = t  # not accounting for scr refresh
                            text_6.frameNStop = frameN  # exact frame index
                            # update status
                            text_6.status = FINISHED
                            text_6.setAutoDraw(False)
                    
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
                    for thisComponent in generate_textComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "generate_text" ---
                for thisComponent in generate_textComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.100000)
                
                # --- Prepare to start Routine "generating" ---
                continueRoutine = True
                # update component parameters for each repeat
                # Run 'Begin Routine' code from generate_images
                iteration = generations.thisN
                
                repeat = False
                
                ## Generate vectors
                if iteration == 0:
                    this_trial_embeddings = np.random.multivariate_normal(np.zeros(vec_size), cov, size=pop_size)    
                else:
                    ## CMA algorithm
                    fitness = []
                    if condition == "selfreport" or not scr_attached:
                        fitness = all_ratings[iteration - 1, 1:] # Skip dummy image
                    elif condition == "scr":
                        fitness = all_scr[iteration - 1, 1:]
                    elif condition == "combined":
                        scr_fitness = all_scr[iteration - 1, 1:]
                        scr_fitness_norm = np.interp(scr_fitness, (np.min(scr_fitness), np.max(scr_fitness)), (0, 1))
                        selfreport_fitness = all_ratings[iteration - 1, 1:]
                        selfreport_fitness_norm = np.interp(selfreport_fitness, (np.min(selfreport_fitness), np.max(selfreport_fitness)), (0, 1))
                        fitness = scr_fitness_norm + selfreport_fitness_norm # Get sum of two scores
                    elif condition == "control":
                        fitness = np.random.uniform(size = pop_size-1)
                    else:
                        raise Exception("Error: Invalid condition name")
                    embeddings = all_embeddings[iteration - 1, 1:, :]
                
                    # Create a dummy image from the mean of the previous generation
                    dummy_embedding = np.mean(all_embeddings[iteration-1,1:,:], axis=0)   
                    this_trial_embeddings = np.vstack((dummy_embedding, new_generation(fitness, embeddings, iteration)))
                
                # Concatenate new generation embeddings
                all_embeddings[iteration, :, :] = this_trial_embeddings
                
                ## Generate batch of images
                for i in range(pop_size):
                    image_name = f"images/num-{i}.png"
                    pcs = this_trial_embeddings[i,:]
                    embedding = pca.inverse_transform(pcs)
                    if not dev_mode:
                        generate_image(embedding, image_name)
                
                # Reset the EDA file after each use to avoid bloating it
                with open("eda_data.txt", 'w') as file:
                    pass
                with open("hr_data.txt", 'w') as file:
                    pass
                with open("temp_data.txt", 'w') as file:
                    pass
                with open("samp_data.txt", 'w') as file:
                    pass
                continueRoutine = False
                # keep track of which components have finished
                generatingComponents = [text_2]
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
                    
                    # *text_2* updates
                    
                    # if text_2 is starting this frame...
                    if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_2.frameNStart = frameN  # exact frame index
                        text_2.tStart = t  # local t and not account for scr refresh
                        text_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        text_2.status = STARTED
                        text_2.setAutoDraw(True)
                    
                    # if text_2 is active this frame...
                    if text_2.status == STARTED:
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
                    # Run 'Begin Routine' code from code
                    ## Generate batch of images
                    
                    ## Variables
                    #iteration = generations.thisN
                    trial = trials.thisN
                    #embedding = this_gen_embeddings[trial, :]
                        
                    ## Set image
                    image_name = f"images/num-{trial}.png"
                    
                    
                    
                    image.setImage(image_name)
                    # keep track of which components have finished
                    trialComponents = [image, text_4, text_8]
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
                    while continueRoutine and routineTimer.getTime() < 5.0:
                        # get current time
                        t = routineTimer.getTime()
                        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                        # update/draw components on each frame
                        
                        # *image* updates
                        
                        # if image is starting this frame...
                        if image.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
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
                        
                        # *text_4* updates
                        
                        # if text_4 is starting this frame...
                        if text_4.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                            # keep track of start time/frame for later
                            text_4.frameNStart = frameN  # exact frame index
                            text_4.tStart = t  # local t and not account for scr refresh
                            text_4.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            text_4.status = STARTED
                            text_4.setAutoDraw(True)
                        
                        # if text_4 is active this frame...
                        if text_4.status == STARTED:
                            # update params
                            pass
                        
                        # if text_4 is stopping this frame...
                        if text_4.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_4.tStartRefresh + 1-frameTolerance:
                                # keep track of stop time/frame for later
                                text_4.tStop = t  # not accounting for scr refresh
                                text_4.frameNStop = frameN  # exact frame index
                                # update status
                                text_4.status = FINISHED
                                text_4.setAutoDraw(False)
                        
                        # *text_8* updates
                        
                        # if text_8 is starting this frame...
                        if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_8.frameNStart = frameN  # exact frame index
                            text_8.tStart = t  # local t and not account for scr refresh
                            text_8.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            text_8.status = STARTED
                            text_8.setAutoDraw(True)
                        
                        # if text_8 is active this frame...
                        if text_8.status == STARTED:
                            # update params
                            pass
                        
                        # if text_8 is stopping this frame...
                        if text_8.status == STARTED:
                            # is it time to stop? (based on global clock, using actual start)
                            if tThisFlipGlobal > text_8.tStartRefresh + 1.0-frameTolerance:
                                # keep track of stop time/frame for later
                                text_8.tStop = t  # not accounting for scr refresh
                                text_8.frameNStop = frameN  # exact frame index
                                # update status
                                text_8.status = FINISHED
                                text_8.setAutoDraw(False)
                        
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
                    # Run 'End Routine' code from code
                    ## Update output matrices
                    #all_embeddings[iteration,trial,:] = embedding
                    
                    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                    if routineForceEnded:
                        routineTimer.reset()
                    else:
                        routineTimer.addTime(-5.000000)
                    
                    # --- Prepare to start Routine "rating" ---
                    continueRoutine = True
                    # update component parameters for each repeat
                    thisExp.addData('rating.started', globalClock.getTime())
                    slider.reset()
                    # setup some python lists for storing info about the mouse
                    mouse.x = []
                    mouse.y = []
                    mouse.leftButton = []
                    mouse.midButton = []
                    mouse.rightButton = []
                    mouse.time = []
                    gotValidClick = False  # until a click is received
                    # Run 'Begin Routine' code from code_2
                    ## Get SCR amplitudes
                    
                    # Get start and end times of trial
                    current_time = datetime.now().time()
                    dummy_datetime = datetime.combine(datetime.today(), current_time)
                    # Get relevant times
                    end_time = dummy_datetime.time()
                    start_time = (dummy_datetime - timedelta(seconds=5)).time()
                    peak_start_time = (dummy_datetime - timedelta(seconds=4)).time()
                    peak_end_time = (dummy_datetime - timedelta(seconds=1)).time()
                    
                    ## Load EDA data
                    eda_data = pd.read_csv("eda_data.txt", names=["eda", "time"])
                    samp_data = pd.read_csv("samp_data.txt", names=["samp", "time"])
                    hr_data = pd.read_csv("hr_data.txt", names=["hr", "time"])
                    temp_data = pd.read_csv("temp_data.txt", names=["temp", "time"])
                    
                    # Convert the 'time' column to datetime format
                    eda_data['time'] = pd.to_datetime(eda_data['time'], format='%H:%M:%S.%f').dt.time
                    samp_data['time'] = pd.to_datetime(samp_data['time'], format='%H:%M:%S.%f').dt.time
                    hr_data['time'] = pd.to_datetime(hr_data['time'], format='%H:%M:%S.%f').dt.time
                    temp_data['time'] = pd.to_datetime(temp_data['time'], format='%H:%M:%S.%f').dt.time
                    
                    # Filter the DataFrame based on the time range
                    peak_data = eda_data[(eda_data['time'] >= peak_start_time) & (eda_data['time'] <= peak_end_time)]
                    all_data = eda_data[(eda_data['time'] >= start_time) & (eda_data['time'] <= end_time)]
                    peak_samp = samp_data[(samp_data['time'] >= peak_start_time) & (samp_data['time'] <= peak_end_time)]
                    
                    hr_data = hr_data[(hr_data['time'] >= start_time) & (hr_data['time'] <= end_time)]
                    temp_data = temp_data[(temp_data['time'] >= start_time) & (temp_data['time'] <= end_time)]
                    
                    # Get average baseline SCR
                    scr_mean = all_data['eda'].mean()
                    hr_mean = hr_data['hr'].mean()
                    temp_mean = temp_data['temp'].mean()
                    
                    
                    
                    # Get the max SCR relative to baseline
                    scr_score = 0
                    if not peak_samp['samp'].empty:
                        scr_score = peak_samp['samp'].sum()
                        
                    ## Resample the full data and write to array
                    
                    # Calculate the average pooling factor
                    target_samples = eda_samples
                    arr = all_data['eda'].values
                    
                    scr_resampled = np.zeros(target_samples)
                    if arr.any() and len(arr) >= target_samples:
                        factor = arr.shape[0] // target_samples
                        
                        # Reshape the array into blocks
                        reshaped_arr = arr[:target_samples * factor].reshape(-1, factor)
                        
                        # Take the average value within each block
                        scr_resampled = np.mean(reshaped_arr, axis=1)
                    else:
                        repeat = True
                        trials.finished = True
                    
                    # reset button to account for continued clicks & clear times on/off
                    button.reset()
                    # keep track of which components have finished
                    ratingComponents = [slider, text_3, mouse, button]
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
                        
                        # *slider* updates
                        
                        # if slider is starting this frame...
                        if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            slider.frameNStart = frameN  # exact frame index
                            slider.tStart = t  # local t and not account for scr refresh
                            slider.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            slider.status = STARTED
                            slider.setAutoDraw(True)
                        
                        # if slider is active this frame...
                        if slider.status == STARTED:
                            # update params
                            pass
                        
                        # *text_3* updates
                        
                        # if text_3 is starting this frame...
                        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            text_3.frameNStart = frameN  # exact frame index
                            text_3.tStart = t  # local t and not account for scr refresh
                            text_3.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            text_3.status = STARTED
                            text_3.setAutoDraw(True)
                        
                        # if text_3 is active this frame...
                        if text_3.status == STARTED:
                            # update params
                            pass
                        # *mouse* updates
                        
                        # if mouse is starting this frame...
                        if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                            # keep track of start time/frame for later
                            mouse.frameNStart = frameN  # exact frame index
                            mouse.tStart = t  # local t and not account for scr refresh
                            mouse.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                            # update status
                            mouse.status = STARTED
                            mouse.mouseClock.reset()
                            prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
                        if mouse.status == STARTED:  # only update if started and not finished!
                            buttons = mouse.getPressed()
                            if buttons != prevButtonState:  # button state changed?
                                prevButtonState = buttons
                                if sum(buttons) > 0:  # state changed to a new click
                                    x, y = mouse.getPos()
                                    mouse.x.append(x)
                                    mouse.y.append(y)
                                    buttons = mouse.getPressed()
                                    mouse.leftButton.append(buttons[0])
                                    mouse.midButton.append(buttons[1])
                                    mouse.rightButton.append(buttons[2])
                                    mouse.time.append(mouse.mouseClock.getTime())
                        # *button* updates
                        
                        # if button is starting this frame...
                        if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                            # keep track of start time/frame for later
                            button.frameNStart = frameN  # exact frame index
                            button.tStart = t  # local t and not account for scr refresh
                            button.tStartRefresh = tThisFlipGlobal  # on global time
                            win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
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
                                    # run callback code when button is clicked
                                    if slider.getRating():
                                        continueRoutine = False
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
                    trials.addData('slider.response', slider.getRating())
                    # store data for trials (TrialHandler)
                    trials.addData('mouse.x', mouse.x)
                    trials.addData('mouse.y', mouse.y)
                    trials.addData('mouse.leftButton', mouse.leftButton)
                    trials.addData('mouse.midButton', mouse.midButton)
                    trials.addData('mouse.rightButton', mouse.rightButton)
                    trials.addData('mouse.time', mouse.time)
                    # Run 'End Routine' code from code_2
                    all_ratings[iteration,trial] = slider.getRating()
                    all_scr[iteration,trial] = scr_score
                    all_scr_means[iteration,trial] = scr_mean
                    scr_data[iteration, trial, :] = scr_resampled
                    all_hr_means[iteration, trial] = hr_mean
                    all_temp_means[iteration, trial] = temp_mean
                    
                    trials.addData('button.numClicks', button.numClicks)
                    if button.numClicks:
                       trials.addData('button.timesOn', button.timesOn)
                       trials.addData('button.timesOff', button.timesOff)
                    else:
                       trials.addData('button.timesOn', "")
                       trials.addData('button.timesOff', "")
                    # the Routine "rating" was not non-slip safe, so reset the non-slip timer
                    routineTimer.reset()
                    thisExp.nextEntry()
                    
                    if thisSession is not None:
                        # if running in a Session with a Liaison client, send data up to now
                        thisSession.sendExperimentData()
                # completed pop_size repeats of 'trials'
                
                
                # --- Prepare to start Routine "check_eda" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('check_eda.started', globalClock.getTime())
                key_resp_3.keys = []
                key_resp_3.rt = []
                _key_resp_3_allKeys = []
                # Run 'Begin Routine' code from repeat_code
                if not repeat:
                    repeat_loop.finished = True
                    continueRoutine = False
                elif repeat:
                    win.fullscr = False
                    win.flip()
                    failed_trials.append(f"generation {iteration}, trial {trial}")
                # keep track of which components have finished
                check_edaComponents = [check_eda_text, key_resp_3]
                for thisComponent in check_edaComponents:
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
                
                # --- Run Routine "check_eda" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *check_eda_text* updates
                    
                    # if check_eda_text is starting this frame...
                    if check_eda_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        check_eda_text.frameNStart = frameN  # exact frame index
                        check_eda_text.tStart = t  # local t and not account for scr refresh
                        check_eda_text.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(check_eda_text, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        check_eda_text.status = STARTED
                        check_eda_text.setAutoDraw(True)
                    
                    # if check_eda_text is active this frame...
                    if check_eda_text.status == STARTED:
                        # update params
                        pass
                    
                    # *key_resp_3* updates
                    waitOnFlip = False
                    
                    # if key_resp_3 is starting this frame...
                    if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        key_resp_3.frameNStart = frameN  # exact frame index
                        key_resp_3.tStart = t  # local t and not account for scr refresh
                        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_3.started')
                        # update status
                        key_resp_3.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    if key_resp_3.status == STARTED and not waitOnFlip:
                        theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                        _key_resp_3_allKeys.extend(theseKeys)
                        if len(_key_resp_3_allKeys):
                            key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                            key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                            key_resp_3.duration = _key_resp_3_allKeys[-1].duration
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
                    for thisComponent in check_edaComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "check_eda" ---
                for thisComponent in check_edaComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('check_eda.stopped', globalClock.getTime())
                # check responses
                if key_resp_3.keys in ['', [], None]:  # No response was made
                    key_resp_3.keys = None
                repeat_loop.addData('key_resp_3.keys',key_resp_3.keys)
                if key_resp_3.keys != None:  # we had a response
                    repeat_loop.addData('key_resp_3.rt', key_resp_3.rt)
                    repeat_loop.addData('key_resp_3.duration', key_resp_3.duration)
                # Run 'End Routine' code from repeat_code
                if repeat:
                    win.fullscr = True
                    win.flip()
                # the Routine "check_eda" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
            # completed 100.0 repeats of 'repeat_loop'
            
            
            # --- Prepare to start Routine "save_data" ---
            continueRoutine = True
            # update component parameters for each repeat
            # keep track of which components have finished
            save_dataComponents = [text_7]
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
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_7* updates
                
                # if text_7 is starting this frame...
                if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_7.frameNStart = frameN  # exact frame index
                    text_7.tStart = t  # local t and not account for scr refresh
                    text_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_7.status = STARTED
                    text_7.setAutoDraw(True)
                
                # if text_7 is active this frame...
                if text_7.status == STARTED:
                    # update params
                    pass
                
                # if text_7 is stopping this frame...
                if text_7.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_7.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_7.tStop = t  # not accounting for scr refresh
                        text_7.frameNStop = frameN  # exact frame index
                        # update status
                        text_7.status = FINISHED
                        text_7.setAutoDraw(False)
                
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
            # Run 'End Routine' code from code_4
            ## Print outputs to txt
            root = f"data/{participant}/{condition}"
            if not os.path.exists(root):
                os.makedirs(root)
            np.savetxt(f'{root}/embeddings.txt', all_embeddings.reshape((max_iters*pop_size, vec_size)), delimiter=',')
            np.savetxt(f'{root}/ratings.txt', all_ratings, delimiter=',')
            np.savetxt(f'{root}/scr_scores.txt', all_scr, delimiter=',')
            np.savetxt(f'{root}/scr_means.txt', all_scr_means, delimiter=',')
            np.savetxt(f'{root}/scr_data.txt', scr_data.reshape((max_iters*pop_size, eda_samples)), delimiter=',')
            np.savetxt(f'{root}/hr_means.txt', all_hr_means, delimiter=',')
            np.savetxt(f'{root}/temp_means.txt', all_temp_means, delimiter=',')
            
            # Save failed trials
            with open(f"{root}/failed_trials.txt", 'w') as f:
                for line in failed_trials:
                    f.write(f"{line}\n")
            
            
            ## Save image backups
            dest_path = f"data/{participant}/{condition}/images/generation_{iteration}"
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            shutil.copytree('images', dest_path, dirs_exist_ok=True) 
            
            # Clear EDA file
            with open("eda_data.txt", 'w') as file:
                pass
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
        # completed max_iters repeats of 'generations'
        
    # completed 1.0 repeats of 'block'
    
    
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
    while continueRoutine and routineTimer.getTime() < 5.0:
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
            if tThisFlipGlobal > outro.tStartRefresh + 5.0-frameTolerance:
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
        routineTimer.addTime(-5.000000)
    
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
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
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
