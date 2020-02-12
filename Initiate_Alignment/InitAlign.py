#TODO BeepDetect(sWavFile) Function to detect the beep time of the wav file and return it (in sec) e.g. Second 5.2
#TODO ParseTimeStampCSV(sTimeStampFile) Return 

from collections import namedtuple, defaultdict
import pandas as pd
import logging
from os.path import isfile

"""TimeStamp CSV Columns
0 - id: child id, dtype 'int'
1 - task_id: task id, dtype 'int'
2 - word_id: prompt id, dtype 'int'
3 - answer_value: rate of child answer, dtype 'int'
4 - answer_time: timestamp where RA press eval, dtype 'timestamp'
5 - task1_attempt_count: number of times prompt repeated, dtype 'int'
6 - task1_audio_cue_offset: timestamp where audio instruction ends, dtype 'timestamp'
7 - audio_cue_onset: timestamp where audio instruction starts, dtype 'timestamp'
8 - ....
"""
"""TaskTimeStamp CSV columns
0 - child_id: child id, dtype 'int'
1 - ra_id:
2 - task1_start_time, dtype 'timestamp'
3 - task1_end_time, dtype 'timestamp'
4 - task2_start_time, dtype 'timestamp'
5 - task2_end_time, dtype 'timestamp'
6 - task3_start_time, dtype 'timestamp'
7 - task3_end_time, dtype 'timestamp'
8 - task4_start_time, dtype 'timestamp'
9 - task4_end_time, dtype 'timestamp'
10 - task5_start_time, dtype 'timestamp'
11 - task5_end_time, dtype 'timestamp'

"""
lTasks = ['task1','task2','task3','task4','task5']
tTaskTimes = namedtuple('TaskTimes',lTasks)
tPrompt = namedtuple('Prompt',['taskID','wordID','word','answerTime','cueOnset','cueOffset'])
offset = 2 # seconds added to the end time of each task

def GetBeepTime():
    #TODO implement this
    return 5


def ParseTStampCSV(sTStampFile, sTaskTStampFile, iChildID, sWordIDsFile):
    
    #Check files existance
    if not isfile(sTStampFile):
        raise Exception("child {}: timestamp file not exist".format(iChildID,sTStampFile))
    if not isfile(sTaskTStampFile):
        raise Exception("child {}: task timestamp file not exist".format(iChildID,sTaskTStampFile))
    if not isfile(sWordIDsFile):
        raise Exception("child {}: word mapping file not exist".format(iChildID,sWordIDsFile))


    #Load the prompt mapping file
    pdWordIDs = pd.read_csv(sWordIDsFile,index_col=0)
    dWordIDs = pdWordIDs.to_dict()['name']

    #Load the task timestamps file
    data_task = pd.read_csv(sTaskTStampFile,parse_dates=list(range(2,12)))
    pdChild_Task = data_task[data_task.child_id == iChildID]
    if pdChild_Task.empty:
        logging.error('child {}: No data for the child in the task timestamps file {}'.format(iChildID,sTaskTStampFile))
        raise RuntimeError("Data missing in task timestamp file for child {}, check log for more info".format(iChildID))
    
    if pdChild_Task.shape[0] > 1:
        logging.warning('child {}: more than one line in the task timestamps file {}, only one line expected\nonly last line considered'.format(iChildID,sTaskTStampFile))

    child_task_tstamps = pdChild_Task.iloc[-1]
    if pd.isnull(child_task_tstamps.task1_start_time):
        logging.error('child {}: No time stamp for the start of task 1 in file {}, Reference time can\'t set'.format(iChildID,sTaskTStampFile))

        raise RuntimeError("Error in task timestamp file for child {}".format(iChildID))

    RefTime = child_task_tstamps.task1_start_time.timestamp()
    
    #Load prompt timestamps file
    data = pd.read_csv(sTStampFile,parse_dates=[4,6,7])
    pdChild = data[data.id==iChildID]
    
    if pdChild.empty:
        logging.error('child {}: No data for the child in the prompt timestamps file {}'.format(iChildID, sTStampFile))
        raise RuntimeError("Data missing in task timestamp file for child {}, check log for more info".format(iChildID))

    dTaskPrompts = defaultdict(list)
    lTaskTimes = []

    for i,sTaskID in enumerate(lTasks):
        iTaskID = i+1 
        fTaskST,fTaskET = child_task_tstamps[i+2:i+4] #First two columns for the child_id and ra_id

        if pd.isnull(fTaskST) or pd.isnull(fTaskET):
            logging.error('child {}: No start or end timestamp for task {} in file {}, task will be skipped'.format(iChildID,sTaskTStampFile))
            lTaskTimes.append((-1,-1))
            continue
        lTaskTimes.append((fTaskST.timestamp() - RefTime,fTaskET.timestamp() - RefTime))

        pdTask = pdChild[pdChild.task_id==iTaskID] ##CHANGE if COL CHANGED
        
        if pdTask.empty:
            logging.warning('child {}: No data of task {} in the prompt timestamps file {}, task will be skipped'.format(iChildID, sTaskID, sTStampFile))
            continue
        
        for r in pdTask.iterrows():
            #TODO handle any nonexist field
            data = r[1]
            iWordID = data.word_id #TODO handle if ID not exist
            if pd.isnull(iWordID) or iWordID not in dWordIDs:
                logging.warning('child {}: word id {} of task {} either null or not exist in word-mappingfile word set to NULL'.format(iChildID, str(iWordID), sTaskID))
                sWord = 'NULL'
            else:
                sWord = dWordIDs[iWordID]
            
            answerTime = data.answer_time
            if pd.isnull(answerTime):
                logging.warning('child {}: answer timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                answerTime = -1
            else:
                answerTime = answerTime.timestamp() - RefTime
       
            cueOnset = data.audio_cue_onset
            if pd.isnull(cueOnset):
                logging.warning('child {}: cueOnset timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                cueOnset = -1
            else:
                cueOnset = cueOnset.timestamp() - RefTime

            cueOffset = data.task1_audio_cue_offset
            if pd.isnull(cueOffset):
                logging.warning('child {}: cueOffset timestamp is null in word {} task {}'.format(iChildID, str(iWordID), sTaskID))
                cueOffset = -1
            else:
                cueOffset = cueOffset.timestamp() - RefTime

            prompt = tPrompt(iTaskID, iWordID, sWord, answerTime, cueOnset, cueOffset)
            
            dTaskPrompts[iTaskID].append(prompt)

    tTasks = tTaskTimes(*lTaskTimes)
    return tTasks, dTaskPrompts


def Segmentor(sWavFile, sTimeStampCSV, sTaskTStampCSV, iChildID, sWordIDsFile, sOutDir):
    #Load Wav File (session)
    if not isfile(sWavFile):
        raise Exception("child {}: session speech File {} not exist".format(iChildID,sWavFile))
    
    tTask, tPrompts = ParseTStampCSV(sTimeStampCSV, sTaskTStampCSV, iChildID, sWordIDsFile)




    #Detect the start and end of all beep signal(s)
    #ParseCSV file get tTasks, dTaskPrompts
    #TODO fill interval gaps with empty, this should be done in the writetextgrid function
    """
    _wav_param, RWav = txtgrd.ReadWavFile('Recordings/24_jan_2020/Data/CH001/CH001_1_001
     ...: .wav')
     fRefPos = IA.GetBeepTime()
     fStTime = tTasks.task1[0] + fRefPos
     fEtTime = tTasks.task1[1] + fRefPos
     iFSt = int(fStTime*_wav_param.framerate*_wav_param.sampwidth)
     iFEt = int(fEtTime*_wav_param.framerate*_wav_param.sampwidth)
     nFrams = int((iFEt-iFSt)/_wav_param.sampwidth)
     txtgrd.WriteWaveSegment(RWav[iFSt:iFEt],_wav_param,nFrams,'task1.wav')
     dTiers = defaultdict(lambda: [[],[],[]])
     lSt = [t.cueOffset for t in tPrompts[1]]
     lEt = [t.answerTime for t in tPrompts[1]]
     lLabel = [t.word for t in tPrompts[1]]
     dTiers['Prompt']=[lSt,lEt,lLabel]
     txtgrd.WriteTxtGrdFromDict('task1.txtgrid',dTiers,0.0,lEt[-1])

    """

