#TODO map ipa to sampa
#TODO merge textgrid files
import argparse
from collections import defaultdict
import wave
from os.path import basename, splitext, join, isfile
import bisect
import re
from difflib import SequenceMatcher
import numpy as np
from numba import jit
import logging


OFFSET = 0.01 #sec
DEVTH = 0.02

lTextRm = ('?','`',',','!','.')

@jit 
def is_sorted(a): 
    for i in range(a.size-1): 
        if a[i+1] < a[i] : 
           return False 
    return True 


def ReadWavFile(sWavFileName):
    if not isfile(sWavFileName):
        raise Exception(" Wave file {} not exist".format(sWavFileName))
    with wave.open(sWavFileName,'rb') as fWav:
        _wav_params = fWav.getparams()
        data = fWav.readframes(_wav_params.nframes)
    return _wav_params, data

def WriteWaveSegment(lCrnt_data, _wav_params, nFrams, sCrnt_wav_name):
    lWav_params = list(_wav_params)
    lWav_params[3] = nFrams
    with wave.open(sCrnt_wav_name,'wb') as fCrnt_wav:
        fCrnt_wav.setparams(tuple(lWav_params))
        fCrnt_wav.writeframes(lCrnt_data)
    return

def ParseTextTxtGrid(lLines):
    pItem = re.compile('(?<=item \[)([0-9]*)(?=\]:)')
    pTierSize = re.compile('(?<=intervals: size = )([0-9]*)')
    pTierName = re.compile('(?<=name = ")(.*)(?=")')
    pST = re.compile('(?<=xmin = )([0-9\.]*)')
    pET = re.compile('(?<=xmax = )([0-9\.]*)')
    pLabel = re.compile('(?<=text = ")(.*)(?=")')

    dTiers = defaultdict(lambda: [[],[],[]])
    lTiers = []
    sCurLine = ''
    while 'tiers?' not in sCurLine and lLines:
        sCurLine = lLines.pop()
    if not lLines or 'exists' not in sCurLine:
        print('Bad format or no tiers')
        return
    nTiers = int(lLines.pop().split()[-1])
    for j in range(nTiers):
        _match = None
        while not _match:
            try:
                sCurLine = lLines.pop()
            except IndexError:
                print('Bad format')
                return
            _match = pItem.findall(sCurLine)
        iItemIndx = _match[0]
        _match = None
        while not _match:
            try:
                sCurLine = lLines.pop()
            except IndexError:
                print('Bad format')
                return
            _match = pTierName.findall(sCurLine)
        sCTierName = _match[0]
        _match = None
        while not _match:
            try:
                sCurLine = lLines.pop()
            except IndexError:
                print('Bad format')
                return
            _match = pTierSize.findall(sCurLine)
        nIntervals = int(_match[0])
        for i in range(nIntervals):
            sCurLine = lLines.pop()
            fST = float(pST.findall(lLines.pop())[0])
            fET = float(pET.findall(lLines.pop())[0])
            sLabel = pLabel.findall(lLines.pop())[0]
            dTiers[sCTierName][0].append(fST)
            dTiers[sCTierName][1].append(fET)
            dTiers[sCTierName][2].append(sLabel)
    return dTiers

def CompareTxtGrids(sTxtGrd1, sTxtGrd2, sTier1, sTier2, sMapFile, fDevThr = DEVTH):
    #load TxtGrid 1
    dTiers1 = ParseTxtGrd(sTxtGrd1)
    dTiers2 = ParseTxtGrd(sTxtGrd2)
    lST1, lET1, lLabels1 = dTiers1[sTier1]
    lST2, lET2, lLabels2 = dTiers2[sTier2]
    #Load phoneme map
    #print(lLabels1, lLabels2)
    if sMapFile:
        with open(sMapFile) as fMap:
            dMap = dict([l.strip().split() for l in fMap])
        for lLabels in lLabels1, lLabels2:
            for i in range(len(lLabels)):
                sLabel = lLabels[i].strip()
                if sLabel in dMap:
                    lLabels[i] = dMap[sLabel]
                elif sLabel:
                    print('{} not in MapFile'.format(sLabel))
    lJunks = ['sil','<p:>','','SIL']
    seq = SequenceMatcher(lambda x: x in lJunks,lLabels1,lLabels2)
    nPh1 = len([p for p in lLabels1 if p not in lJunks])
    nPh2 = len([p for p in lLabels2 if p not in lJunks])
    nMatchedPh = 0
    nDeviatedPh = 0
    lDevPhs = []
    lMatchedPhs = []
    #print(lLabels1,lLabels2)
    for a, b, size in seq.get_matching_blocks():
        #nMatchedPh += size
        for i in range(size):
            if lLabels1[a+i] not in lJunks:
                nMatchedPh += 1
                if abs(lST1[a+i] - lST2[b+i]) + abs(lET1[a+i] - lET2[b+i]) > fDevThr:
                    nDeviatedPh += 1
                    lDevPhs.append(lLabels1[a+i])
                lMatchedPhs.append(lLabels1[a+i])

    return (nPh1, nPh2, nMatchedPh, nDeviatedPh, lDevPhs, lMatchedPhs, seq.ratio())


#lTxtGrids --> is a list of textgrid file names ('txtGrid1','txtGrid2',...)
#sOutputFile --> name of merged text grid file
#sWavFile --> the path to the wav file of the textgrids, used to get the end time if not specified
#aSlctdTiers --> list of lists, each list contains the requested tiers in each textgrid file in lTxtGrids [[tier1,tier2],[tier1,tier2,],...] Note: number of lists should be same as number of TxtGrid files in lTxtGrids, if empty all tiers in all textgrids will be merged
#aMapper: list of tubles each with name of tier and file to map labels to other symboles [('tier1','mapper file')]

def MergeTxtGrids(lTxtGrids, sOutputFile, sWavFile='', aSlctdTiers=[], aMapper = [], fST = None, fET = None):
    fST = 0.0 if not fST else fST
    if not fET:
        if not sWavFile:
            print('Specify either Ent time or path to wave file')
            return
        _wav_params, data = ReadWavFile(sWavFile)
        fET = _wav_params.nframes/_wav_params.framerate
    if not aSlctdTiers:
        aSlctdTiers = [[] for i in lTxtGrids]

    dTierMapper = dict(aMapper) if aMapper else None

    assert len(aSlctdTiers) == len(lTxtGrids)

    dMergTiers = defaultdict(lambda: [[],[],[]])

    i = 0

    for sTxtGrd, lSlctdTiers in zip(lTxtGrids,aSlctdTiers):
        dTiers = ParseTxtGrd(sTxtGrd)
        lSlctdTiers = dTiers.keys() if not lSlctdTiers else lSlctdTiers
        if dTierMapper:
            for sTier in lSlctdTiers:
                if sTier in dTierMapper:
                    with open(dTierMapper[sTier]) as fMap:
                        dMap = dict([l.strip().split() for l in fMap])
                    dTiers[sTier][2] = list(map(lambda x: dMap[x] if x in dMap else x,dTiers[sTier][2]))

        dMergTiers.update(dict([('{}-{}'.format(i,k),v) for k,v in dTiers.items() if k in lSlctdTiers]))

    WriteTxtGrdFromDict(sOutputFile, dMergTiers, fST, fET)
    return

    
def ParseChronTxtGrd(lLines):
    dTiers = defaultdict(lambda: [[],[],[]])
    lTiers = []
    fST, sET = map(float,lLines.pop().split()[:2])
    nTiers = int(lLines.pop().split()[0])
    for i in range(nTiers):
        sTierName = lLines.pop().split()[1].strip('"')
        lTiers.append(sTierName)
    while lLines:
        sLine = lLines.pop()
        if sLine and sLine[0] == '!':
            lLine = lLines.pop().split()
            sCTierName = lTiers[int(lLine[0])-1]
            fST, fET = map(float,lLine[1:])
            sLabel = lLines.pop().strip('"')
            dTiers[sCTierName][0].append(fST)
            dTiers[sCTierName][1].append(fET)
            dTiers[sCTierName][2].append(sLabel)
    return(dTiers)

def ParseTxtGrd(sTxtGrd):
    with open(sTxtGrd) as fTxtGrd:
        lLines = fTxtGrd.read().splitlines()
    lLines.reverse()
    sCurLine = lLines.pop()
    if 'chronological' in sCurLine:
        dTiers = ParseChronTxtGrd(lLines)
    elif 'ooTextFile' in sCurLine:
        dTiers = ParseTextTxtGrid(lLines)
    else:
        print('TxtGrd format error or not supported')
        return
    return dTiers


def ValidateTextGridDict(dTiers, lSlctdTiers=[]):
    #Check if lSlctdTiers is in dTiers
    if lSlctdTiers:
        lNotIn = [i for i in lSlctdTiers if i not in dTiers.keys()]
        assert not lNotIn, '{} tiers not in dict'.format(' '.join(lNotIn))
    else:
        lSlctdTiers = dTiers.keys()
    for sTier in lSlctdTiers:
        lSTs, lETs, lLabls = dTiers[sTier]
        
        assert len(lSTs) == len(lETs), "Number of start times not equal number of end times in tier {}".format(sTier)
        
        assert len(lSTs) == len(lLabls), "Number of labels not equal to number of intervals in tier {}".format(sTier)

        aSTs = np.asarray(lSTs)
        aETs = np.asarray(lETs)

        assert (aETs > aSTs).all(), "End time is greater than strat time in one or more intervals in tier {}".format(sTier)

        assert is_sorted(aETs) and is_sorted(aSTs), "Either End times or Start Times of tier {} not in order".format(sTier)

    return

def FillGapsInTxtGridDict(dTiers,sFilGab = "", lSlctdTiers=[]):
    """
    Should be called always after ValidateTextGridDict to make sure that the dTiers is valid
    """
    dTiers_f = defaultdict(lambda: [[],[],[]])
    lSlctdTiers = lSlctdTiers if lSlctdTiers else dTiers.keys()
    for sTier in lSlctdTiers:
        lSTs, lETs, lLabls = dTiers[sTier]
        aSTs = np.asarray(lSTs)
        aETs = np.asarray(lETs)
        aLabels = np.asarray(lLabls,dtype=str)
        pos = np.where(aSTs[1:]!=aETs[:-1])
        lSTs_f = list(np.insert(aSTs,pos[0]+1,aETs[pos[0]]))
        lETs_f = list(np.insert(aETs,pos[0]+1,aSTs[pos[0]+1]))
        lLabel_f = list(np.insert(aLabels,pos[0]+1,sFilGab))
        dTiers_f[sTier] = [lSTs_f,lETs_f,lLabel_f]
    return dTiers_f

def WriteTxtGrdFromDict(sFName, dTiers, fST, fET, bReset=True, lSlctdTiers=[], sFilGab = None):
    ValidateTextGridDict(dTiers,lSlctdTiers)
    if sFilGab != None:
        dTiers = FillGapsInTxtGridDict(dTiers,sFilGab,lSlctdTiers)
 
    fST = round(fST,4)
    fET = round(fET,4)
    if bReset:
        fNewST = 0
        fNewET = fET - fST
    else:
        fNewST = fST
        fNewET = fET
    lAllIntervals = []
    lSlctdTiers = [i for i in lSlctdTiers if i in dTiers.keys()]
    if not lSlctdTiers:
        lSlctdTiers = list(dTiers.keys())
    for sTier in lSlctdTiers:
        lSTs, lETs, lLabls = dTiers[sTier]
        lSTs = [round(i,4) for i in lSTs]
        lETs = [round(i,4) for i in lETs]
        iSindx = bisect.bisect_left(lSTs, fST)
        iEindx = bisect.bisect(lETs, fET)
        #print(iSindx,iEindx)
        #Adjust start and end times
        #print(lSTs,lETs)
        lNewSTs = lSTs[iSindx:iEindx]
        lNewETs = lETs[iSindx:iEindx]
        lNewLabls = lLabls[iSindx:iEindx]
        #print(lNewSTs,lNewETs)
        #TODO consider not to make same start and end time for all
        #Add sil interval at the start and end
        #lNewSTs = [fST] + lNewSTs + [lNewETs[-1]]
        #lNewETs = [lNewSTs[0]] + lNewETs + [fET]

        #Reset to 0
        if bReset and fST != 0.0:
            lNewSTs = [i-fST for i in lNewSTs]
            lNewETs = [i-fST for i in lNewETs]
        try:
            if fNewST < lNewSTs[0]:
                lAllIntervals.append((fNewST, lNewSTs[0], 'sil' ,sTier))
        except:
            print('Error:',lNewSTs,lNewETs, iSindx, iEindx)
            return
        for fiST, fiET, siLabl in zip(lNewSTs,lNewETs,lNewLabls):
            lAllIntervals.append((fiST, fiET, siLabl,sTier))
        if fNewET > lNewETs[-1]:
            #print(round(fNewET,3),round(lNewETs[-1],3),sTier)
            lAllIntervals.append((lNewETs[-1], fNewET, 'sil' ,sTier))

    lAllIntervals = sorted(lAllIntervals)
    #Writing the chron txtgrid
    with open(sFName,'w') as fTxtGrd:
        print('"Praat chronological TextGrid text file"', file=fTxtGrd)
        print('{} {}   ! Time domain.'.format(fNewST, fNewET),file=fTxtGrd)
        print('{}   ! Number of tiers.'.format(len(lSlctdTiers)), file=fTxtGrd)
        for sTier in lSlctdTiers:
            print('"IntervalTier" "{}" {} {}'.format(sTier,fNewST,fNewET), file=fTxtGrd)
        print('', file=fTxtGrd)
        for fiST, fiET, siLabl, sTier in lAllIntervals:
            indx = lSlctdTiers.index(sTier) + 1
            print('! :{}'.format(sTier), file=fTxtGrd)
            print('{} {} {}'.format(indx, fiST, fiET), file=fTxtGrd)
            print('"{}"'.format(siLabl), file=fTxtGrd)
            print('', file=fTxtGrd)
    return        

def TextNormalize(sTxt):
    pNorm = re.compile('\(.*?\)|'+'['+re.escape(''.join(lTextRm))+']')
    return pNorm.sub('',sTxt)

def Process(sTxtGrd, sWavFile, sSplitBy, sOutputDir, bTxtGrd = False, bNorm = True):
    dTiers = ParseTxtGrd(sTxtGrd)
    _wav_params, data = ReadWavFile(sWavFile)
    sBaseName = splitext(basename(sWavFile))[0]
    for fST, fET, sLabel in zip(*dTiers[sSplitBy]):
        if sLabel:
            print('File: {} - time {} to {} - label {}'.format(sTxtGrd,fST, fET, sLabel))
            indxFS = int((fST - OFFSET) * _wav_params.framerate) #start one frame before
            indxFE = int((fET + OFFSET) * _wav_params.framerate) #end one frame after
            nFrams = indxFE - indxFS
            lCrnt_data = data[indxFS * _wav_params.sampwidth:indxFE * _wav_params.sampwidth]
            sCrnt_name = "{0}_{1:.2f}_{2:.2f}".format(sBaseName,fST,fET)
            cCrnt_wav_name = join(sOutputDir,"{0}.wav".format(sCrnt_name))
            sCrnt_txt_name = join(sOutputDir,"{0}.txt".format(sCrnt_name))
            WriteWaveSegment(lCrnt_data, _wav_params, nFrams, cCrnt_wav_name)
            with open(sCrnt_txt_name,'w') as fCrnt_txt:
                if bNorm:
                    sLabel = TextNormalize(sLabel)
                print(sLabel,file=fCrnt_txt)
            if bTxtGrd:
                sCrnt_txtGrd_name = join(sOutputDir,"{0}.textgrid".format(sCrnt_name))
                fST = indxFS/_wav_params.framerate #Recalculate the time after OFFSET
                fET = indxFE/_wav_params.framerate
                WriteTxtGrdFromDict(sCrnt_txtGrd_name, dTiers, fST, fET, bReset=True, lSlctdTiers=[])
    return

def ArgParser():
    parser = argparse.ArgumentParser(description='This code split wav based on textgrid alignment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('TxtGrd',  help='The path to the TextGrid file', type=str)
    parser.add_argument('WavFile',  help='The path to the associated wav file', type=str)
    parser.add_argument('SplitBy',  help='The tier name to split the wav file based on', type=str)
    parser.add_argument('OutputDir',  help='The path to the output dir', type=str)
    parser.add_argument('-t','--txtgrd', help='Use this option to output TextGrid file for each segment', dest='txtgrd_o', action='store_true',default=False)
    parser.add_argument('-s','--spontaneous', help='Use this option to enable processing of spontaneous part of the data', dest='process_spontaneous_data', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = ArgParser()
    sTxtGrd, sWavFile, sSplitBy, sOutputDir = args.TxtGrd, args.WavFile, args.SplitBy, args.OutputDir
    bTxtGrd = args.txtgrd_o
    Process(sTxtGrd, sWavFile, sSplitBy, sOutputDir, bTxtGrd = bTxtGrd)




if __name__=='__main__':
    main()
