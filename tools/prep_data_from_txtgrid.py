import txtgrid_master.TextGrid_Master as tgm
from os.path import basename,join
from os import makedirs
import pandas as pd
import mysql.connector
import logging
import configparser, argparse
from collections import defaultdict

#TODO add logging and check if files exists

def get_args():
    parser = argparse.ArgumentParser(description='Create kaldi data dir from txtgrid')
    parser.add_argument("sWaveFile", type=str, help='The path to the speech file')

    parser.add_argument("sTxtgridFile", type=str, help='The path to the txtgrid file')

    parser.add_argument("sOutDir", type=str, help='Destination to save kadi data files')

    parser.add_argument("-a", "--append", dest='isAppend', action='store_true', default=False, help='Append to existing files, if not called replace them')

    parser.add_argument("-sid", "--spkr_ID", default='0001', help='Speaker ID')

    parser.add_argument("-rid", "--rcrd_ID", default='0001', help='Record ID')

    parser.add_argument("-p", "--prmpt_tier", default='Prompt', help='Txtgrid tier contains prompts')
    
    return(parser.parse_args())

def Generate_strings(sWaveFile, sTxtgridFile, sSpkrID='0001', sRcrdID='0001', sPromptTier='Prompt'):
    strWavScp = strSegment = strUtt2Spk = strSpk2Utt = strText = ''
    strSpk2Utt += '{0} '.format(sSpkrID)
    recordID = sRcrdID
    strWavScp += '{0} sox {1} -t wav -b 16 -r 16000 -c 1 - |\n'.format(recordID,sWaveFile)
    lST, lET, lLabels = tgm.ParseTxtGrd(sTxtgridFile)[sPromptTier]
    data = pd.DataFrame.from_dict({'start_time':lST,'end_time':lET,'label':lLabels})
    data_valid = data[(data.label != '') & (data.label != 'sil')]
    for indx,r in data_valid.iterrows():
        prompt = r.label
        promptID = indx
        fST = r.start_time
        fET = r.end_time
        utterID = '{0}_{1}'.format(recordID, promptID) 
        strSegment += '{0} {1} {2} {3}\n'.format(utterID,recordID,fST,fET)
        strUtt2Spk += '{0} {1}\n'.format(utterID,sSpkrID)
        strSpk2Utt += '{0} '.format(utterID)
        strText += '{0} {1}\n'.format(utterID,prompt)
    strSpk2Utt += '\n'

    return (('wav.scp',strWavScp),('segments',strSegment),('text',strText),('utt2spk',strUtt2Spk),('spk2utt',strSpk2Utt))

def Write_files(tPairs, sOutDir, isAppend=False):
    makedirs(sOutDir,exist_ok=True)
    mode='a' if isAppend else 'w'
    for fName, strLines in tPairs[:-1]: #Write all except spk2utt
        with open(join(sOutDir,fName),mode=mode) as oFile:
            oFile.write(strLines)
    #Write spk2utt
    fName, strLines = tPairs[-1]
    if not isAppend:
        with open(join(sOutDir,fName),mode=mode) as oFile:
            oFile.write(strLines)
    else:
        #load current file
        spk2utt_dict = defaultdict(list)
        with open(join(sOutDir,fName),'r') as iFile:
            lLines = iFile.read().splitlines()
            for line in lLines+strLines.splitlines():
                sLine = line.split()
                spk2utt_dict[sLine[0]].extend(sLine[1:])
        with open(join(sOutDir,fName),mode='w') as oFile:
            for spkID in spk2utt_dict:
                print('{0} {1}'.format(spkID,' '.join(spk2utt_dict[spkID])),file=oFile)

def main():
    args = get_args()

    tFileStrings = Generate_strings(args.sWaveFile, args.sTxtgridFile, args.spkr_ID, args.rcrd_ID, args.prmpt_tier)
    Write_files(tFileStrings, args.sOutDir, args.isAppend)


if __name__ == '__main__':
    main()


