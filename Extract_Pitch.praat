##################################
#select all
#Remove
##################################
#mainDirectory$ = "/media/mostafa/TamuqCAS/root/Lexical_Stress_Detection/2-Feature_Extraction/somewav/"
form Extr Features
    text ListF /media/mostafa/TamuqCAS/root/Lexical_Stress_Detection/2-Feature_Extraction/aa
endform
print 'listF$'
#Create Strings as file list... list 'mainDirectory$'/*.wav
Read Strings from raw text file... 'listF$'
lstOp$ = selected$ ("Strings")
nOF = Get number of strings
print 'nOF' 'newline$'
for i from 1 to nOF
	select Strings 'lstOp$'
	filename$ = Get string... 'i'
	print 'filename$''newline$'
	Read from file... 'filename$'
	ns = Get number of samples
	sr = Get sampling frequency
	tms = (ns/sr)*1000
	nfram = tms/10
	print "ns: " 'ns' "sr: " 'sr' "tms: " 'tms' "nfram: " 'nfram' 'newline$'
	current$ = selected$ ("Sound")
	fnameP$ = replace$(filename$,".wav",".ac",0)
	To Pitch... 0.0 75.0 600.0
	#To Matrix...
	for fram from 0 to nfram-1
	    p = Get value in frame... 'fram' Hertz
	    if p = undefined
	        p = 0
	    endif
	    fileappend 'fnameP$' 'p' 'newline$'
	endfor
        Remove
	select Sound 'current$'
        Remove
endfor
