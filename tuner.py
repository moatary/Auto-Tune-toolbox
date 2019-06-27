


class arg(): # An empty class used as struct to get filled later.
    pass


class tuner(): # fine-tune the parameters and methods
    def __init__(self,*tunes):
        names=[]
        # get names+  for each tune, index name of key:
        tunes_index=[[] for _ in tunes]
        for ii,tune in enumerate(tunes):
            keys=list(tune.keys())
            for jj,key in enumerate(keys):
                if key not in names:
                    names.append(key)
                tunes_index[ii].append(names.index(key))
        #
        tuneslist=[[[]] for _ in tunes]
        keyslist=[[[]]  for _ in tunes]
        newtuneslist=[[] for _ in tunes]
        newkeyslist=[[]  for _ in tunes]
        currentcnt=1
        # get list:
        for i,(tune,tuneindex) in enumerate(zip(tunes,tunes_index)):
            remaineddict=tune.copy()
            for indofnameind, nameind in enumerate( tuneindex):  # key,values in zip(remaineddict.keys(), remaineddict.values()):
                key = names[nameind]
                values = tune[key]
                for value in values:
                    for keyy,itm in zip(keyslist[i],tuneslist[i]):
                        newtuneslist[i].append([*itm, value ])
                        newkeyslist[i].append([*keyy, key ])
                tuneslist[i]= newtuneslist[i].copy()
                newtuneslist[i]=[]
                keyslist[i]= newkeyslist[i].copy()
                newkeyslist[i]=[]

        # concat results :
        tuness = []
        for itm in tuneslist:
            tuness.extend(itm)
        keyss=[]
        for itm in keyslist:
            keyss.extend(itm)

        # return list of dicts:
        self.listoftunes=[dict(zip(itmkeys,itmvalues)) for itmkeys,itmvalues in zip(keyss,tuness)]
        self.currentindex=-1



    def __iter__(self): # iterate over all possible combinations of truning parameters' states
        for self.currentindex in range(self.currentindex,len(self.listoftunes)):
            self.tune= self.listoftunes[self.currentindex]
            yield self.applytune()


    def applytune(self): # turn string names of parameters to real variable.
        args = arg()
        for itmname,itmvalue in zip(self.tune.keys(),self.tune.values()):
            exec('args.'+itmname+'='+itmvalue)
        return args




