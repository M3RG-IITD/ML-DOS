
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

class Lammps_Traj():
    def __init__(self,filepath :str,Timestep_end=None):

        self.Filepath=filepath
        file=open(self.Filepath,'r+')
        prev_line=[]
        Frame=dict()
        self.Traj=[]
        k=0
        flag=False
        lines=[a for a in file]
        temp=0
        while(temp<len(lines)):
            line=lines[temp]
            if("TIMESTEP" in prev_line):
                Frame.update(TIMESTEP=int(line))
            elif("NUMBER OF ATOMS" in prev_line):
                Frame.update(N=int(line))
            elif("ATOMS" in prev_line):
                words=self.break_line_to_words(prev_line,' ')
                myheader=words[2:]
                #print(Frame.get('TIMESTEP'))
                data=pd.read_table(self.Filepath,sep='\s+',skiprows=temp,nrows=Frame.get('N'),header=None,names=myheader)
                Frame.update(Data=data)
                self.Traj+=[dict(Frame)]
                temp+=Frame.get('N')-1
                Frame.clear()
            temp+=1
            prev_line=line


    def break_line_to_words(self,line,delim :str =' '):
        """To give list of words in a line with the delimiter as given"""
        res_wordlist=[]
        n=len(line)
        i=0
        word=""
        while(i<n):
            if(line[i]==' ' or line[i]=="\t" or i==n-1):
                if(word!=""):
                    res_wordlist+=[word]
                    word=""
            else:
                word+=line[i]
            i+=1
        return res_wordlist

    def getTraj(self):
        return self.Traj