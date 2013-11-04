#############################################################################
#Copyright (c) 2010, Jo Bovy, David W. Hogg, Dustin Lang
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without 
#modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, 
#      this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice, 
#      this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#   The name of the author may not be used to endorse or promote products 
#      derived from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.
#############################################################################
import re
import scipy as sc
import scipy.stats as stats
import scipy.linalg as linalg
import math as m
import numpy as nu
#from sample_wishart import sample_wishart
#from sample_normal import sample_normal

def read_data(datafilename='data_allerr_backup.dat',allerr=True):
    """read_data_yerr: Read the data from the file into a python structure
    Reads {x_i,y_i,sigma_yi}

    Input:
       datafilename    - the name of the file holding the data
       allerr          - If set to True, read all of the errors

    Output:
       Returns a list {i,datapoint, y_err}, or {i,datapoint,y_err, x_err, corr}

    History:
       2009-05-20 - Started - Bovy (NYU)
    """
    if allerr:
        ncol= 7
    else:
        ncol= 4
    #Open data file
    datafile= open(datafilename,'r')
    #catch-all re that reads numbers
    expr= re.compile(r"-?[0-9]+(\.[0-9]*)?(E\+?-?[0-9]+)?")
    rawdata=[]
    nline= 0
    for line in datafile:
        if line[0] == '#':#Comments
            continue
        nline+= 1
        values= expr.finditer(line)
        nvalue= 0
        for i in values:
            rawdata.append(float(i.group()))
            nvalue+= 1
        if nvalue != ncol:
            print "Warning, number of columns for this record does not match the expected number"
    #Now process the raw data
    out=[]
    for ii in range(nline):
        #First column is the data number
        thissample= []
        thissample.append(rawdata[ii*ncol])
        sample= sc.array([rawdata[ii*ncol+1],rawdata[ii*ncol+2]])
        thissample.append(sample)
        thissample.append(rawdata[ii*ncol+3])
        if allerr:
            thissample.append(rawdata[ii*ncol+4])
            thissample.append(rawdata[ii*ncol+5])
        out.append(thissample)
    return out
