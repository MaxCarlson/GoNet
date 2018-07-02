import os
import random
import numpy as np
import tensorflow as tf
from Globals import BoardSize, BoardLength, BoardDepth, BLACK, WHITE
import multiprocessing, threading, queue, asyncio

# Handles the loading and processing of individual files
# into data that is partially readable (not sepperated into batches)
# for the model. Also keeps and maintains a queue of 
# proccessed data for reading without wait
#
# Args:
# fileShape: The range of files to read data from; 
# useful for sepperating training/validation data
# feature/labelPath: path to features/labels containing folder
# maxQSize: Maximum number of files contents to hold in memory
#
# TODO: Make more generic and move specific model code outside to another class
class FileLoader():
    def __init__(self, fileShape, featurePath, labelPath, maxQSize):
        self.fileShape = fileShape
        self.featurePath = featurePath
        self.labelPath = labelPath
        self.featureList = self.shapeFileList(os.listdir(featurePath))
        self.labelList = self.shapeFileList(os.listdir(labelPath))
        # Indices in file lists which we'll shuffle to randomize files
        self.idx = 0
        self.maxQSize = maxQSize
        self.indices = np.arange(0, len(self.featureList))
        self.queue = queue.Queue(maxsize=maxQSize)
        thread = threading.Thread(target=self.fillQueue)
        thread.start()

    # Use only the # of files we're told to
    def shapeFileList(self, fileList):
        return fileList[self.fileShape[0]:self.fileShape[1]]

    def loadFile(self, i):
        sampleF = self.featureList[self.indices[self.idx]]
        sampleL = self.labelList[self.indices[self.idx]]      
        wholePathF = self.featurePath + '/' + sampleF
        wholePathL = self.labelPath + '/' + sampleL

        return self.extractNpy(wholePathF, wholePathL)

    # Set the first layer of the feature maps 
    # to the side-to-move color code (all 0's for black, 1's for white)
    def setColorLayer(self, X, C, m):
        for i in range(0, m):
            X[i, 0] = C[i] - 1

    # Read the data from the file created in CurateData
    # Format it into the binary feature map explained in CurateData
    def extractNpy(self, xFile, yFile):
        XComp = np.load(xFile)
        YCol  = np.load(yFile)
        m = np.shape(YCol)[0]
        Y = YCol[0:m, 0]
        C = YCol[0:m, 1]

        # TODO: Figure out how to use scipy compressed catagorical here
        #Y = scipy.sparse.csr_matrix((np.ones(minibatchSize, np.float32), (range(minibatchSize), Y)), shape=(minibatchSize, BoardSize))
        #Y = scipy.sparse.csc_matrix(Y, shape=(minibatchSize, BoardSize))    
        Y = tf.keras.utils.to_categorical(Y, BoardSize)
        X = np.zeros((m, BoardDepth, BoardLength, BoardLength))
        self.setColorLayer(X, C, m)
        
        # Build the binary feature maps for all recorded board states
        # for each example
        for i in range(0, m):
            # Add both layers representing player and opponent stones
            color = C[i]
            opponent = WHITE if color == BLACK else BLACK
            index = 0
            for j in range(1, BoardDepth, 2):
                X[i, j]   = XComp[i, index] == color
                X[i, j+1] = XComp[i, index] == opponent
                index += 1

        return X, Y

    def fillQueue(self):
        
        lock = threading.Lock()
        lock.acquire()
        while self.queue.full() == False:
            if self.idx >= np.shape(self.indices)[0]:
                random.shuffle(self.indices)
                self.idx = 0

            self.queue.put((self.loadFile(self.idx)))
            self.idx += 1

        lock.release()
    # When a new file is needed retrieve it
    # Also call the thread 
    def nextFile(self):
        XX, YY = self.queue.get()
        self.queue.task_done()

        # Only start a thread if we're running low
        qCount = self.queue.qsize()
        if qCount == 0 or qCount <= self.maxQSize // 2:
            thread = threading.Thread(target=self.fillQueue)
            thread.start()

        return XX, YY
    
# Generates batches of data for the model
# Uses FileLoader's multiple threads to keep processing/reading data while
# model is running
#
# Args:
# feature/labelPath: is the path to the feature/label files 
# batchSize: Number of items processed in each batch. Also used to calculate stepsPerEpoch and samplesEst
# loadSize: Number of files to hold queued for model
#
# TODO: Make more generic and move specific model code outside to another class
class Generator():
    def __init__(self, featurePath, labelPath, fileShape, batchSize, loadSize = 3):
        self.featurePath = featurePath
        self.labelPath = labelPath
        self.batchSize = batchSize
        self.fileShape = fileShape
        self.samplesEst = 0
        self.stepsPerEpoch = 0
        self.loader = FileLoader(self.fileShape, self.featurePath, self.labelPath, loadSize)
        self.calcStepsPerEpoch()

    # Grab the next chunk of the file
    def getNextChunk(self, XX, YY, m, roll):
            
            X = np.zeros((self.batchSize, BoardDepth, BoardLength, BoardLength)) 
            Y = np.zeros((self.batchSize, BoardSize))
            
            if roll + self.batchSize < m:
                X = XX[roll:roll + self.batchSize]
                Y = YY[roll:roll + self.batchSize]

            # This shouldn't happen all that often,
            # Should curate data to avoid it infact!
            elif self.batchSize > m:
                for b in range(self.batchSize):
                    roll = random.randint(0, np.shape(XX)[0]-1)        
                    X[b] = XX[roll]
                    Y[b] = YY[roll]

            else:
                # If it's not a perfect fit, take the upper slice then the lower one
                m1 = np.shape(XX[roll:m])[0]
                m2 = self.batchSize - m1
                X[0:m1] = XX[roll:m]
                Y[0:m1] = YY[roll:m]
                X[m1:self.batchSize] = XX[0:m2]
                Y[m1:self.batchSize] = YY[0:m2]

            roll += self.batchSize
            if roll > m:
                roll -= m

            return X, Y, roll

    # Continuously read and generate data for the model
    # As it likely can't fit in memory
    def generator(self):

        i = 0
        m = 0
        mi = 0
        roll = 0
        XX, YY = np.zeros(1), np.zeros(1)
        # Number of mini-batches we can read from a file
        fileLoadsPb = -1
        while True:
            # Handle loading from new files when needed
            if mi >= fileLoadsPb:
                mi = 0
                XX, YY = self.loader.nextFile()
                m = np.shape(YY)[0]
                fileLoadsPb = m // self.batchSize
                # Roll for a random spot to start the batch range slice
                roll = random.randint(0, m - 1)


            X, Y, roll = self.getNextChunk(XX, YY, m, roll)

            yield X.astype(np.float32), Y.astype(np.float32)

            mi += 1
            if mi >= fileLoadsPb:
                i += 1

    # This function estimates the # of batches per epoch to do
    # It works under the assumption that all files are equal in size 
    # except possibly one
    def calcStepsPerEpoch(self):
        
        largest = 0
        numFiles = len(self.loader.labelList)
        maxIt = numFiles if numFiles <= 2 else 2

        for i in range(0 , maxIt):
            featurePath = self.featurePath + '/' + self.loader.featureList[i]
            labelPath = self.labelPath + '/' + self.loader.labelList[i]
            X, Y = self.loader.extractNpy(featurePath, labelPath)
            size = np.shape(Y)[0]

            if size > largest: largest = size

        self.samplesEst = largest * numFiles
        self.stepsPerEpoch = self.samplesEst // self.batchSize