import numpy as np
from Globals import BoardLength, BLACK, WHITE
import matplotlib.pyplot as plt
import cntk

class NetHeatMap:
    def __init__(self, net, generator):
        self.net    = net
        self.gen    = generator
        self.wStn   = plt.imread('./img/whiteStone.png')
        self.bStn   = plt.imread('./img/blackStone.png')
        self.shp    = np.shape(self.wStn)

    def genHeatmap(self):
        X, Y, W = next(self.gen)
        outs    = self.net(X)
        exOut   = np.zeros((BoardLength, BoardLength))
        for x in range(BoardLength):
            for y in range(BoardLength):
                exOut[x,y] = outs[0][0][y * BoardLength + x]

        exOut         = cntk.softmax(exOut).eval()
        imgIn, imgOut = self.buildImages(X[0], exOut)

        fig = plt.figure(frameon=False)
        plt.imshow(imgIn)
        plt.imshow(imgOut, cmap=plt.cm.viridis, alpha=.6, interpolation='bilinear')
        plt.colorbar()
        ticks  = np.arange((5000//BoardLength)//2, 5000, 5000//BoardLength)
        labels = ('A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','Q','R','S','T')
        plt.xticks(ticks, labels)
        plt.yticks(ticks, reversed(range(1,BoardLength+1)))
        plt.show()

    def buildImages(self, boards, netOut):
        toMove  = boards[0,0,0]
        # Build the boards with the right color stone based on stm
        stm     = boards[1] if toMove == 0 else boards[1] + boards[1]
        opp     = boards[2] if toMove == 1 else boards[2] + boards[2]
        board   = stm + opp
        boardImg    = np.zeros((BoardLength*self.shp[0],BoardLength*self.shp[1],self.shp[2]))
        boardImg    = self.buildBoardImg(board, boardImg, toMove)
        outImg      = self.buildOutImage(netOut)

        return boardImg, outImg

    def addStoneToImg(self, boardImg, x, y, stoneImg):
        
        offX = x * self.shp[0]
        offY = y * self.shp[1]
        boardImg[offX:offX+self.shp[0], offY:offY+self.shp[1]] = stoneImg

    def buildBoardImg(self, board, boardImg, toMove):

        for x in range(BoardLength):
            for y in range(BoardLength):
                img = None
                if board[x, y] == BLACK:
                    img = self.bStn
                elif board[x, y] == WHITE:
                    img = self.wStn
                else:
                    continue
                self.addStoneToImg(boardImg, x, y, img)

        return boardImg

    # Convert the output image to the same
    # shape as the board image with pieces
    def buildOutImage(self, netOut):
        outImg = np.zeros((BoardLength*self.shp[0],BoardLength*self.shp[1]))

        for x in range(BoardLength):
            for y in range(BoardLength):
                offX = x * self.shp[0]
                offY = y * self.shp[1]
                outImg[offX:offX+self.shp[0], offY:offY*self.shp[1]] = netOut[x,y]
        return outImg
