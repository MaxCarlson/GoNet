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

    def genHeatmap(self, count):

        inpGen = self.processInput(count)
        for i in range(count):
            imgIn, imgOut, toMove, win, predWin = next(inpGen)
            toMvTxt, toMvStr = self.toMoveStr(toMove)
            predStr, winStr  = self.winChanceStr(win, toMvStr, predWin)

            fig     = plt.figure(frameon=False)
            ax      = fig.add_subplot(111)
            plt.text(0.5, 1.10, toMvTxt, ha='center', va='center', transform=ax.transAxes)
            plt.text(0.5, 1.06, predStr, ha='center', va='center', transform=ax.transAxes)
            plt.text(0.5, 1.02, winStr , ha='center', va='center', transform=ax.transAxes)

            plt.imshow(imgIn)
            plt.imshow(imgOut, cmap=plt.cm.hot, alpha=.6, interpolation='bilinear')
            cbar    = plt.colorbar()
            cbar.set_label('% Move chance', rotation=270, labelpad=10)
            ticks   = np.arange((5000/BoardLength)/2, 5000, 5000/BoardLength)
            labels  = ('A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','Q','R','S','T')
            plt.xticks(ticks, labels) 
            plt.yticks(ticks, reversed(range(1,BoardLength+1)))
            plt.show(block=True)

    def toMoveStr(self, toMove):
        str = '{} to move'
        toMvStr = 'Black' if toMove == BLACK else 'White'
        return str.format(toMvStr), toMvStr

    def winChanceStr(self, win, toMvStr, predWin):
        predStr = '{:.2f}% predicted win chance for {}'.format(predWin*100.0, toMvStr)
        winName = toMvStr if win == 1 else 'Not'
        if winName == 'Not':
            winName = 'Black' if toMvStr == 'White' else 'White'
        actWin  = 'Actual winner of game: {}'.format(winName)
        return predStr, actWin

    def processInput(self, count):
        X, Y, W = next(self.gen)
        outs    = self.net(X)

        for i in range(count):
            exOut   = np.zeros((BoardLength, BoardLength))
            outVec  = cntk.softmax(outs[0][i]).eval() * 100.0
            winVec  = cntk.softmax(outs[1][i]).eval()
            for x in range(BoardLength):
                for y in range(BoardLength):
                    exOut[x,y] = outVec[y * BoardLength + x]

            win           = W[i,1]
            predWin       = winVec[1]
            toMove        = X[i,0,0,0]
            imgIn, imgOut = self.buildImages(X[i], exOut, toMove)
            yield imgIn, imgOut, toMove, win, predWin

    def buildImages(self, boards, netOut, toMove):
        # Build the boards with the right color stone based on stm
        stm     = boards[1] if toMove == 0 else boards[1] + boards[1]
        opp     = boards[2] if toMove == 1 else boards[2] + boards[2]
        board   = stm + opp
        boardImg    = np.zeros((BoardLength*self.shp[0],BoardLength*self.shp[1],self.shp[2]))
        boardImg    = self.buildBoardImg(board, boardImg, toMove)
        outImg      = self.buildOutImage(netOut)
        # Remove moves with < 0.5% from heatmap
        #outImg[outImg < 0.005] = np.nan

        return boardImg, outImg

    def addStoneToImg(self, boardImg, x, y, stoneImg):
        offX = x * self.shp[0]
        offY = y * self.shp[1]
        boardImg[offX:offX+self.shp[0], offY:offY+self.shp[1]] = stoneImg

    # Build the board image
    # Scale up the input 19x19 'image' to a 19*(stoneImgSize)x19*(stoneImgSize) image
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
