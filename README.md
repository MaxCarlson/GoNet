# GoNet
Classifier for the game of Go. Give it a Go board state and it will predict where the pro would move and how likely either side is to win the game from the current board state. 

## Best accuracy
Policy Head ~50%

Value Head  ~70%

### Data format
### Example 
Heatmap of networks PolicyHead outputs over an input board state. Also includes ValueHeads winner prediction, as well as the actual winner. Generated using the NetHeatMap class.
![](https://raw.githubusercontent.com/MaxCarlson/GoNet/master/img/Examples/NetInOut.PNG)

#### Features
Go net is designed to take input data similarly formatted to [AlphaZero](https://applied-data.science/static/main/res/alpha_go_zero_cheat_sheet.png). The input data is in the format (Samples)xPx19x19 where P is the total number of binary feature planes, and 19x19 is the go game board or the color plane. The first feature plane (color) is either all zeros (BLACK) or all ones (WHITE). The following feature planes are separated into stacks of two planes per board state. The first plane (m) after the color plane is a binary grid of the side-to-moves pieces from the most recent board state (S), while plane m+1 is composed of a similar structure but consists of the opponents pieces from state S. Further feature planes are built from states S-n where n is at most (P-1)/2.

#### Labels
The input format GoNet takes labels in is (Samples)x3. The three labels in order are: Index of the move on the board, color of side to move (BLACK = 1, WHITE = 2), and who won the game.

### Current work
Refining the network composition.

## Data
In order to produce the data format required for GoNet you must either feed GoNet data produced from the [TYGEM dataset](https://github.com/yenw/computer-go-dataset#1-tygem-dataset) with my [TygemParser](https://github.com/MaxCarlson/TygemParser) or use a similar data converter for the [Professional dataset](https://github.com/yenw/computer-go-dataset#1-tygem-dataset) located in my [GoClassifier](https://github.com/MaxCarlson/GoClassifier). Or, produce your own dataset in the correct format.
