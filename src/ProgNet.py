
## The credits for this file go to Maxwell J Jacobson for implementing Doric
## you can learn more details about Doric in this repository :
# https://github.com/arcosin/Doric


import torch.nn as nn


"""
Class that acts as the base building-blocks of ProgNets.
Includes a module (usually a single layer),
a set of lateral modules, and an activation.
"""
class ProgBlock(nn.Module):
    """
    Runs the block on input x.
    Returns output tensor or list of output tensors.
    """
    def runBlock(self, x):
        raise NotImplementedError

    """
    Runs lateral i on input x.
    Returns output tensor or list of output tensors.
    """
    def runLateral(self, i, x):
        raise NotImplementedError

    """
    Runs activation of the block on x.
    Returns output tensor or list of output tensors.
    """
    def runActivation(self, x):
        raise NotImplementedError



"""
A column representing one sequential ANN with all of its lateral modules.
Outputs of the last forward run are stored for child column laterals.
Output of each layer is calculated as:
y = activation(block(x) + sum(laterals(x)))
"""
class ProgColumn(nn.Module):
    def __init__(self, colID, blockList, parentCols = []):
        super().__init__()
        self.colID = colID
        self.isFrozen = False
        self.parentCols = parentCols
        self.blocks = nn.ModuleList(blockList)
        self.numRows = len(blockList)
        self.lastOutputList = []

    def freeze(self, unfreeze = False):
        if not unfreeze:    # Freeze params.
            self.isFrozen = True
            for param in self.parameters():   param.requires_grad = False
        else:               # Unfreeze params.
            self.isFrozen = False
            for param in self.parameters():   param.requires_grad = True

    def forward(self, input):
        outputs = []
        x = input
        for row, block in enumerate(self.blocks):
            currOutput = block.runBlock(x)
            if row == 0 or len(self.parentCols) < 1:
                y = block.runActivation(currOutput)
            else:
                for c, col in enumerate(self.parentCols):
                    temp = block.runLateral(c, col.lastOutputList[row - 1])
                    currOutput += temp
                y = block.runActivation(currOutput)
            outputs.append(y)
            x = y
        self.lastOutputList = outputs
        return outputs[-1]


"""
A progressive neural network as described in Progressive Neural Networks (Rusu et al.).
Columns can be added manually or with a ProgColumnGenerator.
https://arxiv.org/abs/1606.04671
"""
class ProgNet(nn.Module):
    def __init__(self, colGen = None):
        super().__init__()
        self.columns = nn.ModuleList()
        self.numRows = None
        self.numCols = 0
        self.colMap = dict()
        self.colGen = colGen
        self.colShape = None

    def addColumn(self, device,col = None, msg = None):
        if not col:
            parents = [colRef for colRef in self.columns]
            col = self.colGen.generateColumn(device = device,parent_cols = parents)
        self.columns.append(col)
        self.colMap[col.colID] = self.numCols
        self.numRows = col.numRows
        self.numCols += 1
        return col.colID

    def freezeColumn(self, id):
        col = self.columns[self.colMap[id]]
        col.freeze()

    def freezeAllColumns(self):
        for col in self.columns:
            col.freeze()

    def unfreezeColumn(self, id):
        col = self.columns[self.colMap[id]]
        col.freeze(unfreeze = True)

    def unfreezeAllColumns(self):
        for col in self.columns:
            col.freeze(unfreeze = True)

    def getColumn(self, id):
        col = self.columns[self.colMap[id]]
        return col

    def forward(self, id, x):
        colToOutput = self.colMap[id]
        for i, col in enumerate(self.columns):
            y = col(x)
            if i == colToOutput:
                return y




"""
Class that generates new ProgColumns using the method generateColumn.
The parentCols list will contain references to each parent column,
such that columns can access lateral outputs.
Additional information may be passed through the msg argument in
generateColumn and ProgNet.addColumn.
"""
class ProgColumnGenerator:
    def generateColumn(self, parentCols, msg = None):
        raise NotImplementedError