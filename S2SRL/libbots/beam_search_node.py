

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, logits):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.logits = logits

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        # # It gives preference to return longer action sequences.
        # return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

        # It gives preference to return shorter action sequences.
        return self.logp + alpha * reward