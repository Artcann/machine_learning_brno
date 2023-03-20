from mcts import mcts
import chess
import chess.pgn
from copy import deepcopy
from chessboard import display
from time import sleep

class ChessState:
    def __init__(self, board, player) -> None:
        self.board = board
        self.children = set()
        self.player = player
        self.current_player = 1

    def getCurrentPlayer(self):
        return self.current_player

    def getPossibleActions(self):
        return list(self.board.legal_moves)

    def takeAction(self, action):
        new_state = deepcopy(self)
        new_state.board = self.board.copy()
        new_state.board.push(action)
        new_state.current_player = -self.current_player
        return new_state

    def isTerminal(self):
        return self.board.outcome() != None

    def getReward(self):
        if self.board.outcome().winner == self.player:
            return 1
        if self.board.outcome().winner == (not self.player):
            return -1
        if self.board.outcome().winner == None:
            return 0
        return False

def play_best_move(board, player):
    initial_state = ChessState(board, player)
    searcher = mcts(timeLimit=10000)

    bestAction = searcher.search(initialState=initial_state)
    board.push(bestAction)