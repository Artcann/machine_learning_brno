import chess
import chess.pgn
from copy import deepcopy
from chessboard import display
from chessboard.board import Board
from time import sleep
import pygame
from monte_carlo_tree_search import play_best_move

pgn = open("Chess/training_data/pgn/kasparov_deep_blue_1996.pgn")
game = chess.pgn.read_game(pgn)

board = game.board()
initial_display = display.start(board.fen())

""" for move in game.mainline_moves():
     
    board.push(move)
    display.update(board.fen(), initial_display) """

selected_square = []
ranks = [7, 6, 5, 4, 3, 2, 1, 0]

while not display.check_for_quit():
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            x_mouse, y_mouse = pygame.mouse.get_pos()
            file = (x_mouse - 100) // 50
            rank = (y_mouse - 100) // 50
            
            print(file, rank)
            if file > 8 or rank > 8:
                continue

            square = chess.square(file, ranks[rank])

            if (len(selected_square) == 0 and board.piece_at(square) is not None) or len(selected_square) == 1:
                selected_square.append(square)
                print(x_mouse, y_mouse)
                initial_display.highlight_selected((x_mouse - round(x_mouse % 50), y_mouse - round(y_mouse % 50)))
                print(selected_square)
            else:
                selected_square = []
                display.update(board.fen(), initial_display)
            

        if len(selected_square) == 2:
            move = chess.Move(from_square=selected_square[0], to_square=selected_square[1])
            if move in board.legal_moves:
                board.push(move)
                display.update(board.fen(), initial_display)

                play_best_move(board, board.turn)
                display.update(board.fen(), initial_display)

                
            selected_square = []

display.terminate()