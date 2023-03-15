import chess
import chess.pgn
import numpy as np
import zstandard as zstd
import matplotlib.pyplot as plt

def sigmoid(x):
  if x < 0:
      return np.exp(x) / (1 + np.exp(x))
  else:
      return 1/(1+np.exp(-x))

def square_to_index(square):
    return chess.square_rank(square), chess.square_file(square)

def get_bitmap(board) -> np.array:
    board3d = np.zeros((14, 8, 8), dtype=np.int8)

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            id = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - id[0]][id[1]] = 1

        for square in board.pieces(piece, chess.BLACK):
            id = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - id[0]][id[1]] = 1

    temp = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1

    board.turn = temp

    return board3d

dctx = zstd.ZstdDecompressor()

with open("Chess/raw/lichess_db_standard_rated_2016-12.pgn.zst", 'rb') as ifh, open("Chess/training_data/pgn/lichess_db_standard_rated_2016-12.pgn", 'wb') as ofh:
    dctx.copy_stream(ifh, ofh)


parsed_games = []
ratings = []
normalized_rating = []

decompressed_file = open("Chess/training_data/pgn/lichess_db_standard_rated_2016-12.pgn")

i = 0
j = 0

while i < 10000:
    game = chess.pgn.read_game(decompressed_file)
    if game is None:
        break
    
    if not game.end().eval() == None:
        board = game.board()
        for move in game.mainline():
            board.push(move.move)
            if not move.eval().relative.score() == None and not():
                parsed_games.append([get_bitmap(board), (move.eval().relative.wdl().expectation() - 0.5) * 2])


    print("Parsed game #" + str(i))

    i += 1

#print(*parsed_games, file=open("output2.pgn", "w"), sep="\n\n", end="\n\n")

parsed_games = np.array(parsed_games, dtype=object)

np.savez("Chess/training_data/bitmaps/lichess_2023_01.npz", *parsed_games)