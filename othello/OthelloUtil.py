import numpy as np

def getValidMoves(board, color):
    moves = set()
    for y,x in zip(*np.where(board==color)):
        for direction in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
            flips = []
            for size in range(1, len(board)):
                ydir = y + direction[1] * size
                xdir = x + direction[0] * size
                if xdir >= 0 and xdir < len(board) and ydir >= 0 and ydir < len(board):
                    if board[ydir][xdir]==-color:
                        flips.append((ydir, xdir))
                    elif board[ydir][xdir]==0:
                        if len(flips)!=0:
                            moves.add((ydir, xdir))
                        break
                    else:
                        break
                else:
                    break
    return np.array(list(moves))

def isValidMove(board, color, position):
    valids=getValidMoves(board, color)
    if len(valids)!=0 and (valids==np.array(position)).all(1).sum()!=0:
        return True
    else:
        return False

def executeMove(board, color, position):
    y, x = position

    board[y][x] = color
    for direction in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
        flips = []
        valid_route=False
        for size in range(1, len(board)):
            ydir = y + direction[1] * size
            xdir = x + direction[0] * size
            if xdir >= 0 and xdir < len(board) and ydir >= 0 and ydir < len(board):
                if board[ydir][xdir]==-color:
                    flips.append((ydir, xdir))
                elif board[ydir][xdir]==color:
                    if len(flips)>0:
                        valid_route=True
                    break
                else:
                    break
            else:
                break
        if valid_route:
            board[tuple(zip(*flips))]=color
    
def isEndGameGeneral(board):
    white_valid_moves=len(getValidMoves(board, -1))
    black_valid_moves=len(getValidMoves(board, 1))
    if white_valid_moves==0 and black_valid_moves==0:
        v,c=np.unique(board, return_counts=True)
        white_count=c[np.where(v==-1)]
        black_count=c[np.where(v==1)]
        if white_count>black_count:
            return -1
        elif black_count>white_count:
            return 1
        else:
            return 0
    else:
        return None
