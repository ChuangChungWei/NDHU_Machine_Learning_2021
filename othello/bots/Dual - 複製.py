import numpy as np
from othello.OthelloUtil import getValidMoves
from othello.OthelloUtil import executeMove,isEndGameGeneral
import numpy as np
import heapq
from random import choice
from othello.bots.DeepLearning.OthelloModelDual import OthelloModel
from othello.OthelloGame import OthelloGame
import random
from math import sqrt
class BOT1():
    def __init__(self, *args, **kargs):
        self.board_size=8
        self.model = OthelloModel( input_shape=(self.board_size, self.board_size) )
        self.history=[]
        self.collect_gaming_data=False
        try:
            self.model.load_weights()
            print('model loaded')
        except:
            print('no model exist')
            pass
        self.square_weights = [
                [120, -20,  20,   5,   5,  20, -20, 120],
                [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
                [ 20,  -5,  15,   3,   3,  15,  -5,  20],
                [  5,  -5,   3,   3,   3,   3,  -5,   5],
                [  5,  -5,   3,   3,   3,   3,  -5,   5],
                [ 20,  -5,  15,   3,   3,  15,  -5,  20],
                [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
                [120, -20,  20,   5,   5,  20, -20, 120]
            ]
    
    
    # def getAction(self, game, color):
    #     valids=getValidMoves(game, color)
    #     position=np.random.choice(range(len(valids)), size=1)[0]
    #     position=valids[position]
    #     return position
    
    def get_squares(self,board, color):
        squares=[]
        for y in range(8):
            for x in range(8):
                if board[x][y]==color:
                    squares.append((x,y))
        return squares

    def evaluate(self, board, color):
        opponent = -color
        total = 0

        my_squares = self.get_squares(board, color)

        for square in my_squares:
            total += self.square_weights[square[0]][square[1]]

        opp_squares = self.get_squares(board, opponent)

        for square in opp_squares:
            total -= self.square_weights[square[0]][square[1]]

        return total

    def do_minimax_with_alpha_beta(self, board, color, depth, my_best, opp_best):
        #This was for the statistics section. Commented it out now
        #self.node_count += 1

        if depth == 0:
            return (self.evaluate(board, color), None)

        move_list = getValidMoves(board, color)
        
        #This was for the statistics section. Commented it out now
        #self.branches.append(len(move_list))

        if move_list.size==0:
            return (self.evaluate(board, color), None)

        best_score = my_best
        best_move = None

        for move in move_list:
            new_board = board.copy()
            executeMove(new_board, color, move)



            try_tuple = self.do_minimax_with_alpha_beta(new_board, -color, depth-1, -opp_best, -best_score)
            try_score = -try_tuple[0]
            

            if try_score > best_score:
                best_score = try_score
                best_move = move

            if best_score > opp_best:
                return (best_score, best_move)

        return (best_score, best_move)

    def getAction(self, game, color):
        predict = self.model.predict( game )
        #print(predict)
        
        if self.collect_gaming_data:
            d=random.randint(1, 5)
            d=4
            valid_positions=getValidMoves(game, color)
            valids=np.zeros((game.size), dtype='int')
            valids[ [i[0]*self.board_size+i[1] for i in valid_positions] ]=1
            print("history")
            print(len(self.history))
            if len(self.history)>=40:
                position=self.do_minimax_with_alpha_beta(game, color, 5, -100000, 10000)[1]
            elif len(self.history)>=6:
                position=self.do_minimax_with_alpha_beta(game, color, 4, -100000, 10000)[1]
            else:
                position=None
            
            
            #position=pv_mcts_action(game,1.0,color,self.model)






            zeros=np.zeros(64)
            
            d=random.randint(0, 63)
            try:
                p=position[0]*8+position[1]
            except:
                d=random.randint(0, 63)
                position=d                                                 
                while valids[d]==0:
                    d=random.randint(0,63)
                    position=d
                p=d
                
                position=(position//self.board_size, position%self.board_size)
         
            tmp=zeros=np.zeros(64)
            
            tmp[p]=1.0
            self.history.append([np.array(game.copy()), tmp, color])
        return position
    def self_play_train(self, args):
        self.collect_gaming_data=True
        def gen_data():
            def getSymmetries(board, pi):
                # mirror, rotational
                pi_board = np.reshape(pi, (len(board), len(board)))
                l = []
                for i in range(1, 5):
                    for j in [True, False]:
                        newB = np.rot90(board, i)
                        newPi = np.rot90(pi_board, i)
                        if j:
                            newB = np.fliplr(newB)
                            newPi = np.fliplr(newPi)
                        l += [( newB, list(newPi.ravel()) )]
                return l
            self.history=[]
            history=[]
            game=OthelloGame(self.board_size)
            game.play(self, self, verbose=args['verbose'])
            for step, (board, probs, player) in enumerate(self.history):
                sym = getSymmetries(board, probs)
                for b,p in sym:
                    history.append([b, p, player])
            #print(history)
            self.history.clear()
            game_result=game.isEndGame()
            self.game_result=game_result
            return [(x[0],x[1],[game_result]) for x in history if (game_result==0 or x[2]==game_result)]
        
        data=[]
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('self playing', i+1)
            
            data+=gen_data()
        
        
        self.collect_gaming_data=False
        
        self.model.fit(data, batch_size = args['batch_size'], epochs = 8)
        self.model.save_weights()

def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

def pv_mcts_scores(board, temperature,color,model):
    class Node:
        def __init__(self, board, p,color):
            self.board = board # 盤面狀態
            
            self.p = p # 策略
            self.w = 0 # 累計價值
            self.n = 0 # 試驗次數
            self.color=color
            self.child_nodes = None # 子節點群

        def evaluate(self):
            self.isEndGame=isEndGameGeneral(board)
            if self.isEndGame:
                value=-1 if self.isEndGame==(-1*self.color) else 0
                self.w+=value
                self.n+=1
                return value
            
            if not self.child_nodes:
                policies, value = model.predict(board)
                self.w += value[0][0]
                self.n += 1

                # 擴充子節點
                self.child_nodes = []

                valid_positions=getValidMoves(board, self.color)
                valids=np.zeros((board.size), dtype='int')
                valids[ [i[0]*8+i[1] for i in valid_positions] ]=1
                policies*=valids


                
                for i in range(len(policies[0])):
                    if policies[0][i]==0:
                        continue
                    else:
                        new_board = board.copy()
                        executeMove(new_board, color, [int(i/8),int(i%8)])
                        policy, value = model.predict(new_board)
                        
                        self.child_nodes.append(Node(new_board, policies[0][i],-self.color))
                return value[0][0]
            else:
                value=-self.next_child_node().evaluate()
                self.w+=value
                self.n+=1
                return value
        def next_child_node(self):
            C_PUCT=1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                #print("child_node:",type(child_node.w))
                #print("cacu:",(-child_node.w / child_node.n if child_node.n else 0.0) +
                 #   C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
                
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))
            #print("len",len(self.child_nodes),len(pucb_values))
            
            #print(pucb_values,np.argmax(pucb_values))
            return self.child_nodes[np.argmax(pucb_values)]
    
    root_node = Node(board, 0,color)

    for _ in range(50):
        root_node.evaluate()

    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores


def pv_mcts_action(board,temperature,color,model):
    scores = pv_mcts_scores(board, temperature,color,model)
    valid_positions=getValidMoves(board, color)
    valids=np.zeros((board.size), dtype='int')
    v=[]
    for i in valid_positions:
        v.append((i[0]*8+i[1]))
    #print("success")
    if(len(v)==0):
        return None
    else:
        return np.random.choice(v, p=scores)


# 波茲曼分布
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


                
        

