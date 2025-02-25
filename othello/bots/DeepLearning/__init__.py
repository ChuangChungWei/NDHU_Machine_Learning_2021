import numpy as np
from othello.OthelloUtil import getValidMoves,executeMove
from othello.bots.DeepLearning.OthelloModelDual import OthelloModel
from othello.OthelloGame import OthelloGame
import heapq
from random import choice
import random

class BOT():

    def __init__(self, board_size, *args, **kargs):
        self.board_size=board_size
        self.model = OthelloModel( input_shape=(self.board_size, self.board_size) )
        try:
            self.model.load_weights()
            print('model loaded')
        except:
            print('no model exist')
            pass
        
        self.collect_gaming_data=False
        self.history=[]
        self.weight=[50,-25,10,5,5,10,-25,50,-25,-50,1,1,1,1,-50,-25,10,1,3,2,2,3,1,10,5,1,2,1,1,2,1,5,5,1,2,1,1,2,1,5,10,1,3,2,2,3,1,10,-25,-50,1,1,1,1,-50,-25,50,-25,10,5,5,10,-25,50]
    
    def getAction(self, game, color):
        predict,value = self.model.predict( game )
        predict=predict[0]
        print("value:")
        print(value)
        #print("predict:")
        #print(predict)
        valid_positions=getValidMoves(game, color)
        valids=np.zeros((game.size), dtype='int')
        valids[ [i[0]*self.board_size+i[1] for i in valid_positions] ]=1

        best_position=valid_positions[0]
        best_diff=2
        for x,y in valid_positions:
            new_board = game.copy()
            executeMove(new_board, color, [x,y])
            policy, value = self.model.predict(new_board)
            diff=abs(value[0][0]-color)
            #print("x,y,color,value")
            #print(x,y,color,value[0][0])
            if(diff<best_diff):
                best_diff=diff
                best_position=(x,y)
        #print("best")
        #print(best_position)


        predict*=valids
        zeros=np.zeros(64)
        print(np.array_equal(zeros, predict))
        if np.array_equal(zeros, predict):
            predict=valids

            d=random.randint(0,63)
            position=d
            while valids[d]==0:
                d=random.randint(0,63)
                position=d

        else: 
            
            '''
            if game[0][0]==0:
                predict[9]=predict[9]/2
            if game[0][7]==0:
                predict[14]=predict[14]/2
            if game[7][0]==0:
                predict[49]=predict[49]/2
            if game[7][7]==0:
                predict[54]=predict[54]/2

            if predict[0]>0:
                predict[0]=predict[0]*2
            if predict[7]>0:
                predict[7]=predict[7]*2
            if predict[56]>0:
                predict[56]=predict[56]*2
            if predict[63]>0:
                predict[63]=predict[63]*2

            if game[0][0] and game[0][2]==0:
                predict[1]=predict[1]/2
            if game[0][5] and game[0][7]==0:
                predict[6]=predict[6]/2
            if game[7][0] and game[7][2]==0:
                predict[57]=predict[57]/2
            if game[7][5] and game[7][7]==0:
                predict[62]=predict[62]/2
            if game[0][0] and game[2][0]==0:
                predict[8]=predict[8]/2
            if game[7][0] and game[5][0]==0:
                predict[48]=predict[48]/2
            if game[0][7] and game[2][7]==0:
                predict[15]=predict[15]/2
            if game[7][7] and game[5][7]==0:
                predict[55]=predict[55]/2
            '''
            for i in range(64):
                if predict[i]==0:
                    predict[i]=-100
                else:
                    predict[i]=predict[i]*(self.weight[i]/10)

            position=np.argmax(predict)
            
            
        
        '''
        for i in range(64):
            if predict[i]==0:
                predict[i]=-100
            else:
                predict[i]=predict[i]*self.weight[i]
        #print(predict)




        best= np.argmax(predict)
        print("best")
        print(best)
        tpredict=list(predict)
        max_num_index=map(tpredict.index,heapq.nlargest(3,tpredict))
        #print(tpredict)
        t_max=list(max_num_index)
        print("t_max")
        print(t_max)
        c=0
        for i in range(len(t_max)):
            print(t_max[c])
            if(t_max[c]==best):
                c=c+1
                continue
            if(tpredict[t_max[c]]==-100):
                
                t_max.remove(t_max[c])
                c=c-1
            elif tpredict[t_max[c]]/best<0.9:
                t_max.remove(t_max[c])
                c=c-1
            c=c+1
            
        print("t_max")
        print(t_max)
        position=choice(t_max)
        print("position")
        print(position)
        '''
        

        
        if self.collect_gaming_data:
            tmp=np.zeros_like(predict)
            tmp[position]=1.0
            self.history.append([np.array(game.copy()), tmp, color])
        
        position=(position//self.board_size, position%self.board_size)
        #position=best_position
        #print("position",position)
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
            
            self.history.clear()
            game_result=game.isEndGame()
            print("history")
            print(history)
            print("x012")
            print(x[0],x[1],x[2])
            return [(x[0],x[1]) for x in history if (game_result==0 or x[2]==game_result)]

        
        data=[]
        for i in range(args['num_of_generate_data_for_train']):
            if args['verbose']:
                print('self playing', i+1)
            data+=gen_data()
        print(data)
        self.collect_gaming_data=False
        
        self.model.fit(data, batch_size = args['batch_size'], epochs = 8)
        self.model.save_weights()
        
