from AIGamePlatform import Othello
#from othello.bots.Random import BOT
from othello import OthelloGame

from othello.bots.DeepLearning import BOT
from othello.bots.Dual import BOT1
from othello1.bots.DeepLearning import BOTRandom
import time



class Human:
    def getAction(self, game, color):
        print('input coordinate:', end='')
        coor=input()
        return (int(coor[1])-1, ord(coor[0])-ord('A'))
        
BOARD_SIZE=8
bot=BOT(board_size=BOARD_SIZE)
#bot1=BOT1(board_size=BOARD_SIZE)


args={
    'num_of_generate_data_for_train': 8,
    'epochs': 5,
    'batch_size': 4,
    'verbose': True
}
# bot.self_play_train(args)

countB=0
countW=0

app = OthelloGame(n=8) # 會開啟瀏覽器登入Google Account，目前只接受@mail1.ncnu.edu.tw及@mail.ncnu.edu.tw

for i in range(100):
    app = OthelloGame(n=8)
    win=app.play(black=BOTRandom(),white=bot)
    time.sleep(1.5)
    if win==1:
        countB=countB+1
    elif win==-1:
        countW=countW+1
print("Final result")
print("Black:"+str(countB))
print("White:"+str(countW))
    
    

