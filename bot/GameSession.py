import enum
import numpy as np
import socket
import json
import collections
from typing import Dict, List, Tuple, NamedTuple, Union, Any


class Point(NamedTuple):
    x: int = 0
    y: int = 0

class PlayerInfos(NamedTuple):
    position: Point
    direction: Point
    score : int

class GameSession:
    class Type(enum.Enum):
        NORMAL = enum.auto()
        ALLVS1 = enum.auto()

    class Order(enum.Enum):
        STAY = enum.auto()
        TURNLEFT = enum.auto()
        TURNRIGHT = enum.auto()
        MOVE = enum.auto()
        STAY_SHOOT = enum.auto()
        TURNLEFT_SHOOT = enum.auto()
        TURNRIGHT_SHOOT = enum.auto()
        MOVE_SHOOT = enum.auto()
        SHOOT_TURNLEFT = enum.auto()
        SHOOT_TURNRIGHT = enum.auto()

    socket : Any
    width : int
    height : int
    walls : np.ndarray
    bonus : np.ndarray
    players : np.ndarray
    playerInfos : Dict[int, PlayerInfos]
    playerId : int
    lastOrder : Order

    def __init__(self, host:str, port:int) -> None:
        self.socket = socket.socket()
        self.socket.connect((host, int(port)))

        self.width = 0
        self.height = 0
        self.walls = None # 0: empty, 1: breakable, 2: unbreakable
        self.bonus = None # 0: bonus, 1: bonus
        self.players = None # -1: no players, else: player ID
        self.playerInfos = None
        self.playerId = None

        self.lastOrder = None

    def login(self, login:str) -> None:
        message = {'nickname': login}
        self.socket.send(json.dumps(message).encode() + b'\n')
        self._update()

    def __del__(self) -> None:
        self.socket.close()

    def _compute(self, networkData:Any) -> None:
        print(networkData)
        if 'idJoueur' in networkData:
            self.playerId = networkData['idJoueur']

        if 'map' in networkData:
            # Retrieve map size
            self.width = -1
            self.height = -1
            for cell in networkData['map']:
                x, y = cell['pos']
                if x >= self.width:
                    self.width = x+1
                if y >= self.height:
                    self.height = y+1

            # Initialization
            self.walls = np.ndarray(shape=(self.height,self.width), dtype=int) # 0: empty, 1: breakable, 2: unbreakable
            self.walls.fill(0)
            self.bonus = np.ndarray(shape=(self.height,self.width), dtype=int) # 0: no bonus, else: points of the bonus
            self.bonus.fill(0)
            self.players = np.ndarray(shape=(self.height,self.width), dtype=int) # -1: no players, else: player ID
            self.players.fill(-1)

            # Fill the map
            for cell in networkData['map']:
                if 'pos' in cell:
                    x, y = cell['pos']
                if 'cassable' in cell:
                    self.walls[y, x] = 1 if cell['cassable'] else 2
                elif 'points' in cell:
                    self.bonus[y, x] = cell['points']
            print("WALLS:", self.walls)

        if 'joueurs' in networkData:
            self.playerInfos = dict()

            # Update the position
            for player in networkData['joueurs']:
                playerId = player['id']
                x, y = player['position']
                dx, dy = player['direction']
                score = player['score']
                self.playerInfos[playerId] = PlayerInfos(position=Point(x,y), direction=Point(dx,dy), score=score)

                self.players.fill(-1)
                for playerId in self.playerInfos:
                    x, y = self.playerInfos[playerId].position
                    self.players[y, x] = playerId

        if type(networkData) == list:
            assert self.walls is not None
            assert self.bonus is not None
            assert self.players is not None
            assert self.playerInfos is not None

            selfRotate = False
            selfMove = False
            selfShoot = False

            shouldMove = False
            if self.lastOrder in [GameSession.Order.MOVE, GameSession.Order.MOVE_SHOOT]:
                shouldMove = True
                playerInfo = self.playerInfos[self.playerId]
                newPos = Point(playerInfo.position.x+playerInfo.direction.x, playerInfo.position.y+playerInfo.direction.y)
                if self.walls[newPos.y, newPos.x]:
                    shouldMove = False

            # Update: displacement, rotation, bullets (displacement & explosions)
            for actions in networkData:
                if actions[0] == 'joueur':
                    if actions[1] == 'rotate':
                        playerId = actions[2]
                        playerInfo = self.playerInfos[playerId]
                        dx, dy = actions[3]
                        self.playerInfos[playerId] = PlayerInfos(position=playerInfo.position, direction=Point(dx,dy), score=playerInfo.score)
                        if playerId == self.playerId:
                            selfRotate = True
                    elif actions[1] == 'move':
                        playerId = actions[2]
                        playerInfo = self.playerInfos[playerId]
                        newPos = Point(playerInfo.position.x+playerInfo.direction.x, playerInfo.position.y+playerInfo.direction.y)
                        if newPos.x in range(self.width) and newPos.y in range(self.height):
                            if self.walls[newPos.y, newPos.x] == 0:
                                self.playerInfos[playerId] = PlayerInfos(position=newPos, direction=playerInfo.direction, score=playerInfo.score)
                        if playerId == self.playerId:
                            selfMove = True
                    elif actions[1] == 'recupere_bonus':
                        playerId = actions[2]
                        x, y = actions[3]
                        assert self.bonus[y, x] > 0
                        playerInfo = self.playerInfos[playerId]
                        newScore = playerInfo.score + self.bonus[y, x]
                        self.playerInfos[playerId] = PlayerInfos(position=playerInfo.position, direction=playerInfo.direction, score=newScore)
                        self.bonus[y, x] = 0
                    elif actions[1] == 'shoot':
                        playerId = actions[2]
                        bulletId = actions[3]
                        bulletPos = actions[4]
                        bulletDir = actions[5]
                        if playerId == self.playerId:
                            selfShoot = True
                        pass # TODO
                elif actions[0] == 'projectile':
                    if actions[1] == 'move':
                        bulletId = actions[2]
                        bulletPos = actions[3]
                        pass # TODO
                    elif actions[1] == 'explode':
                        projectileId = actions[2]
                        x, y = actions[3]
                        if self.walls[y, x] == 1:
                            self.walls[y, x] = 0

            # Update: player map
            self.players.fill(-1)
            for playerId in self.playerInfos:
                x, y = self.playerInfos[playerId].position
                self.players[y, x] = playerId

            #print("PLAYERS:", self.players)
            #print("BONUS:", self.bonus)

            # Check if the response of the server match with the last order (to check iteration mismatch)
            # Not fully-checked since we cannot make a shoot when a bullets is already launched
            assert self.lastOrder == None or {
                GameSession.Order.STAY: not selfRotate and not selfMove and not selfShoot,
                GameSession.Order.TURNLEFT: selfRotate and not selfMove and not selfShoot,
                GameSession.Order.TURNRIGHT: selfRotate and not selfMove and not selfShoot,
                GameSession.Order.MOVE: not selfRotate and (selfMove == shouldMove) and not selfShoot,
                GameSession.Order.STAY_SHOOT: not selfRotate and not selfMove,
                GameSession.Order.TURNLEFT_SHOOT: selfRotate and not selfMove,
                GameSession.Order.TURNRIGHT_SHOOT: selfRotate and not selfMove,
                GameSession.Order.MOVE_SHOOT: not selfRotate and (selfMove == shouldMove),
                GameSession.Order.SHOOT_TURNLEFT: selfRotate and not selfMove,
                GameSession.Order.SHOOT_TURNRIGHT: selfRotate and not selfMove
            }[self.lastOrder], 'Mismatch between order sent and order reported from the server'
            self.lastOrder = None

    def _update(self) -> None:
        rawData = b''
        while not rawData.endswith(b'\n'):
            rawData += self.socket.recv(4096)

        for networkData in rawData.decode().strip().split('\n'):
            self._compute(json.loads(networkData))

    def startRound(self) -> bool:
        self._update()
        return True # TODO

    def endRound(self, order:Order) -> None:
        action : List[str]
        rawData : bytes

        if order == GameSession.Order.STAY:
            action = []
        elif order == GameSession.Order.TURNLEFT:
            action = ['trotate']
        elif order == GameSession.Order.TURNRIGHT:
            action = ['hrotate']
        elif order == GameSession.Order.MOVE:
            action = ['move']
        elif order == GameSession.Order.STAY_SHOOT:
            action = ['shoot']
        elif order == GameSession.Order.TURNLEFT_SHOOT:
            action = ['trotate', 'shoot']
        elif order == GameSession.Order.TURNRIGHT_SHOOT:
            action = ['hrotate', 'shoot']
        elif order == GameSession.Order.MOVE_SHOOT:
            action = ['move', 'shoot']
        elif order == GameSession.Order.SHOOT_TURNLEFT:
            action = ['shoot', 'trotate']
        elif order == GameSession.Order.SHOOT_TURNRIGHT:
            action = ['shoot', 'hrotate']
        else:
            assert False, 'Invalid order'
        rawData = json.dumps(action).encode() + b'\n'
        self.socket.send(rawData)
        self.lastOrder = order

    def printMap(self, colored:bool=True) -> None:
        pass
        #print('+%s+' % ('-'*(self.map.shape[1]*2+2)))
        #for y in range(self.map.shape[0]):
        #    print('| ', end='')
        #    for x in range(self.map.shape[1]):
        #        if colored:
        #            if self.bombs[y,x]:
        #                print('\033[48;5;1m', end='')
        #            elif self.map[y,x] in [GameSession.Map.BONUS_BOMB.value,
        #                                    GameSession.Map.BONUS_POINT.value,
        #                                    GameSession.Map.BONUS_EXPLOSION.value,
        #                                    GameSession.Map.BONUS_TELEPORT.value]:
        #                print('\033[48;5;2m', end='')
        #            elif self.map[y,x] in [GameSession.Map.PLAYER0.value,
        #                                    GameSession.Map.PLAYER1.value,
        #                                    GameSession.Map.PLAYER2.value,
        #                                    GameSession.Map.PLAYER3.value]:
        #                print('\033[48;5;4m', end='')
        #            elif self.map[y,x] == GameSession.Map.WALL.value:
        #                print('\033[48;5;8m', end='')
        #            else:
        #                print('\033[48;5;0m', end='')
        #        print(('%s' % self.map[y,x].decode())*2, end='')
        #    if colored:
        #        print('\033[0m', end='')
        #    print(' |')
        #print('+%s+' % ('-'*(self.map.shape[1]*2+2)))

#if __name__ == '__main__':
#    pass

