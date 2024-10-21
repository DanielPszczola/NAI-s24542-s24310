"""
Zasady gry w Hexapawn:

1. Hexapawn to gra planszowa dla dwóch graczy, grających na planszy 3x3.
2. Gracz 1 (X) i Gracz 2 (O) mają po 3 pionki na początku gry.
3. Celem Gracza 1 (X) jest dotarcie do ostatniego rzędu planszy (rzędu 2).
4. Celem Gracza 2 (O) jest zablokowanie Gracza 1 (X) lub dotarcie do pierwszego rzędu planszy (rzędu 0).
5. Gracze wykonują ruchy na zmianę.
6. Pionki poruszają się do przodu na jedno pole lub biją na ukos pionka przeciwnika.
7. Jeśli pionek dotrze do ostatniego rzędu planszy, gracz wygrywa.
8. Gra kończy się również, gdy jeden z graczy nie ma możliwości wykonania ruchu.

Autorzy: Michał Kaczmarek s24310, Daniel Pszczoła s24542

Przygotowanie środowiska: Instalacja easyai np. za pomocą pip

"""

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax


class HexapawnGame(TwoPlayerGame):
    """
    Class for the Hexapawn game using the EasyAI library to add artificial intelligence.
    """

    def __init__(self, players):
        """
        Initializes the game, sets up the board and players.

        Parameters:
        players (list): List of players (can include AI_Player and Human_Player).
        """
        self.players = players
        self.board = [['X', 'X', 'X'],
                      ['.', '.', '.'],
                      ['O', 'O', 'O']]
        self.current_player = 1

    def possible_moves(self):
        """
        Returns a list of all possible moves for the current player.

        Returns:
        list: A list of possible moves in the format ['start_row start_col end_row end_col'].
        """
        moves = []
        player = 'X' if self.current_player == 1 else 'O'  # Player 1 is 'X', player 2 is 'O'

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == player:
                    if player == 'X' and i < 2 and self.board[i + 1][j] == '.':
                        moves.append(f"{i} {j} {i + 1} {j}")
                    elif player == 'O' and i > 0 and self.board[i - 1][j] == '.':
                        moves.append(f"{i} {j} {i - 1} {j}")

                    if player == 'X' and i < 2:
                        if j > 0 and self.board[i + 1][j - 1] == 'O':
                            moves.append(f"{i} {j} {i + 1} {j - 1}")
                        if j < 2 and self.board[i + 1][j + 1] == 'O':
                            moves.append(f"{i} {j} {i + 1} {j + 1}")
                    elif player == 'O' and i > 0:
                        if j > 0 and self.board[i - 1][j - 1] == 'X':
                            moves.append(f"{i} {j} {i - 1} {j - 1}")
                        if j < 2 and self.board[i - 1][j + 1] == 'X':
                            moves.append(f"{i} {j} {i - 1} {j + 1}")

        return moves

    def make_move(self, move):
        """
        Executes a given move.

        Parameters:
        move (str): Move in the format 'start_row start_col end_row end_col'.
        """
        start_row, start_col, end_row, end_col = map(int, move.split())
        player = 'X' if self.current_player == 1 else 'O'

        self.board[start_row][start_col] = '.'
        self.board[end_row][end_col] = player

    def win(self):
        """
        Checks if either player has won the game.

        Returns:
        bool: True if the current player has won, False otherwise.
        """
        player = 'X' if self.current_player == 1 else 'O'

        if player == 'X' and 'X' in self.board[2]:
            return True
        if player == 'O' and 'O' in self.board[0]:
            return True

        return self.possible_moves() == []

    def is_over(self):
        """
        Checks if the game is over.

        Returns:
        bool: True if the game is over, False otherwise.
        """
        return self.win()

    def scoring(self):
        """
        Evaluates the game state from the perspective of the AI.

        Returns:
        int: Game score for the current player (100 for win, 0 for loss).
        """
        return 100 if self.win() else 0

    def show(self):
        """
        Displays the current state of the board in the console.
        """
        for row in self.board:
            print(" ".join(row))
        print()

    def move_is_valid(self, move):
        """
        Checks if a given move is valid.

        Parameters:
        move (str): Move in the format 'start_row start_col end_row end_col'.

        Returns:
        bool: True if the move is valid, False otherwise.
        """
        return move in self.possible_moves()


def print_instructions():
    """
    Displays instructions at the beginning of the game, explaining how to input moves.
    """
    print("Welcome to Hexapawn!")
    print("Instructions:")
    print("Player X (human) starts the game. AI controls the O pieces.")
    print("To make a move, enter the coordinates in the format:")
    print("Start_row Start_col End_row End_col")
    print("For example: '0 0 1 0' moves a piece from position (0,0) to (1,0).")
    print("Good luck!\n")


def start_game():
    """
    Initializes and starts the Hexapawn game.

    Sets up the players (human and AI) and displays instructions.
    The game loop runs until it ends, and the reason for the game ending is printed.
    """
    ai_algo = Negamax(4)
    game = HexapawnGame([Human_Player(), AI_Player(ai_algo)])

    print_instructions()
    game.play()

    if game.win():
        if 'X' in game.board[2]:
            print("Player X has won by reaching the end of the board!")
        elif 'O' in game.board[0]:
            print("Player O (AI) has won by reaching the end of the board!")
        else:
            if game.current_player == 1:
                print("Player O (AI) has won as Player X has no more moves!")
            else:
                print("Player X has won as Player O has no more moves!")


start_game()