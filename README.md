# Tic Tac Toe - MinMax

This is my implementation of tic tac toe game using the minimax algorithm.

## Assignment Description

Implement Minimax algorithm as a simple tic-tac-toe game. You play as X and the computer is O.

## How the computer thinks

When it's the computer's turn, it follows these steps:

1. Look at all possible moves it can make
2. For each move, it imagines what you might do next
3. It keeps thinking ahead until it reaches an end point (like someone winning or a tie)
4. It picks the move that gives it the best chance of winning

## Scoring system

The computer uses a simple scoring system:

- +10 points if the computer wins
- -10 points if you win
- 0 points if it's a tie

## How it works (Tree of possibilities)

Think of it like a tree of possibilities:

- At the top is the current game board
- Each branch is a possible move
- At the bottom are all the possible results

## Taking turns

The computer switches between two modes of thinking:

1. **Maximizing mode** (computer's turn):

   - Looks for moves that get the highest score
   - Tries to win or force a tie

2. **Minimizing mode** (player's turn):
   - Assumes you'll make moves that get the lowest score
   - Prepares for your best possible moves
