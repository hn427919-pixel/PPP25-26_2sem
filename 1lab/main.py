from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Tuple, Any
from enum import Enum

# ========== Константы и вспомогательные функции ==========

class Color(Enum):
    WHITE = 'white'
    BLACK = 'black'
    
    def opposite(self):
        return Color.BLACK if self == Color.WHITE else Color.WHITE
    
    def __str__(self):
        return 'белых' if self == Color.WHITE else 'чёрных'

WHITE, BLACK = Color.WHITE, Color.BLACK

ICONS = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
    'C': '☩', 'c': '☨', 'A': '⚜', 'a': '⚝', 'Z': '⛨', 'z': '⛧',
}

# Направления движения
KNIGHT_DIRS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
               (1, -2), (1, 2), (2, -1), (2, 1)]
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
QUEEN_DIRS = ROOK_DIRS + BISHOP_DIRS

# ========== Позиция ==========

@dataclass(frozen=True)
class Position:
    row: int
    col: int
    
    def valid(self) -> bool:
        return 0 <= self.row < 8 and 0 <= self.col < 8
    
    def shift(self, dr: int, dc: int) -> 'Position':
        return Position(self.row + dr, self.col + dc)
    
    def __repr__(self) -> str:
        return f"{chr(97 + self.col)}{self.row + 1}"
    
    @staticmethod
    def parse(s: str) -> Optional['Position']:
        if s and len(s) == 2 and s[0] in 'abcdefgh' and s[1] in '12345678':
            return Position(int(s[1]) - 1, ord(s[0]) - 97)
        return None

# ========== Ход ==========

@dataclass
class Move:
    piece: 'Piece'
    frm: Position
    to: Position
    captured: Optional['Piece'] = None
    castling_rook: Optional['Piece'] = None
    rook_frm: Optional[Position] = None
    rook_to: Optional[Position] = None
    ep_pos: Optional[Position] = None
    promoted_to: Optional['Piece'] = None
    prev_ep: Optional[Position] = None
    prev_moved: bool = False
    rook_prev_moved: bool = False
    prev_halfmove: int = 0
    
    def __repr__(self) -> str:
        ic = ICONS.get(self.piece.symbol(), '?')
        cap = 'x' if self.captured else '-'
        p = f"={ICONS.get(self.promoted_to.symbol().upper(), '?')}" if self.promoted_to else ''
        c = (' O-O' if self.rook_frm.col == 7 else ' O-O-O') if self.castling_rook else ''
        e = ' e.p.' if self.ep_pos else ''
        return f"{ic}{self.frm}{cap}{self.to}{p}{c}{e}"

# ========== Абстрактная фигура ==========

class Piece(ABC):
    value: int = 0
    _sym: str = ''
    _name: str = ''
    
    def __init__(self, color: Color, pos: Position):
        self.color = color
        self.pos = pos
        self.moved = False
    
    def symbol(self) -> str:
        return self._sym.upper() if self.color == WHITE else self._sym.lower()
    
    def name(self) -> str:
        return self._name
    
    def icon(self) -> str:
        return ICONS.get(self.symbol(), self.symbol())
    
    @abstractmethod
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        pass
    
    def legal_moves(self, board: 'Board') -> List[Position]:
        return [p for p in self.pseudo_moves(board) if board.move_legal(self, p)]
    
    def attack_moves(self, board: 'Board') -> List[Position]:
        return self.pseudo_moves(board)
    
    def _slide(self, board: 'Board', dirs: List[Tuple[int, int]]) -> List[Position]:
        res = []
        for dr, dc in dirs:
            p = self.pos.shift(dr, dc)
            while p.valid():
                t = board.at(p)
                if t is None:
                    res.append(p)
                elif t.color != self.color:
                    res.append(p)
                    break
                else:
                    break
                p = p.shift(dr, dc)
        return res
    
    def _jumps(self, board: 'Board', deltas: List[Tuple[int, int]]) -> List[Position]:
        res = []
        for dr, dc in deltas:
            p = self.pos.shift(dr, dc)
            if p.valid():
                t = board.at(p)
                if t is None or t.color != self.color:
                    res.append(p)
        return res
    
    def _combined(self, board: 'Board', slide_dirs: List[Tuple[int, int]], 
                  jump_dirs: List[Tuple[int, int]]) -> List[Position]:
        seen = set()
        res = []
        for p in self._slide(board, slide_dirs) + self._jumps(board, jump_dirs):
            if p not in seen:
                seen.add(p)
                res.append(p)
        return res
    
    def __repr__(self) -> str:
        return f"{self.icon()}({self.pos})"

# ========== Конкретные фигуры ==========

class King(Piece):
    _sym, _name = 'K', 'Король'
    
    def attack_moves(self, board: 'Board') -> List[Position]:
        return self._jumps(board, QUEEN_DIRS)
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        moves = self._jumps(board, QUEEN_DIRS)
        if self.moved or board.attacked(self.pos, self.color.opposite()):
            return moves
        
        r = self.pos.row
        for rc, kdc in [(7, 2), (0, -2)]:
            rook = board.at(Position(r, rc))
            if not (rook and isinstance(rook, Rook) and not rook.moved and rook.color == self.color):
                continue
            
            lo, hi = min(rc, self.pos.col) + 1, max(rc, self.pos.col)
            if not all(board.at(Position(r, c)) is None for c in range(lo, hi)):
                continue
            
            step = 1 if kdc > 0 else -1
            if all(not board.attacked(Position(r, self.pos.col + step * i), 
                                      self.color.opposite()) for i in range(1, 3)):
                moves.append(Position(r, self.pos.col + kdc))
        return moves

class Queen(Piece):
    _sym, _name, value = 'Q', 'Ферзь', 9
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        return self._slide(board, QUEEN_DIRS)

class Rook(Piece):
    _sym, _name, value = 'R', 'Ладья', 5
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        return self._slide(board, ROOK_DIRS)

class Bishop(Piece):
    _sym, _name, value = 'B', 'Слон', 3
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        return self._slide(board, BISHOP_DIRS)

class Knight(Piece):
    _sym, _name, value = 'N', 'Конь', 3
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        return self._jumps(board, KNIGHT_DIRS)

class Pawn(Piece):
    _sym, _name, value = 'P', 'Пешка', 1
    
    @property
    def _dir(self) -> int:
        return 1 if self.color == WHITE else -1
    
    @property
    def _start_row(self) -> int:
        return 1 if self.color == WHITE else 6
    
    @property
    def _promo_row(self) -> int:
        return 7 if self.color == WHITE else 0
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        moves = []
        d = self._dir
        
        one = self.pos.shift(d, 0)
        if one.valid() and not board.at(one):
            moves.append(one)
            two = self.pos.shift(2 * d, 0)
            if self.pos.row == self._start_row and not board.at(two):
                moves.append(two)
        
        for dc in (-1, 1):
            cp = self.pos.shift(d, dc)
            if cp.valid():
                t = board.at(cp)
                if (t and t.color != self.color) or cp == board.ep_target:
                    moves.append(cp)
        return moves
    
    def attack_moves(self, board: 'Board') -> List[Position]:
        res = []
        for dc in (-1, 1):
            p = self.pos.shift(self._dir, dc)
            if p.valid():
                res.append(p)
        return res

# ========== Дополнительные фигуры для кастомного режима ==========

class Prince(Piece):
    _sym, _name, value = 'C', 'Князь', 8
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        return self._combined(board, ROOK_DIRS, KNIGHT_DIRS)

class Noble(Piece):
    _sym, _name, value = 'A', 'Боярин', 6
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        return self._combined(board, BISHOP_DIRS, KNIGHT_DIRS)

class Rider(Piece):
    _sym, _name, value = 'Z', 'Наездница', 12
    
    def pseudo_moves(self, board: 'Board') -> List[Position]:
        return self._combined(board, QUEEN_DIRS, KNIGHT_DIRS)

# ========== Доска ==========

class Board:
    def __init__(self):
        self._grid: List[List[Optional[Piece]]] = [[None] * 8 for _ in range(8)]
        self.pieces: List[Piece] = []
        self.ep_target: Optional[Position] = None
    
    def put(self, piece: Piece) -> None:
        self._grid[piece.pos.row][piece.pos.col] = piece
        self.pieces.append(piece)
    
    def remove(self, pos: Position) -> Optional[Piece]:
        piece = self._grid[pos.row][pos.col]
        if piece:
            self._grid[pos.row][pos.col] = None
            if piece in self.pieces:
                self.pieces.remove(piece)
        return piece
    
    def at(self, pos: Position) -> Optional[Piece]:
        return self._grid[pos.row][pos.col] if pos.valid() else None
    
    def move_piece(self, piece: Piece, to: Position) -> None:
        self._grid[piece.pos.row][piece.pos.col] = None
        self._grid[to.row][to.col] = piece
        piece.pos = to
    
    def pieces_by_color(self, color: Color) -> List[Piece]:
        return [p for p in self.pieces if p.color == color]
    
    def king(self, color: Color) -> Optional[King]:
        for piece in self.pieces:
            if isinstance(piece, King) and piece.color == color:
                return piece
        return None
    
    def attacked(self, pos: Position, by: Color) -> bool:
        for piece in self.pieces_by_color(by):
            if pos in piece.attack_moves(self):
                return True
        return False
    
    def in_check(self, color: Color) -> bool:
        king = self.king(color)
        return king is not None and self.attacked(king.pos, color.opposite())
    
    def move_legal(self, piece: Piece, to: Position) -> bool:
        frm = piece.pos
        captured = self.at(to)
        
        # Временно выполняем ход
        self._grid[frm.row][frm.col] = None
        self._grid[to.row][to.col] = piece
        piece.pos = to
        
        if captured and captured in self.pieces:
            self.pieces.remove(captured)
        
        ep_removed = None
        if isinstance(piece, Pawn) and to == self.ep_target and not captured:
            ep_pos = Position(frm.row, to.col)
            ep_removed = self._grid[ep_pos.row][ep_pos.col]
            if ep_removed:
                self._grid[ep_pos.row][ep_pos.col] = None
                if ep_removed in self.pieces:
                    self.pieces.remove(ep_removed)
        
        legal = not self.in_check(piece.color)
        
        # Откатываем изменения
        piece.pos = frm
        self._grid[frm.row][frm.col] = piece
        self._grid[to.row][to.col] = captured
        
        if captured and captured not in self.pieces:
            self.pieces.append(captured)
        
        if ep_removed:
            self._grid[frm.row][to.col] = ep_removed
            if ep_removed not in self.pieces:
                self.pieces.append(ep_removed)
        
        return legal
    
    def has_moves(self, color: Color) -> bool:
        for piece in self.pieces_by_color(color):
            if piece.legal_moves(self):
                return True
        return False
    
    def threatened(self, color: Color) -> List[Piece]:
        return [p for p in self.pieces_by_color(color) if self.attacked(p.pos, color.opposite())]
    
    def material(self, color: Color) -> int:
        return sum(p.value for p in self.pieces_by_color(color))
    
    def setup(self, custom: bool = False) -> None:
        if custom:
            back_row = [Rook, Noble, Bishop, Rider, King, Bishop, Prince, Rook]
        else:
            back_row = [Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook]
        
        for col, piece_class in enumerate(back_row):
            self.put(piece_class(WHITE, Position(0, col)))
            self.put(piece_class(BLACK, Position(7, col)))
        
        for col in range(8):
            self.put(Pawn(WHITE, Position(1, col)))
            self.put(Pawn(BLACK, Position(6, col)))

# ========== Игра ==========

class Game:
    PROMOTION_MAP = {
        'q': Queen, 'r': Rook, 'b': Bishop, 'n': Knight,
        'c': Prince, 'a': Noble, 'z': Rider
    }
    
    def __init__(self, custom: bool = False):
        self.board = Board()
        self.board.setup(custom)
        self.turn: Color = WHITE
        self.history: List[Move] = []
        self.over: bool = False
        self.result: str = ''
        self.halfmove: int = 0
    
    def make_move(self, piece: Piece, to: Position, promo: str = 'q') -> Move:
        board = self.board
        frm = piece.pos
        captured = board.at(to)
        
        move_data = {
            'piece': piece, 'frm': frm, 'to': to,
            'prev_moved': piece.moved, 'prev_ep': board.ep_target,
            'prev_halfmove': self.halfmove
        }
        
        # Рокировка
        if isinstance(piece, King) and abs(to.col - frm.col) == 2:
            rook_col = 7 if to.col > frm.col else 0
            rook_pos = Position(frm.row, rook_col)
            rook_to = Position(frm.row, 5 if rook_col == 7 else 3)
            rook = board.at(rook_pos)
            
            move_data.update(castling_rook=rook, rook_frm=rook_pos, 
                           rook_to=rook_to, rook_prev_moved=rook.moved)
            board.move_piece(rook, rook_to)
            rook.moved = True
        
        # Взятие на проходе
        if isinstance(piece, Pawn) and to == board.ep_target and not captured:
            ep_pos = Position(frm.row, to.col)
            captured = board.remove(ep_pos)
            move_data['ep_pos'] = ep_pos
        
        # Обычное взятие
        if captured and not move_data.get('ep_pos'):
            board.remove(to)
        
        move_data['captured'] = captured
        board.move_piece(piece, to)
        piece.moved = True
        
        # Обновление счётчика полуходов
        self.halfmove = 0 if isinstance(piece, Pawn) or captured else self.halfmove + 1
        
        # Установка цели для взятия на проходе
        board.ep_target = None
        if isinstance(piece, Pawn) and abs(to.row - frm.row) == 2:
            board.ep_target = Position((frm.row + to.row) // 2, to.col)
        
        # Превращение пешки
        if isinstance(piece, Pawn) and to.row == piece._promo_row:
            piece_class = self.PROMOTION_MAP.get(promo.lower(), Queen)
            new_piece = piece_class(piece.color, to)
            new_piece.moved = True
            board.remove(to)
            board.put(new_piece)
            move_data['promoted_to'] = new_piece
        
        self.history.append(Move(**move_data))
        self.turn = self.turn.opposite()
        self._check_end()
        return self.history[-1]
    
    def undo(self, count: int = 1) -> List[Move]:
        undone = []
        for _ in range(count):
            if not self.history:
                break
            
            move = self.history.pop()
            self.over, self.result = False, ''
            board = self.board
            
            # Откат превращения
            if move.promoted_to:
                board.remove(move.to)
                board.put(move.piece)
                move.piece.pos = move.to
            
            # Откат хода фигуры
            board.move_piece(move.piece, move.frm)
            move.piece.moved = move.prev_moved
            
            # Откат рокировки
            if move.castling_rook:
                board.move_piece(move.castling_rook, move.rook_frm)
                move.castling_rook.moved = move.rook_prev_moved
            
            # Восстановление взятой фигуры
            if move.captured:
                restore_pos = move.ep_pos or move.to
                move.captured.pos = restore_pos
                board._grid[restore_pos.row][restore_pos.col] = move.captured
                if move.captured not in board.pieces:
                    board.pieces.append(move.captured)
            
            board.ep_target = move.prev_ep
            self.halfmove = move.prev_halfmove
            self.turn = self.turn.opposite()
            undone.append(move)
        
        return undone
    
    def _check_end(self) -> None:
        if not self.board.has_moves(self.turn):
            self.over = True
            if self.board.in_check(self.turn):
                winner = 'белых' if self.turn == BLACK else 'чёрных'
                self.result = f"Мат! Победа {winner}!"
            else:
                self.result = 'Пат! Ничья.'
        elif self.halfmove >= 100:
            self.over, self.result = True, 'Ничья по правилу 50 ходов.'
        elif self._insufficient_material():
            self.over, self.result = True, 'Ничья — недостаточно материала.'
    
    def _insufficient_material(self) -> bool:
        whites = [p for p in self.board.pieces_by_color(WHITE) if not isinstance(p, King)]
        blacks = [p for p in self.board.pieces_by_color(BLACK) if not isinstance(p, King)]
        
        if not whites and not blacks:
            return True
        
        for minors, opponents in [(whites, blacks), (blacks, whites)]:
            if not opponents and len(minors) == 1 and isinstance(minors[0], (Bishop, Knight)):
                return True
        
        return (len(whites) == 1 and len(blacks) == 1 and
                isinstance(whites[0], Bishop) and isinstance(blacks[0], Bishop) and
                (whites[0].pos.row + whites[0].pos.col) % 2 == (blacks[0].pos.row + blacks[0].pos.col) % 2)

# ========== Отображение ==========

def render(board: Board, moves: Optional[Set[Position]] = None, 
          threats: Optional[List[Piece]] = None, check_king: Optional[King] = None) -> None:
    moves_set = set(moves or [])
    threats_set = {p.pos for p in (threats or [])}
    check_pos = check_king.pos if check_king else None
    
    print('\n    a  b  c  d  e  f  g  h')
    print('  +' + '---' * 8 + '+')
    
    for row in range(7, -1, -1):
        cells = []
        for col in range(8):
            pos = Position(row, col)
            piece = board.at(pos)
            
            if pos == check_pos:
                cells.append(f'!{piece.icon()}!' if piece else '!K!')
            elif pos in moves_set and piece:
                cells.append(f'[{piece.icon()}]')
            elif pos in moves_set:
                cells.append(' * ')
            elif pos in threats_set and piece:
                cells.append(f'#{piece.icon()}#')
            elif piece:
                cells.append(f' {piece.icon()} ')
            else:
                cells.append(' . ' if (row + col) % 2 == 0 else ' - ')
        
        print(f" {row + 1}|{''.join(cells)}|{row + 1}")
    
    print('  +' + '---' * 8 + '+')
    print('    a  b  c  d  e  f  g  h')
    
    material_diff = board.material(WHITE) - board.material(BLACK)
    if material_diff:
        print(f"  Материал: {'белые' if material_diff > 0 else 'чёрные'} +{abs(material_diff)}")
    print()

# ========== Приложение ==========

class ChessApp:
    def __init__(self):
        self.game: Optional[Game] = None
    
    def run(self) -> None:
        print('   ШАХМАТЫ   ')
        print(' 1 - Стандартные\n 2 - С новыми фигурами')
        choice = input(' Выбор [1]: ').strip()
        self.game = Game(custom=(choice == '2'))
        print('\nКоманды: e2 e4 | select e2 | undo [N] | '
              'threats | history | material | help | quit\n')
        self._main_loop()
    
    def _main_loop(self) -> None:
        game = self.game
        
        while not game.over:
            check_king = game.board.king(game.turn) if game.board.in_check(game.turn) else None
            threatened = game.board.threatened(game.turn)
            render(game.board, threats=threatened, check_king=check_king)
            
            if check_king:
                print('  ШАХ!')
            if threatened:
                threats_str = ', '.join(f'{p.icon()} {p.name()}({p.pos})' for p in threatened)
                print(f'  Под боем: {threats_str}')
            
            move_num = len(game.history) // 2 + 1
            cmd = input(f'  [{str(game.turn).capitalize()}, ход {move_num}]: ').strip()
            
            if not cmd:
                continue
            
            if cmd == 'quit':
                print('  Выход.')
                return
            elif cmd == 'help':
                self._show_help()
            elif cmd == 'threats':
                self._show_threats()
            elif cmd == 'history':
                self._show_history()
            elif cmd == 'material':
                self._show_material()
            elif cmd.startswith('undo'):
                self._handle_undo(cmd)
            elif cmd.startswith('select'):
                self._handle_select(cmd)
            else:
                self._handle_move(cmd)
        
        check_king = game.board.king(game.turn) if game.board.in_check(game.turn) else None
        render(game.board, check_king=check_king)
        print(f'   {game.result} ')
    
    def _show_help(self) -> None:
        print('  e2 e4 | select e2 | undo [N] | '
              'threats | history | material | quit')
    
    def _show_threats(self) -> None:
        for color in (WHITE, BLACK):
            threatened = self.game.board.threatened(color)
            name = 'Белые' if color == WHITE else 'Чёрные'
            if threatened:
                threats_str = ', '.join(f'{p.icon()} {p.name()}({p.pos})' for p in threatened)
                print(f"  {name}: {threats_str}")
            else:
                print(f"  {name}: нет угроз")
    
    def _show_history(self) -> None:
        if not self.game.history:
            print('  Пусто.')
            return
        
        for i, move in enumerate(self.game.history, 1):
            turn = 'Б' if i % 2 else 'Ч'
            print(f"  {(i + 1) // 2}.{turn} {move}")
    
    def _show_material(self) -> None:
        print(f'  Белые: {self.game.board.material(WHITE)} | '
              f'Чёрные: {self.game.board.material(BLACK)}')
    
    def _handle_undo(self, cmd: str) -> None:
        parts = cmd.split()
        count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        undone = self.game.undo(count)
        for move in undone:
            print(f'   Откат: {move}')
        if not undone:
            print('  Нечего откатывать.')
    
    def _handle_select(self, cmd: str) -> None:
        parts = cmd.split()
        if len(parts) != 2:
            print('  Формат: select e2')
            return
        
        pos = Position.parse(parts[1])
        if not pos:
            print('  Неверная клетка.')
            return
        
        piece = self.game.board.at(pos)
        if not piece:
            print('  Клетка пуста.')
            return
        if piece.color != self.game.turn:
            print('  Не ваша фигура.')
            return
        
        legal_moves = piece.legal_moves(self.game.board)
        moves_str = ', '.join(str(m) for m in legal_moves) or 'нет ходов'
        print(f"  {piece.icon()} {piece.name()}: {moves_str}")
        render(self.game.board, moves=set(legal_moves))
    
    def _handle_move(self, cmd: str) -> None:
        parts = cmd.split()
        if len(parts) < 2:
            print("  Формат: e2 e4")
            return
        
        frm = Position.parse(parts[0])
        to = Position.parse(parts[1])
        
        if not frm or not to:
            print('  Неверная нотация.')
            return
        
        piece = self.game.board.at(frm)
        if not piece:
            print(f'  На {frm} нет фигуры.')
            return
        if piece.color != self.game.turn:
            print('  Не ваша фигура.')
            return
        
        legal_moves = piece.legal_moves(self.game.board)
        if to not in legal_moves:
            moves_str = ', '.join(str(m) for m in legal_moves) or 'нет'
            print(f"  Нельзя! Можно: {moves_str}")
            return
        
        promo = 'q'
        if isinstance(piece, Pawn) and to.row == piece._promo_row:
            options = ' '.join(f"{k}={ICONS.get(k.upper(), k)}" for k in Game.PROMOTION_MAP)
            print(f'  Превращение: {options}')
            promo = input('  Выбор [q]: ').strip() or 'q'
            if promo not in Game.PROMOTION_MAP:
                promo = 'q'
        
        print(f'  {self.game.make_move(piece, to, promo)}')

# ========== Запуск ==========

if __name__ == '__main__':
    ChessApp().run()
