import random
import pickle
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox

class QLearningAgent:
    def __init__(self):
        self.q_values = defaultdict(float)
        self.alpha = 0.7  # زيادة معدل التعلم
        self.gamma = 0.9
        self.epsilon = 0.3  # زيادة نسبة الاستكشاف في البداية
        self.epsilon_decay = 0.995  # تقليل الاستكشاف مع الوقت
        self.min_epsilon = 0.1
        
    def get_state_key(self, board):
        return tuple(board)
    
    def choose_action(self, board, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # اختيار أفضل حركة بناء على Q-values
        action_values = {}
        for action in available_actions:
            new_board = board.copy()
            new_board[action] = 'O'
            state_key = self.get_state_key(new_board)
            action_values[action] = self.q_values.get(state_key, 0)
        
        max_value = max(action_values.values())
        best_actions = [k for k, v in action_values.items() if v == max_value]
        return random.choice(best_actions) if best_actions else random.choice(available_actions)
    
    def update_q_value(self, old_board, action, reward, new_board):
        old_state_key = self.get_state_key(old_board)
        new_state_key = self.get_state_key(new_board)
        
        max_future_value = max(
            [self.q_values.get(self.get_state_key(self.make_move(new_board.copy(), a, 'O')), 0) 
             for a in self.get_available_moves(new_board)] or [0]
        )
        
        current_q = self.q_values.get(old_state_key, 0)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_value - current_q)
        self.q_values[old_state_key] = new_q
        
        # تقليل نسبة الاستكشاف تدريجياً
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def make_move(self, board, position, player):
        board[position] = player
        return board
    
    def get_available_moves(self, board):
        return [i for i, x in enumerate(board) if x == ' ']
    
    def save_q_values(self, filename='xo_qvalues.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_values), f)
    
    def load_q_values(self, filename='xo_qvalues.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.q_values = defaultdict(float, pickle.load(f))
            print("تم تحميل خبرات الروبوت السابقة بنجاح!")
        except (FileNotFoundError, EOFError):
            print("لم يتم العثور على ملف التعلم السابق، سيبدأ الروبوت من الصفر")

class XOGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("لعبة XO - الروبوت المتعلم")
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.agent = QLearningAgent()
        self.agent.load_q_values()
        self.last_agent_board = None
        self.last_agent_action = None
        self.game_count = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(pady=20)
        
        self.buttons = []
        for i in range(9):
            button = tk.Button(
                self.board_frame,
                text=' ',
                font=('Arial', 30, 'bold'),
                width=4,
                height=2,
                bg='lightgray',
                command=lambda idx=i: self.on_button_click(idx)
            )
            button.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(button)
        
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        self.reset_button = tk.Button(
            self.control_frame,
            text="لعبة جديدة",
            font=('Arial', 14),
            command=self.reset_game
        )
        self.reset_button.pack(side=tk.LEFT, padx=10)
        
        self.stats_label = tk.Label(
            self.control_frame,
            text="عدد الألعاب: 0",
            font=('Arial', 12)
        )
        self.stats_label.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(
            self.root,
            text="دورك أنت (X)",
            font=('Arial', 14, 'bold'),
            fg='blue'
        )
        self.status_label.pack(pady=10)
    
    def on_button_click(self, position):
        if self.board[position] == ' ' and self.current_player == 'X':
            self.make_move(position, 'X')
            
            winner = self.check_winner()
            if not winner:
                self.current_player = 'O'
                self.status_label.config(text="الروبوت يفكر...")
                self.root.after(500, self.agent_move)
    
    def agent_move(self):
        available_moves = [i for i, x in enumerate(self.board) if x == ' ']
        
        if available_moves:
            self.last_agent_board = self.board.copy()
            move = self.agent.choose_action(self.board, available_moves)
            self.last_agent_action = move
            self.make_move(move, 'O')
            
            winner = self.check_winner()
            if not winner:
                self.current_player = 'X'
                self.status_label.config(text="دورك أنت (X)")
    
    def make_move(self, position, player):
        self.board[position] = player
        self.buttons[position].config(
            text=player,
            state=tk.DISABLED,
            bg='white',
            fg='blue' if player == 'X' else 'red'
        )
    
    def check_winner(self):
        win_lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for line in win_lines:
            a, b, c = line
            if self.board[a] != ' ' and self.board[a] == self.board[b] == self.board[c]:
                self.highlight_winning_line(a, b, c)
                self.game_over(self.board[a])
                return self.board[a]
        
        if ' ' not in self.board:
            self.game_over('T')
            return 'T'
        
        return None
    
    def highlight_winning_line(self, a, b, c):
        for pos in [a, b, c]:
            self.buttons[pos].config(bg='lightgreen')
    
    def game_over(self, winner):
        self.game_count += 1
        self.stats_label.config(text=f"عدد الألعاب: {self.game_count}")
        
        for button in self.buttons:
            button.config(state=tk.DISABLED)
        
        if winner == 'T':
            message = "تعادل!"
            reward = 0.1
        else:
            message = f"{'أنت' if winner == 'X' else 'الروبوت'} فاز!"
            reward = 1 if winner == 'O' else -1
        
        if self.last_agent_board and self.last_agent_action is not None:
            self.agent.update_q_value(
                self.last_agent_board,
                self.last_agent_action,
                reward,
                self.board
            )
            self.agent.save_q_values()
        
        self.status_label.config(
            text=message,
            fg='green' if winner == 'T' else ('blue' if winner == 'X' else 'red')
        )
        messagebox.showinfo("نهاية اللعبة", message)
    
    def reset_game(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.last_agent_board = None
        self.last_agent_action = None
        
        for i, button in enumerate(self.buttons):
            button.config(
                text=' ',
                state=tk.NORMAL,
                bg='lightgray',
                fg='black'
            )
        
        self.status_label.config(
            text="دورك أنت (X)",
            fg='blue'
        )

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.geometry("450x550")
        game = XOGameGUI(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"خطأ في تحميل الواجهة الرسومية: {e}")
        print("تأكد من أن لديك مكتبة tkinter مثبتة")
        print("على لينكس/أوبونتو، جرب: sudo apt-get install python3-tk")