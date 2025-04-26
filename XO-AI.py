import random
import pickle
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

class QLearningAgent:
    def __init__(self):
        self.q_values = defaultdict(float)
        self.alpha = 0.7
        self.gamma = 0.9
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.training_phase = 0
        
    def get_state_key(self, board):
        return tuple(board)
    
    def choose_action(self, board, available_actions, player='O'):
        current_epsilon = max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** self.training_phase))
        if random.random() < current_epsilon:
            return random.choice(available_actions)
        
        action_values = {
            a: self.q_values[self.get_state_key(self.simulate_move(board, a, player))]
            for a in available_actions
        }
        max_value = max(action_values.values())
        best_actions = [a for a, v in action_values.items() if v == max_value]
        return random.choice(best_actions) if best_actions else random.choice(available_actions)
    
    def simulate_move(self, board, action, player):
        new_board = board.copy()
        new_board[action] = player
        return new_board
    
    def update_q_value(self, old_board, action, reward, new_board):
        old_state = self.get_state_key(old_board)
        future_rewards = [
            self.q_values[self.get_state_key(self.simulate_move(new_board, a, 'O'))]
            for a in self.get_available_moves(new_board)
        ]
        max_future = max(future_rewards) if future_rewards else 0
        
        self.q_values[old_state] += self.alpha * (reward + self.gamma * max_future - self.q_values[old_state])
    
    def get_available_moves(self, board):
        return [i for i, x in enumerate(board) if x == ' ']
    
    def save_q_values(self, filename='xo_qvalues.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_values), f)
    
    def load_q_values(self, filename='xo_qvalues.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.q_values = defaultdict(float, pickle.load(f))
            print("تم تحميل الخبرات السابقة بنجاح!")
        except FileNotFoundError:
            print("بدون خبرات سابقة، سيبدأ من الصفر")

class XOGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("XO - روبوت ذكي")
        self.board = [' '] * 9
        self.current_player = 'X'
        self.agent = QLearningAgent()
        self.agent.load_q_values()
        self.training_mode = False
        self.training_games = 0
        self.after_id = None
        
        self.setup_ui()
        self.setup_progress()
    
    def setup_ui(self):
        # لوحة اللعبة
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(pady=10)
        
        self.buttons = []
        for i in range(9):
            btn = tk.Button(
                self.board_frame,
                text=' ', font=('Arial', 24), width=4, height=2,
                bg='#f0f0f0', command=lambda idx=i: self.on_click(idx)
            )
            btn.grid(row=i//3, column=i%3, padx=2, pady=2)
            self.buttons.append(btn)
        
        # أدوات التحكم
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.new_game_btn = tk.Button(
            control_frame, text="لعبة جديدة", font=('Arial', 12),
            command=self.reset_game, bg='#e1f5fe'
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = tk.Button(
            control_frame, text="بدء التدريب", font=('Arial', 12),
            command=self.start_training_dialog, bg='#e8f5e9'
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            control_frame, text="إيقاف التدريب", font=('Arial', 12),
            command=self.confirm_stop, bg='#ffebee', state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # معلومات الحالة
        self.status_label = tk.Label(
            self.root, text="دورك أنت (X)", 
            font=('Arial', 14, 'bold'), fg='#0d47a1'
        )
        self.status_label.pack(pady=5)
        
        self.stats_label = tk.Label(
            self.root, text="الألعاب: 0 | التدريب: 0",
            font=('Arial', 10), fg='#616161'
        )
        self.stats_label.pack()
    
    def setup_progress(self):
        self.progress_frame = tk.Frame(self.root)
        self.progress_frame.pack(pady=5)
        
        self.progress = ttk.Progressbar(
            self.progress_frame,
            orient='horizontal',
            length=300,
            mode='determinate'
        )
        self.progress.pack()
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="تقدم التدريب: 0%",
            font=('Arial', 10)
        )
        self.progress_label.pack()
    
    def start_training_dialog(self):
        games = simpledialog.askinteger(
            "عدد الألعاب",
            "أدخل عدد ألعاب التدريب (100-5000):",
            parent=self.root,
            minvalue=100,
            maxvalue=5000
        )
        if games:
            self.start_training(games)
    
    def start_training(self, games):
        self.training_mode = True
        self.training_games = games
        self.original_games = games
        self.update_controls()
        self.train_robot()
    
    def train_robot(self):
        if self.training_games <= 0 or not self.training_mode:
            self.stop_training()
            return
        
        self.reset_game(silent=True)
        self.process_training_move()
    
    def process_training_move(self):
        if not self.training_mode:
            return
        
        available = self.agent.get_available_moves(self.board)
        if available:
            player = 'X' if self.current_player == 'X' else 'O'
            move = self.get_training_move(player, available)
            self.make_move(move, player)
            
            winner = self.check_winner()
            if winner:
                self.handle_training_result(winner)
            else:
                self.after_id = self.root.after(10, self.process_training_move)
        else:
            self.after_id = self.root.after(10, self.train_robot)
    
    def get_training_move(self, player, available_moves):
        if player == 'X':
            return random.choice(available_moves)
        else:
            return self.agent.choose_action(self.board, available_moves, player)
    
    def handle_training_result(self, winner):
        reward = 1 if winner == 'O' else (-1 if winner == 'X' else 0)
        self.agent.update_q_value(self.board, None, reward, self.board)
        self.training_games -= 1
        self.update_progress()
        self.after_id = self.root.after(10, self.train_robot)
    
    def update_progress(self):
        progress = ((self.original_games - self.training_games) / self.original_games) * 100
        self.progress['value'] = progress
        self.progress_label.config(text=f"تقدم التدريب: {progress:.1f}%")
        if self.training_games % 50 == 0:
            self.agent.save_q_values()
    
    def confirm_stop(self):
        if messagebox.askyesno("تأكيد", "هل تريد إيقاف التدريب؟"):
            self.stop_training()
    
    def stop_training(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.training_mode = False
        self.agent.save_q_values()
        self.update_controls()
        self.reset_game()
        messagebox.showinfo("معلومات", "تم حفظ التقدم بنجاح!")
    
    def update_controls(self):
        state = tk.NORMAL if not self.training_mode else tk.DISABLED
        self.train_btn.config(state=state)
        self.stop_btn.config(state=tk.DISABLED if not self.training_mode else tk.NORMAL)
        self.new_game_btn.config(state=state)
    
    def on_click(self, pos):
        if not self.training_mode and self.board[pos] == ' ' and self.current_player == 'X':
            self.make_move(pos, 'X')
            winner = self.check_winner()
            if not winner:
                self.current_player = 'O'
                self.status_label.config(text="الروبوت يفكر...")
                self.after_id = self.root.after(500, self.agent_move)
    
    def agent_move(self):
        available = self.agent.get_available_moves(self.board)
        if available:
            move = self.agent.choose_action(self.board, available, 'O')
            self.make_move(move, 'O')
            winner = self.check_winner()
            if not winner:
                self.current_player = 'X'
                self.status_label.config(text="دورك أنت (X)")
    
    def make_move(self, pos, player):
        self.board[pos] = player
        self.buttons[pos].config(
            text=player,
            state=tk.DISABLED,
            bg='white',
            fg='#1565c0' if player == 'X' else '#c62828'
        )
    
    def check_winner(self):
        win_patterns = [
            [0,1,2], [3,4,5], [6,7,8],  # صفوف
            [0,3,6], [1,4,7], [2,5,8],  # أعمدة
            [0,4,8], [2,4,6]            # أقطار
        ]
        
        for pattern in win_patterns:
            a, b, c = pattern
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.highlight_win(pattern)
                return self.board[a]
        
        if ' ' not in self.board:
            return 'T'
        return None
    
    def highlight_win(self, pattern):
        for pos in pattern:
            self.buttons[pos].config(bg='#c8e6c9')
    
    def reset_game(self, silent=False):
        self.board = [' '] * 9
        self.current_player = 'X'
        for btn in self.buttons:
            btn.config(text=' ', state=tk.NORMAL, bg='#f0f0f0')
        if not silent:
            self.status_label.config(text="دورك أنت (X)")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.geometry("450x600")
        XOGameGUI(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"خطأ: {e}")
        print("تأكد من تثبيت مكتبة Tkinter")