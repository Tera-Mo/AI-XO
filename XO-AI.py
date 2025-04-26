import random
import pickle
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox, simpledialog

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
        current_epsilon = max(self.min_epsilon, 
                            self.epsilon * (self.epsilon_decay ** self.training_phase))
        
        if random.random() < current_epsilon:
            return random.choice(available_actions)
        
        action_values = {}
        for action in available_actions:
            new_board = board.copy()
            new_board[action] = player
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
        learning_rate = self.alpha * (1.0 - 0.9 * (self.training_phase / 1000))
        new_q = current_q + learning_rate * (reward + self.gamma * max_future_value - current_q)
        self.q_values[old_state_key] = new_q
    
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
            print("بدون خبرة سابقة، سيبدأ الروبوت من الصفر")

class XOGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("XO - روبوت متطور")
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.agent = QLearningAgent()
        self.agent.load_q_values()
        self.game_count = 0
        self.training_mode = False
        self.training_games = 0
        self.stop_training = False
        
        # تعريف development_stages قبل استدعاء setup_ui
        self.development_stages = [
            {"games": 100, "desc": "المرحلة 1: تعلم أساسيات الفوز (100 لعبة)"},
            {"games": 300, "desc": "المرحلة 2: تحسين الاستراتيجيات (300 لعبة)"},
            {"games": 600, "desc": "المرحلة 3: تعلم الدفاع ضد الهجمات (600 لعبة)"},
            {"games": 1000, "desc": "المرحلة 4: إتقان اللعبة (1000 لعبة)"}
        ]
        self.current_stage = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        # إطار اللوحة
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
        
        # إطار التحكم
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        
        tk.Button(
            ctrl_frame, text="لعبة جديدة", font=('Arial', 12),
            command=self.reset_game, bg='#e1f5fe'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            ctrl_frame, text="بدء التدريب", font=('Arial', 12),
            command=self.start_training_dialog, bg='#e8f5e9'
        ).pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            ctrl_frame, text="إيقاف التدريب", font=('Arial', 12),
            command=self.stop_training, bg='#ffebee', state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # إطار المعلومات
        info_frame = tk.Frame(self.root)
        info_frame.pack(pady=10)
        
        self.status_label = tk.Label(
            info_frame, text="دورك أنت (X)", 
            font=('Arial', 14, 'bold'), fg='#0d47a1'
        )
        self.status_label.pack()
        
        self.stats_label = tk.Label(
            info_frame, text="الألعاب: 0 | التدريب: 0/0",
            font=('Arial', 10), fg='#616161'
        )
        self.stats_label.pack()
        
        # خطة التطوير
        dev_frame = tk.LabelFrame(self.root, text="خطة تطوير الروبوت", font=('Arial', 10))
        dev_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.dev_text = tk.Text(
            dev_frame, height=6, font=('Arial', 9),
            bg='#fafafa', wrap=tk.WORD
        )
        self.dev_text.pack(fill=tk.X, padx=5, pady=5)
        self.update_development_plan()
    
    def update_development_plan(self):
        self.dev_text.config(state=tk.NORMAL)
        self.dev_text.delete(1.0, tk.END)
        total_games = sum(s["games"] for s in self.development_stages)
        progress = min(100, int(self.agent.training_phase / total_games * 100))
        
        self.dev_text.insert(tk.END, f"تقدم التدريب: {progress}%\n\n")
        
        for i, stage in enumerate(self.development_stages):
            prefix = "✓ " if self.agent.training_phase >= sum(s["games"] for s in self.development_stages[:i+1]) else "◌ "
            self.dev_text.insert(tk.END, prefix + stage["desc"] + "\n")
        
        self.dev_text.config(state=tk.DISABLED)
    
    def start_training_dialog(self):
        games = simpledialog.askinteger(
            "عدد ألعاب التدريب",
            "أدخل عدد ألعاب التدريب (100-10000):",
            parent=self.root,
            minvalue=100,
            maxvalue=10000
        )
        if games:
            self.start_training(games)
    
    def start_training(self, games):
        if not self.training_mode:
            self.training_mode = True
            self.training_games = games
            self.stop_training = False
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"جاري التدريب ({games} لعبة)...")
            self.train_robot()
    
    def stop_training(self):
        self.stop_training = True
        self.training_mode = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="تم إيقاف التدريب")
        self.update_development_plan()
    
    def train_robot(self):
        if self.stop_training or self.training_games <= 0:
            self.stop_training()
            return
        
        self.reset_game(silent=True)
        self.training_move()
    
    def training_move(self):
        if self.stop_training:
            return
        
        available = [i for i, x in enumerate(self.board) if x == ' ']
        
        if available:
            if self.current_player == 'X':
                if random.random() < 0.7 - (self.agent.training_phase * 0.0005):
                    move = random.choice(available)
                else:
                    move = self.agent.choose_action(self.board, available, 'X')
                self.make_move(move, 'X')
            else:
                self.agent.last_board = self.board.copy()
                move = self.agent.choose_action(self.board, available, 'O')
                self.agent.last_action = move
                self.make_move(move, 'O')
            
            winner = self.check_winner()
            if not winner:
                self.root.after(10, self.training_move)
            else:
                self.update_training(winner)
                self.training_games -= 1
                self.stats_label.config(text=f"الألعاب: {self.game_count} | التدريب: {self.training_games}")
                self.root.after(10, self.train_robot)
    
    def update_training(self, winner):
        if hasattr(self.agent, 'last_board') and hasattr(self.agent, 'last_action'):
            reward = 0
            if winner == 'O':
                reward = 1
                self.agent.training_phase += 2
            elif winner == 'X':
                reward = -1
                self.agent.training_phase += 1
            else:
                reward = 0.2
                self.agent.training_phase += 1
            
            self.agent.update_q_value(
                self.agent.last_board,
                self.agent.last_action,
                reward,
                self.board
            )
            self.agent.save_q_values()
            self.update_development_plan()
    
    def on_click(self, pos):
        if self.training_mode:
            return
            
        if self.board[pos] == ' ' and self.current_player == 'X':
            self.make_move(pos, 'X')
            winner = self.check_winner()
            if not winner:
                self.current_player = 'O'
                self.status_label.config(text="الروبوت يفكر...")
                self.root.after(500, self.agent_move)
    
    def agent_move(self):
        available = [i for i, x in enumerate(self.board) if x == ' ']
        if available:
            self.agent.last_board = self.board.copy()
            move = self.agent.choose_action(self.board, available, 'O')
            self.agent.last_action = move
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
        lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],
                [1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        
        for a,b,c in lines:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.highlight_winner(a,b,c)
                self.game_over(self.board[a])
                return self.board[a]
        
        if ' ' not in self.board:
            self.game_over('T')
            return 'T'
        return None
    
    def highlight_winner(self, a, b, c):
        for pos in [a,b,c]:
            self.buttons[pos].config(bg='#c8e6c9')
    
    def game_over(self, winner):
        self.game_count += 1
        self.stats_label.config(text=f"الألعاب: {self.game_count} | التدريب: {self.training_games}")
        
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
        
        if not self.training_mode:
            msg = "تعادل!" if winner == 'T' else f"{'أنت' if winner == 'X' else 'الروبوت'} فاز!"
            self.status_label.config(text=msg, fg='#2e7d32' if winner == 'T' else ('#1565c0' if winner == 'X' else '#c62828'))
            messagebox.showinfo("نهاية اللعبة", msg)
    
    def reset_game(self, silent=False):
        self.board = [' ']*9
        self.current_player = 'X'
        
        for i, btn in enumerate(self.buttons):
            btn.config(text=' ', state=tk.NORMAL, bg='#f0f0f0')
        
        if not silent and not self.training_mode:
            self.status_label.config(text="دورك أنت (X)", fg='#1565c0')

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.geometry("450x650")
        game = XOGameGUI(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"خطأ في تحميل الواجهة الرسومية: {e}")
        print("تأكد من أن لديك مكتبة tkinter مثبتة")
        print("على ويندوز: python -m pip install tk")
        print("على لينكس: sudo apt-get install python3-tk")