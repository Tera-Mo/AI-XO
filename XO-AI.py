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
        self.training_strength = 1  # شدة التدريب الجديدة

    def get_state_key(self, board):
        return tuple(board)

    def choose_action(self, board, available_actions, player='O'):
        current_epsilon = max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** self.training_phase))

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
        self.root.configure(bg="#e3f2fd")
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'
        self.agent = QLearningAgent()
        self.agent.load_q_values()
        self.game_count = 0
        self.training_mode = False
        self.training_games = 0
        self.stop_training_flag = False

        self.setup_ui()

    def setup_ui(self):
        self.board_frame = tk.Frame(self.root, bg="#e3f2fd")
        self.board_frame.pack(pady=20)

        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.board_frame, text=' ', font=('Arial', 26, 'bold'), width=5, height=2,
                            bg='white', command=lambda idx=i: self.on_click(idx))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(btn)

        control_frame = tk.Frame(self.root, bg="#e3f2fd")
        control_frame.pack(pady=10)

        tk.Button(control_frame, text="لعبة جديدة", font=('Arial', 12), command=self.reset_game,
                  bg="#81d4fa").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="بدء التدريب", font=('Arial', 12), command=self.start_training_dialog,
                  bg="#a5d6a7").pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(control_frame, text="إيقاف التدريب", font=('Arial', 12), command=self.stop_training,
                                  bg="#ef9a9a", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text="دورك أنت (X)", font=('Arial', 16, 'bold'), bg="#e3f2fd", fg="#1565c0")
        self.status_label.pack(pady=10)

    def start_training_dialog(self):
        if self.training_mode:
            return

        games = simpledialog.askinteger("عدد ألعاب التدريب", "كم عدد الألعاب التي تريد تدريب الروبوت عليها؟ (100-10000)", minvalue=100, maxvalue=10000)
        strength = simpledialog.askfloat("شدة التدريب", "اختر شدة التدريب (1 = عادي, أعلى = تعلم أسرع)", minvalue=0.5, maxvalue=10.0)

        if games:
            self.agent.training_strength = strength or 1
            self.start_training(games)

    def start_training(self, games):
        self.training_mode = True
        self.training_games = games
        self.stop_training_flag = False
        self.stop_btn.config(state=tk.NORMAL)
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
        self.status_label.config(text="جاري التدريب...")
        self.root.after(10, self.train_robot)

    def stop_training(self):
        self.stop_training_flag = True
        self.training_mode = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="تم إيقاف التدريب.")
        self.reset_game(silent=True)

    def train_robot(self):
        if self.stop_training_flag or self.training_games <= 0:
            self.stop_training()
            self.agent.save_q_values()
            return

        self.reset_game(silent=True)
        self.training_move()

    def training_move(self):
        if self.stop_training_flag:
            return

        available = self.agent.get_available_moves(self.board)

        if available:
            move = self.agent.choose_action(self.board, available, 'O')
            self.make_move(move, 'O')

            winner = self.check_winner()
            if not winner:
                move = random.choice(self.agent.get_available_moves(self.board))
                self.make_move(move, 'X')
                winner = self.check_winner()

        self.training_games -= 1
        self.agent.training_phase += int(1 * self.agent.training_strength)

        if self.training_games > 0:
            self.root.after(1, self.train_robot)
        else:
            self.stop_training()

    def on_click(self, idx):
        if self.training_mode or self.board[idx] != ' ':
            return

        self.make_move(idx, 'X')
        winner = self.check_winner()
        if not winner:
            self.status_label.config(text="الروبوت يفكر...")
            self.root.after(700, self.robot_move)

    def robot_move(self):
        if self.training_mode:
            return

        available = self.agent.get_available_moves(self.board)
        if available:
            move = self.agent.choose_action(self.board, available, 'O')
            self.make_move(move, 'O')
            self.check_winner()

    def make_move(self, pos, player):
        self.board[pos] = player
        self.buttons[pos].config(text=player, state=tk.DISABLED,
                                 fg='#1565c0' if player == 'X' else '#c62828')

    def check_winner(self):
        lines = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.highlight_winner([a,b,c])
                self.show_result(self.board[a])
                return self.board[a]
        if ' ' not in self.board:
            self.show_result('T')
            return 'T'
        return None

    def highlight_winner(self, positions):
        for pos in positions:
            self.buttons[pos].config(bg="#a5d6a7")

    def show_result(self, winner):
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
        msg = "تعادل!" if winner == 'T' else f"{'أنت' if winner == 'X' else 'الروبوت'} فاز!"
        self.status_label.config(text=msg)
        if not self.training_mode:
            messagebox.showinfo("نتيجة اللعبة", msg)

    def reset_game(self, silent=False):
        self.board = [' '] * 9
        for btn in self.buttons:
            btn.config(text=' ', state=tk.NORMAL, bg='white')
        if not silent:
            self.status_label.config(text="دورك أنت (X)")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x600")
    app = XOGameGUI(root)
    root.mainloop()