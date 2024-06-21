#全ての合法手をの中から一番一致するマスが増える手を選択する
#計算量が多すぎる＆ループに陥りやすい問題がある

import numpy as np
import random
from collections import Counter

class Board:
    def __init__(self, size):
        # boardのサイズを指定
        self.size = size
        self.board = np.full(size, -1)
        self.board = np.random.randint(0, 4, size)
        self.board_goal = self.create_shuffled_goal()
    
    def move_elements(self, indices, direction):
        arr = np.copy(self.board)
        
        if direction in [2, 3]:  # 行に対して操作
            rows = set(row for row, _ in indices)
            for row in rows:
                elements = [arr[row, idx] for row_, idx in indices if row_ == row]
                remaining_elements = np.delete(arr[row], [idx for row_, idx in indices if row_ == row])
                
                if direction == 3:
                    arr[row] = np.concatenate((elements, remaining_elements))
                elif direction == 2:
                    arr[row] = np.concatenate((remaining_elements, elements))
        
        elif direction in [0, 1]:  # 列に対して操作
            arr = arr.T  # 配列を転置して列を行のように扱う
            cols = set(col for _, col in indices)
            for col in cols:
                elements = [arr[col, idx] for idx, col_ in indices if col_ == col]
                remaining_elements = np.delete(arr[col], [idx for idx, col_ in indices if col_ == col])
                
                if direction == 1:
                    arr[col] = np.concatenate((elements, remaining_elements))
                elif direction == 0:
                    arr[col] = np.concatenate((remaining_elements, elements))
            arr = arr.T  # 元の形に戻す
        else:
            raise ValueError("direction must be either 0, 1, 2, or 3.")
        
        return arr

    # 型抜きとboardが重なる部分を見つける
    def find_overlapping_indices(self, mask_size, mask_position):
        mask_rows, mask_cols = mask_size
        board_rows, board_cols = self.size

        if mask_rows > board_rows and mask_cols > board_cols:
            print("Mask is larger than the board.")
            return []

        mask_start_row = max(0, -mask_position[0])
        mask_start_col = max(0, -mask_position[1])

        mask_end_row = min(mask_rows, board_rows - mask_position[0])
        mask_end_col = min(mask_cols, board_cols - mask_position[1])

        return [(i + mask_position[0], j + mask_position[1]) for i in range(mask_start_row, mask_end_row) for j in range(mask_start_col, mask_end_col)]

    # 型抜きを適応する場所を指定する
    def apply_mask_shifts(self, mask, mask_position, dir=0):
        overlapping_indices = self.find_overlapping_indices(mask.shape, mask_position)

        valid_mask_positions = [(i, j) for i, j in overlapping_indices if mask[i - mask_position[0], j - mask_position[1]] == 1]
        self.board = self.move_elements(valid_mask_positions, direction=dir)

    # sizeで指定された範囲のboardの要素をシャッフルしたものを返す関数
    def create_shuffled_goal(self):
        elements = self.board.flatten()
        np.random.shuffle(elements)
        return elements.reshape(self.size)

    # boardとboard_goalの一致しているマスを数える関数
    def count_matching_elements(self):
        return np.sum(self.board == self.board_goal)

    # maskを引数にして、maskとboardが1マス以上重なるmask_positionをすべて求める関数
    def find_all_valid_mask_positions(self, mask):
        mask_rows, mask_cols = mask.shape
        board_rows, board_cols = self.size

        return [(row, col) for row in range(-mask_rows + 1, board_rows) for col in range(-mask_cols + 1, board_cols) 
                if any(mask[overlap[0] - row, overlap[1] - col] == 1 for overlap in self.find_overlapping_indices(mask.shape, (row, col)))]

    # 全ての組み合わせを試し、最も一致する要素が多い組み合わせを見つける関数
    def find_best_combination(self, mask):
        directions = [0, 1, 2, 3] # 0: 上詰め, 1: 下詰め, 2: 左詰め, 3: 右詰め
        valid_positions = self.find_all_valid_mask_positions(mask)
        
        best_increase = -1
        best_increase_row = -1
        best_increase_col = -1
        best_combination = None
        best_combination_row = None
        best_combination_col = None
        
        original_matching_count = self.count_matching_elements()
        original_matching_count_row, original_matching_count_col = self.count_rowcol_matching()
        
        for pos in valid_positions:
            overlapping_indices = self.find_overlapping_indices(mask.shape, pos)
            valid_mask_positions = [(i, j) for i, j in overlapping_indices if mask[i - pos[0], j - pos[1]] == 1]
            for dir in directions:
                original_board = np.copy(self.board)
                self.board = self.move_elements(valid_mask_positions, direction=dir)
                new_matching_count = self.count_matching_elements()
                new_matching_count_row, new_matching_count_col = self.count_rowcol_matching()
                increase = new_matching_count - original_matching_count
                increase_row = new_matching_count_row - original_matching_count_row
                increase_col = new_matching_count_col - original_matching_count_col
                if increase > best_increase:
                    best_increase = increase
                    best_combination = (pos, dir)
                if increase_row > best_increase_row:
                    best_increase_row = increase_row
                    best_combination_row = (pos, dir)
                if increase_col > best_increase_col:
                    best_increase_col = increase_col
                    best_combination_col = (pos, dir)
                self.board = original_board
        return best_combination, best_increase, best_combination_row, best_increase_row, best_combination_col, best_increase_col
    
    def count_rowcol_matching(self):
        num_rows, num_cols = self.board.shape
        
        # Count matching elements for each row
        matchings_row = 0
        for i in range(num_rows):
            counter1 = Counter(self.board[i, :])
            counter2 = Counter(self.board_goal[i, :])
            matching_elements = sum(min(counter1[element], counter2[element]) for element in counter1 if element in counter2)
            matchings_row += matching_elements
        
        # Count matching elements for each column
        matchings_col = 0
        for j in range(num_cols):
            counter1 = Counter(self.board[:, j])
            counter2 = Counter(self.board_goal[:, j])
            matching_elements = sum(min(counter1[element], counter2[element]) for element in counter1 if element in counter2)
            matchings_col += matching_elements
        return matchings_row, matchings_col

def teikei():
    # 配列サイズのリスト
    sizes = [2, 4]
    arrays = [np.ones((1, 1), dtype=int)]
    arrays.extend(np.ones((size, size), dtype=int) for size in sizes)
    arrays.extend(np.array([[1 if (i % 2 == 0) else 0 for _ in range(size)] for i in range(size)]) for size in sizes)
    arrays.extend(np.array([[1 if (j % 2 == 0) else 0 for j in range(size)] for _ in range(size)]) for size in sizes)
    return arrays

if __name__ == "__main__":
    board = Board((4,4))
    print("Initial Board:")
    print(board.board)
    print("Goal Board:")
    print(board.board_goal)
    masks = teikei()
    print("Matching elements:", board.count_matching_elements())
    #手数を数える
    n_count = 0

    while not np.array_equal(board.board, board.board_goal):
        best_overall_count = -1
        best_overall_count_row = -1
        best_overall_count_col = -1
        best_overall_index = -1
        best_overall_index_row = -1
        best_overall_index_col = -1
        best_overall_combination = None
        best_overall_combination_row = None
        best_overall_combination_col = None

        for index, mask in enumerate(masks):
            best_combination, best_count, best_combination_row, best_count_row, best_combination_col, best_count_col = board.find_best_combination(mask)
            if best_count > best_overall_count:
                best_overall_count = best_count
                best_overall_index = index
                best_overall_combination = best_combination
            if best_count_row > best_overall_count_row:
                best_overall_count_row = best_count_row
                best_overall_index_row = index
                best_overall_combination_row = best_combination_row
            if best_count_col > best_overall_count_col:
                best_overall_count_col = best_count_col
                best_overall_index_col = index
                best_overall_combination_col = best_combination_col
        print("best_overall_count:", best_overall_count)
        print("best_overall_count_row:", best_overall_count_row)
        print("best_overall_count_col:", best_overall_count_col)

        if best_overall_count > 0 and best_overall_count >= best_overall_count_row and best_overall_count >= best_overall_count_col:
            print(f"Applying best move with mask index {best_overall_index}, position {best_overall_combination[0]}, direction {best_overall_combination[1]}")
            board.apply_mask_shifts(masks[best_overall_index], best_overall_combination[0], dir=best_overall_combination[1])
        
        #best_increaseが0の場合、best_overall_combination_rowを選択
        elif best_overall_count < best_overall_count_row and best_overall_count_row > 0 and best_overall_count_row >= best_overall_count_col:
            print("二の策")
            print(f"Applying row best move with mask index {best_overall_index_row}, position {best_overall_combination_row[0]}, direction {best_overall_combination_row[1]}")
            board.apply_mask_shifts(masks[best_overall_index_row], best_overall_combination_row[0], dir=best_overall_combination_row[1])
        
        #best_increase,best_overall_combination_rowが0の場合、best_overall_combination_colを選択
        elif best_overall_count_col > 0:
            print("三の策")
            print(f"Applying col best move with mask index {best_overall_index_col}, position {best_overall_combination_col[0]}, direction {best_overall_combination_col[1]}")
            board.apply_mask_shifts(masks[best_overall_index_col], best_overall_combination_col[0], dir=best_overall_combination_col[1])

        elif best_overall_count <= 0 and best_overall_count_row <= 0 and best_overall_count_col <= 0:
            print("見つかんなかった!!!!!")
            choice_mask_index = np.random.randint(0, len(masks))
            valid_positions = board.find_all_valid_mask_positions(masks[choice_mask_index])
            choice_dir = random.choice([0, 1, 2, 3])
            choice_position = random.choice(valid_positions)
            print(f"Applying random move with mask index {choice_mask_index}, position {choice_position}, direction {choice_dir}")
            board.apply_mask_shifts(masks[choice_mask_index], choice_position, dir=choice_dir)
        
        print("Current Board:")
        print(board.board)
        print("Matching elements:", board.count_matching_elements())
        n_count += 1
        print("--------------------")

    print("Final Board Configuration:")
    print(board.board)
    print("かかった手数:", n_count)
    if np.array_equal(board.board, board.board_goal):
        print("Board matches the goal!")
    else:
        print("Board does not match the goal.")