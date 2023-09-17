import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    # find the paths of all the txt files
    txt_paths = list(
        Path("../../dataset/moop/results").rglob("*.csv"))

    for txt_path in txt_paths:
        df = pd.read_csv(txt_path, delimiter=';',
                         names=list(range(14)))
        data = df.dropna(axis='columns', how='all').to_numpy()

        for row in range(len(data)):
            if data[row, 1] == 'Tours ACO':
                scores = []
                row_increment = 2
                while not isinstance(data[row+row_increment, 5], float):
                    scores.append(
                        [data[row+row_increment, 5], data[row+row_increment, 6]])
                    row_increment += 1
                scores = pd.DataFrame(scores).astype(float)
                scores_csv = scores.to_csv(
                    index=False, header=False, sep=' ')
                with open('results/'+txt_path.stem+'_ACO', 'a') as file:
                    file.write(scores_csv+'\n')

            if data[row, 1] == 'Tours VNS':
                scores = []
                row_increment = 2
                while (row+row_increment < len(data)) and (not isinstance(data[row+row_increment, 5], float)):
                    scores.append(
                        [data[row+row_increment, 5], data[row+row_increment, 6]])
                    row_increment += 1
                scores = pd.DataFrame(scores).astype(float)
                scores_csv = scores.to_csv(
                    index=False, header=False, sep=' ')
                with open('results/'+txt_path.stem+'_VNS', 'a') as file:
                    file.write(scores_csv+'\n')
