import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    # find the paths of all the txt files
    txt_paths = list(
        Path("../../dataset/moop/results").rglob("*.csv"))

    for txt_path in txt_paths:
        # read and parse the csv file
        df = pd.read_csv(txt_path, delimiter=';',
                         names=list(range(14)))
        data = df.dropna(axis='columns', how='all').to_numpy()

        # collect results for P-ACO
        for row in range(len(data)):
            # start indicator: Tours ACO
            if data[row, 1] == 'Tours ACO':
                scores = []
                row_increment = 2
                # collect results until nan shows up
                # nan is float while the numbers are str
                while not isinstance(data[row+row_increment, 5], float):
                    scores.append(
                        [data[row+row_increment, 5], data[row+row_increment, 6]])
                    row_increment += 1
                # output the results of a run as csv with space as indicator
                # end with /n to seperate runs
                scores = pd.DataFrame(scores).astype(float)
                scores_csv = scores.to_csv(
                    index=False, header=False, sep=' ')
                with open('results/'+txt_path.stem+'_ACO', 'a') as file:
                    file.write(scores_csv+'\n')

            # start indicator: Tours ACO
            if data[row, 1] == 'Tours VNS':
                scores = []
                row_increment = 2
                # collect results until nan shows up or the last line reached
                # nan is float while the numbers are str
                while (row+row_increment < len(data)) and (not isinstance(data[row+row_increment, 5], float)):
                    scores.append(
                        [data[row+row_increment, 5], data[row+row_increment, 6]])
                    row_increment += 1
                # output the results of a run as csv with space as indicator
                # end with /n to seperate runs
                scores = pd.DataFrame(scores).astype(float)
                scores_csv = scores.to_csv(
                    index=False, header=False, sep=' ')
                with open('results/'+txt_path.stem+'_VNS', 'a') as file:
                    file.write(scores_csv+'\n')
