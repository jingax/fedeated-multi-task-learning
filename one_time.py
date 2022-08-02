import pandas as pd

if __name__ == "__main__":

    n_clients = 10
    image_dir = './dataset/img_align_celeba'
    annotations_file = './dataset/annotations.txt'
    
    file1 = open(annotations_file, 'r')
    Lines = file1.readlines()
    # print(Lines[0])
    # print(Lines[1].split())
    # print(len(Lines[2].split()))
    temp = ["file_name"]+Lines[1].split()
    print(len(temp))
    dataset = pd.DataFrame(columns=temp)
    print(dataset.head())
    for idx in range(2,len(Lines)):
        line = Lines[idx].split()
        for i in range(1,len(line)):
            line[i] = max(0,int(line[i]))
        # print(idx)
        # print(len(line))    
        dataset.loc[len(dataset)] = line

    dataset.to_csv("./dataset/anno_csv.csv")     