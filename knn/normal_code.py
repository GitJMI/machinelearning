import csv
import math

# Step 1: Load CSV file
def load_data(filename):
    dataset = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # skip header

        for row in csv_reader:
            # Convert categorical to numeric
            age = int(row[1])
            income = int(row[2])
            gender = 0 if row[3] == "Male" else 1
            region = 0 if row[4] == "Urban" else 1
            label = 1 if row[5] == "Yes" else 0

            dataset.append([age, income, gender, region, label])

    return dataset


# Step 2: Distance Function
def euclidean_distance(p1, p2):
    distance = 0
    for i in range(len(p1) - 1):  # ignore label
        distance += (p1[i] - p2[i]) ** 2
    return math.sqrt(distance)


# Step 3: KNN Algorithm
def knn(train_data, test_point, k):
    distances = []

    # Calculate distances
    for row in train_data:
        dist = euclidean_distance(row, test_point)
        distances.append((dist, row[-1]))

    # Sort distances
    distances.sort(key=lambda x: x[0])

    # Take K nearest
    neighbors = distances[:k]

    # Voting
    count_yes = 0
    count_no = 0

    for n in neighbors:
        if n[1] == 1:
            count_yes += 1
        else:
            count_no += 1

    return "Yes" if count_yes > count_no else "No"


# Step 4: Main Program
data = load_data("data.csv")

# Example test point
test = [32, 38000, 1, 1]  # Age, Income, Female, Rural

result = knn(data, test, 3)
print("Prediction:", result)