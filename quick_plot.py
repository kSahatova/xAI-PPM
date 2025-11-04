import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("a.csv")

plt.figure(figsize=(10, 6))
plt.plot(df['acc position'], df['acc_train'], label='Train Accuracy')
plt.plot(df['acc position'], df['acc_test'], label='Test Accuracy')
plt.xlabel('Accuracy Position')
plt.ylabel('Accuracy')
plt.title('Accuracy per Position for Train and Test Sets')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_per_position.png")